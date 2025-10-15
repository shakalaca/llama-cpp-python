import types
import numpy as np
import pytest

from llama_cpp.llama import Llama

MODEL = "./vendor/llama.cpp/models/ggml-vocab-llama-spm.gguf"


class FakeCache:
    def __init__(self):
        self.store = {}

    def __setitem__(self, key, value):
        # store value but keep key for reconstructing state
        self.store[tuple(key)] = value

    def __getitem__(self, key):
        k = tuple(key)
        # emulate longest prefix hit: pick longest stored key starting with k
        candidates = [sk for sk in self.store.keys() if sk[: len(k)] == k]
        if not candidates:
            raise KeyError
        best = max(candidates, key=len)
        # return an object that has input_ids matching the stored key
        return types.SimpleNamespace(
            input_ids=np.array(best, dtype=np.intc),
            scores=None,
            n_tokens=len(best),
            llama_state=b"x",
            llama_state_size=1,
            seed=0,
        )

    def __contains__(self, key):
        k = tuple(key)
        return any(sk[: len(k)] == k for sk in self.store.keys())


def make_fake_llm():
    return Llama(MODEL, vocab_only=True, n_ctx=64, n_batch=64, n_ubatch=64, logits_all=True, verbose=False)


def test_cache_prefix_hit_and_store():
    llm = make_fake_llm()
    llm.set_cache(FakeCache())

    # Stub save/load_state to avoid touching C state
    def fake_save_state():
        return types.SimpleNamespace(scores=None, input_ids=llm.input_ids.copy(), n_tokens=llm.n_tokens, llama_state=b"x", llama_state_size=1, seed=llm._seed)

    called = {"loaded": False}

    def fake_load_state(state):
        called["loaded"] = True

    llm.save_state = fake_save_state  # type: ignore
    llm.load_state = fake_load_state  # type: ignore

    # Force generate to produce fixed tokens, and ensure detokenize maps to letters
    fixed = [111, 222, 333]
    llm.detokenize = lambda tokens, prev_tokens=None, special=False: b"A" * len(tokens)  # type: ignore
    llm.generate = types.MethodType(lambda self, *a, **k: (t for t in fixed), llm)  # type: ignore

    # First call stores cache under prompt+completion key
    out1 = llm.create_completion("hello", max_tokens=len(fixed))
    assert out1["choices"][0]["text"] == "AAA"

    # Second call should find prefix key and call load_state
    called["loaded"] = False
    out2 = llm.create_completion("hello", max_tokens=len(fixed))
    assert out2["choices"][0]["text"] == "AAA"
    assert called["loaded"]
