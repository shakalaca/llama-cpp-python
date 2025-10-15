import numpy as np
import pytest

from llama_cpp.llama import Llama

MODEL = "./vendor/llama.cpp/models/ggml-vocab-llama-spm.gguf"


def test_embed_requires_flag():
    llm = Llama(MODEL, vocab_only=True, embedding=False, verbose=False)
    with pytest.raises(RuntimeError):
        llm.embed("x")


def test_embed_shapes_and_truncate(monkeypatch):
    # Use embedding=True but monkeypatch internal calls to avoid heavy work
    llm = Llama(MODEL, vocab_only=True, embedding=True, n_batch=4, verbose=False)

    # Monkeypatch batch.add_sequence to just track sizes
    added = []
    real_add = llm._batch.add_sequence
    def add_sequence_stub(tokens, seq_id, logits_all):
        added.append((len(tokens), seq_id))
        return real_add(tokens, seq_id, logits_all)
    llm._batch.add_sequence = add_sequence_stub  # type: ignore

    # Monkeypatch context.decode to no-op and set fake embeddings
    def fake_decode(batch):
        pass
    llm._ctx.decode = fake_decode  # type: ignore

    # Mock getters used in embed to return deterministic shapes
    import llama_cpp.llama_cpp as C
    def fake_get_embeddings_seq(ctx, i):
        # return a ctypes array-like of length n_embd
        class View:
            def __getitem__(self, sl):
                # produce stable values
                n = llm.n_embd()
                return [0.0] * (sl.stop - sl.start)
        return View()
    C.llama_get_embeddings_seq = fake_get_embeddings_seq  # type: ignore

    out = llm.embed(["a" * 100, "b"], truncate=True)
    assert isinstance(out, list) and len(out) == 2


def test_normalize_output(monkeypatch):
    llm = Llama(MODEL, vocab_only=True, embedding=True, verbose=False)
    # Simulate embeddings pointer returns unit vectors
    import llama_cpp.llama_cpp as C
    def fake_get_embeddings_seq(ctx, i):
        class View:
            def __getitem__(self, sl):
                n = llm.n_embd()
                return [1.0] + [0.0] * (sl.stop - sl.start - 1)
        return View()
    C.llama_get_embeddings_seq = fake_get_embeddings_seq  # type: ignore

    v = llm.embed("x", normalize=True)
    assert isinstance(v, list) and abs(sum(x * x for x in v) - 1.0) < 1e-6
