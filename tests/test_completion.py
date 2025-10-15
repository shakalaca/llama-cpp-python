import types
import numpy as np
import pytest

import llama_cpp
from llama_cpp.llama import Llama

MODEL = "./vendor/llama.cpp/models/ggml-vocab-llama-spm.gguf"


def make_fake_llm(logits_all=True):
    llm = Llama(
        MODEL,
        vocab_only=True,
        n_ctx=64,
        n_batch=32,
        n_ubatch=32,
        logits_all=logits_all,
        verbose=False,
    )
    # small deterministic seed
    llm.set_seed(42)
    return llm


def install_fake_detokenize(llm, mapping):
    def fake_detokenize(tokens, prev_tokens=None, special=False):
        return b"".join(mapping.get(int(t), b"?") for t in tokens)
    llm.detokenize = fake_detokenize  # type: ignore


def install_fake_generate(llm, seq):
    def fake_generate(tokens, **kwargs):
        for t in seq:
            yield t
    llm.generate = types.MethodType(lambda self, *a, **k: fake_generate(*a, **k), llm)  # type: ignore


def install_uniform_scores(llm, preferred_tokens):
    # Fill scores so that preferred tokens dominate
    rows = llm.scores.shape[0]
    cols = llm.scores.shape[1]
    llm.scores[:] = -100.0
    for r in range(rows):
        for idx, tok in enumerate(preferred_tokens):
            llm.scores[r, int(tok)] = 10.0 - idx


def test_create_completion_stops_and_finish_reason():
    llm = make_fake_llm(logits_all=True)
    # tokens mapping to ASCII
    T = {
        100: b"A",
        101: b"B",
        102: b"C",
        200: b"S",
        201: b"T",
        202: b"O",
        203: b"P",
        204: b"X",
    }
    install_fake_detokenize(llm, T)
    # sequence yields: ABCSTOPX
    seq = [100, 101, 102, 200, 201, 202, 203, 204]
    install_fake_generate(llm, seq)
    install_uniform_scores(llm, seq)

    out = llm.create_completion("prompt", max_tokens=len(seq), stop=["STOP"], temperature=0.0)
    text = out["choices"][0]["text"]
    assert text == "ABC"  # stopped before STOP
    assert out["choices"][0]["finish_reason"] == "stop"


def test_streaming_yields_and_concat():
    llm = make_fake_llm(logits_all=True)
    T = {100: b"H", 101: b"i", 102: b"!"}
    seq = [100, 101, 102]
    install_fake_detokenize(llm, T)
    install_fake_generate(llm, seq)
    install_uniform_scores(llm, seq)

    chunks = list(llm.create_completion("hi", max_tokens=len(seq), stream=True, temperature=0.0))
    # concatenate streamed tokens
    streamed = "".join(c["choices"][0]["text"] for c in chunks if c["choices"][0]["finish_reason"] is None)
    assert streamed == "Hi!"
    # last chunk has finish_reason
    assert chunks[-1]["choices"][0]["finish_reason"] in {"length", "stop"}


def test_call_equals_create_completion():
    llm = make_fake_llm(logits_all=True)
    T = {100: b"O", 101: b"K"}
    seq = [100, 101]
    install_fake_detokenize(llm, T)
    install_fake_generate(llm, seq)
    install_uniform_scores(llm, seq)

    a = llm.create_completion("q", max_tokens=len(seq), temperature=0.0)
    b = llm("q", max_tokens=len(seq), temperature=0.0)
    assert a["choices"][0]["text"] == b["choices"][0]["text"]
