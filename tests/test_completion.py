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
        prompt_len = len(tokens)
        llm.input_ids[:prompt_len] = np.array(tokens, dtype=np.intc)
        llm.scores[:prompt_len, :] = 0.0
        llm.n_tokens = prompt_len
        for t in seq:
            llm.input_ids[llm.n_tokens] = int(t)
            llm.scores[llm.n_tokens, :] = 0.0
            llm.n_tokens += 1
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
    assert chunks[-1]["choices"][0]["text"] == ""


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


def test_completion_multiple_stops_pick_first_match():
    llm = make_fake_llm(logits_all=True)
    T = {100: b"H", 101: b"E", 102: b"L", 103: b"O"}
    seq = [100, 101, 102, 102, 103]
    install_fake_detokenize(llm, T)
    install_fake_generate(llm, seq)
    install_uniform_scores(llm, seq)

    out = llm.create_completion(
        "hi",
        max_tokens=len(seq),
        stop=["XYZ", "LO"],
        temperature=0.0,
    )
    text = out["choices"][0]["text"]
    assert text == "HEL"
    assert out["choices"][0]["finish_reason"] == "stop"


def test_completion_logit_bias_adjusts_scores():
    llm = make_fake_llm(logits_all=True)
    preferred = 120
    T = {preferred: b"Z"}
    install_fake_detokenize(llm, T)

    captured = {}

    def fake_generate(tokens, **kwargs):
        captured["kwargs"] = kwargs
        yield preferred

    llm.generate = types.MethodType(lambda self, *a, **k: fake_generate(*a, **k), llm)  # type: ignore

    out = llm.create_completion(
        "prompt",
        max_tokens=1,
        logit_bias={preferred: 5.0},
        temperature=0.0,
    )

    assert out["choices"][0]["text"] == "Z"
    logits_processor = captured["kwargs"]["logits_processor"]
    assert isinstance(logits_processor, llama_cpp.LogitsProcessorList)

    base = np.zeros(llm.n_vocab(), dtype=np.float32)
    adjusted = logits_processor(np.array([0], dtype=np.intc), base)
    assert pytest.approx(5.0) == float(adjusted[preferred])
    assert np.count_nonzero(np.abs(adjusted) > 1e-6) == 1


def test_completion_passes_temperature_to_generator():
    llm = make_fake_llm(logits_all=True)
    token = 130
    T = {token: b"T"}
    install_fake_detokenize(llm, T)

    recorded = {}

    def fake_generate(tokens, **kwargs):
        recorded["temp"] = kwargs.get("temp")
        yield token

    llm.generate = types.MethodType(lambda self, *a, **k: fake_generate(*a, **k), llm)  # type: ignore

    out = llm.create_completion("prompt", max_tokens=1, temperature=0.0)

    assert out["choices"][0]["text"] == "T"
    assert recorded["temp"] == 0.0


def test_completion_logprobs_requires_logits_all():
    llm = make_fake_llm(logits_all=False)
    T = {100: b"A"}
    seq = [100]
    install_fake_detokenize(llm, T)
    install_fake_generate(llm, seq)
    install_uniform_scores(llm, seq)

    with pytest.raises(ValueError):
        llm.create_completion("prompt", max_tokens=1, logprobs=1, temperature=0.0)


def test_completion_streaming_logprobs_include_tokens():
    llm = make_fake_llm(logits_all=True)
    T = {100: b"H", 101: b"i"}
    seq = [100, 101]
    install_fake_detokenize(llm, T)
    install_fake_generate(llm, seq)
    install_uniform_scores(llm, seq)
    llm.scores[:] = 0.0

    chunks = list(
        llm.create_completion(
            "hi",
            max_tokens=len(seq),
            stream=True,
            logprobs=2,
            temperature=0.0,
        )
    )

    partial = [c for c in chunks if c["choices"][0]["finish_reason"] is None]
    assert partial, "expected streaming payloads before final chunk"
    logprobs = partial[0]["choices"][0]["logprobs"]
    assert logprobs is not None
    assert logprobs["tokens"]
    assert isinstance(logprobs["top_logprobs"][0], dict)


def test_completion_preserves_user_logits_processor():
    llm = make_fake_llm(logits_all=True)
    token = 140
    T = {token: b"Y"}
    install_fake_detokenize(llm, T)

    invoked = {}

    def fake_generate(tokens, **kwargs):
        invoked["passed"] = kwargs.get("logits_processor")
        yield token

    llm.generate = types.MethodType(lambda self, *a, **k: fake_generate(*a, **k), llm)  # type: ignore

    lp = llama_cpp.LogitsProcessorList([lambda input_ids, scores: scores])

    out = llm.create_completion(
        "prompt",
        max_tokens=1,
        logits_processor=lp,
        temperature=0.0,
    )

    assert out["choices"][0]["text"] == "Y"
    assert invoked["passed"] is lp


def test_completion_combines_user_processor_and_logit_bias():
    llm = make_fake_llm(logits_all=True)
    token = 150
    T = {token: b"Z"}
    install_fake_detokenize(llm, T)

    invoked = {}

    def fake_generate(tokens, **kwargs):
        lp = kwargs.get("logits_processor")
        base = np.zeros(llm.n_vocab(), dtype=np.float32)
        processed = lp(np.array(tokens, dtype=np.intc), base)
        invoked["scores"] = processed
        invoked["lp"] = lp
        yield token

    llm.generate = types.MethodType(lambda self, *a, **k: fake_generate(*a, **k), llm)  # type: ignore

    def user_processor(input_ids, scores):
        updated = np.copy(scores)
        updated[token] = 1.0
        return updated

    lp = llama_cpp.LogitsProcessorList([user_processor])

    out = llm.create_completion(
        "prompt",
        max_tokens=1,
        temperature=0.0,
        logits_processor=lp,
        logit_bias={token: 2.0},
    )

    assert out["choices"][0]["text"] == "Z"
    assert invoked["lp"] is lp
    assert len(lp) == 2
    assert invoked["scores"][token] == pytest.approx(3.0)


def install_recording_generate(llm, seq, recorder):
    def fake_generate(tokens, **kwargs):
        recorder["prompt_tokens"] = list(tokens)
        prompt_len = len(tokens)
        llm.input_ids[:prompt_len] = np.array(tokens, dtype=np.intc)
        llm.scores[:prompt_len, :] = 0.0
        llm.n_tokens = prompt_len
        for t in seq:
            llm.input_ids[llm.n_tokens] = int(t)
            llm.scores[llm.n_tokens, :] = 0.0
            llm.n_tokens += 1
            yield t

    llm.generate = types.MethodType(lambda self, *a, **k: fake_generate(*a, **k), llm)  # type: ignore


def test_completion_suffix_modifies_prompt_without_leaking():
    token = 160
    T = {token: b"Q"}
    seq = [token]

    plain = make_fake_llm(logits_all=True)
    install_fake_detokenize(plain, T)
    install_uniform_scores(plain, seq)
    recorded_plain = {}
    install_recording_generate(plain, seq, recorded_plain)
    out_plain = plain.create_completion("seed", max_tokens=1, temperature=0.0)

    with_suffix = make_fake_llm(logits_all=True)
    install_fake_detokenize(with_suffix, T)
    install_uniform_scores(with_suffix, seq)
    recorded_suffix = {}
    install_recording_generate(with_suffix, seq, recorded_suffix)
    out_suffix = with_suffix.create_completion(
        "seed",
        suffix="SUFFIX",
        max_tokens=1,
        temperature=0.0,
    )

    assert out_plain["choices"][0]["text"] == "Q"
    assert out_suffix["choices"][0]["text"] == "Q"
    assert len(recorded_suffix["prompt_tokens"]) > len(recorded_plain["prompt_tokens"])
    assert not out_suffix["choices"][0]["text"].endswith("SUFFIX")


def test_completion_echo_includes_prompt_and_sets_first_logprob_none():
    llm = make_fake_llm(logits_all=True)
    T = {100: b"A", 101: b"B"}
    seq = [100, 101]
    install_fake_detokenize(llm, T)
    install_fake_generate(llm, seq)
    install_uniform_scores(llm, seq)

    out = llm.create_completion(
        "hi",
        max_tokens=len(seq),
        echo=True,
        logprobs=2,
        temperature=0.0,
    )

    choice = out["choices"][0]
    assert choice["text"].startswith("hi")
    logprobs = choice["logprobs"]
    assert logprobs is not None
    assert logprobs["token_logprobs"][0] is None
