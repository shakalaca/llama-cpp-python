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
    monkeypatch.setattr(llm._batch, "add_sequence", add_sequence_stub, raising=False)

    # Monkeypatch context.decode to no-op and set fake embeddings
    monkeypatch.setattr(llm._ctx, "decode", lambda batch: None, raising=False)
    monkeypatch.setattr(llm._ctx, "kv_cache_clear", lambda: None, raising=False)

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
    monkeypatch.setattr(C, "llama_get_embeddings_seq", fake_get_embeddings_seq, raising=False)

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
    monkeypatch.setattr(C, "llama_get_embeddings_seq", fake_get_embeddings_seq, raising=False)

    v = llm.embed("x", normalize=True)
    assert isinstance(v, list) and abs(sum(x * x for x in v) - 1.0) < 1e-6


def test_embed_multi_inputs_precision(monkeypatch):
    llm = Llama(MODEL, vocab_only=True, embedding=True, n_batch=8, verbose=False)
    monkeypatch.setattr(llm, "n_embd", lambda: 4)

    added = []
    real_add = llm._batch.add_sequence

    def add_sequence_stub(tokens, seq_id, logits_all):
        added.append(len(tokens))
        return real_add(tokens, seq_id, logits_all)

    monkeypatch.setattr(llm._batch, "add_sequence", add_sequence_stub, raising=False)
    monkeypatch.setattr(llm._ctx, "decode", lambda batch: None, raising=False)
    monkeypatch.setattr(llm._ctx, "kv_cache_clear", lambda: None, raising=False)

    import llama_cpp.llama_cpp as C

    monkeypatch.setattr(llm, "pooling_type", lambda: C.LLAMA_POOLING_TYPE_MEAN)

    def fake_get_embeddings_seq(ctx, i):
        base = float(i + 1)

        class View:
            def __getitem__(self, sl):
                length = sl.stop - sl.start
                return [base + 0.01 * j for j in range(length)]

        return View()

    monkeypatch.setattr(C, "llama_get_embeddings_seq", fake_get_embeddings_seq, raising=False)

    inputs = ["hello", "world"]
    embeddings = llm.embed(inputs, truncate=False, return_count=False)

    assert isinstance(embeddings, list) and len(embeddings) == len(inputs)

    first_three = [round(v, 2) for v in embeddings[0][:3]]
    second_three = [round(v, 2) for v in embeddings[1][:3]]
    assert first_three == [1.0, 1.01, 1.02]
    assert second_three == [2.0, 2.01, 2.02]

    expected_tokens = [len(llm.tokenize(text.encode("utf-8"))) for text in inputs]
    assert added == expected_tokens


def test_embed_return_count_reports_total_tokens(monkeypatch):
    llm = Llama(MODEL, vocab_only=True, embedding=True, n_batch=8, verbose=False)
    monkeypatch.setattr(llm, "n_embd", lambda: 3)

    added = []
    real_add = llm._batch.add_sequence

    def add_sequence_stub(tokens, seq_id, logits_all):
        added.append(len(tokens))
        return real_add(tokens, seq_id, logits_all)

    monkeypatch.setattr(llm._batch, "add_sequence", add_sequence_stub, raising=False)
    monkeypatch.setattr(llm._ctx, "decode", lambda batch: None, raising=False)
    monkeypatch.setattr(llm._ctx, "kv_cache_clear", lambda: None, raising=False)

    import llama_cpp.llama_cpp as C

    monkeypatch.setattr(llm, "pooling_type", lambda: C.LLAMA_POOLING_TYPE_MEAN)

    def fake_get_embeddings_seq(ctx, i):
        base = float(i + 1)

        class View:
            def __getitem__(self, sl):
                length = sl.stop - sl.start
                return [base + 0.1 * j for j in range(length)]

        return View()

    monkeypatch.setattr(C, "llama_get_embeddings_seq", fake_get_embeddings_seq, raising=False)

    inputs = ["first", "second"]
    expected_counts = [len(llm.tokenize(text.encode("utf-8"))) for text in inputs]

    embeddings, total_tokens = llm.embed(inputs, return_count=True)

    assert isinstance(embeddings, list) and len(embeddings) == len(inputs)
    assert total_tokens == sum(expected_counts)
    assert added == expected_counts


def test_embed_pooling_none_returns_token_embeddings(monkeypatch):
    llm = Llama(MODEL, vocab_only=True, embedding=True, n_batch=8, verbose=False)
    llm.vocab_only = False  # type: ignore[attr-defined]
    monkeypatch.setattr(llm, "n_embd", lambda: 3)

    recorded_logits_all = []
    real_add = llm._batch.add_sequence

    def add_sequence_stub(tokens, seq_id, logits_all):
        recorded_logits_all.append(logits_all)
        return real_add(tokens, seq_id, logits_all)

    monkeypatch.setattr(llm._batch, "add_sequence", add_sequence_stub, raising=False)
    monkeypatch.setattr(llm._ctx, "decode", lambda batch: None, raising=False)
    monkeypatch.setattr(llm._ctx, "kv_cache_clear", lambda: None, raising=False)

    import llama_cpp.llama_cpp as C

    monkeypatch.setattr(llm, "pooling_type", lambda: C.LLAMA_POOLING_TYPE_NONE)

    inputs = ["alpha", "beta"]
    token_counts = [len(llm.tokenize(text.encode("utf-8"))) for text in inputs]

    flat = []
    for idx, count in enumerate(token_counts):
        base = float(idx + 1)
        for _ in range(count):
            flat.extend([base, base + 0.1, base + 0.2])

    def fake_get_embeddings(ctx):
        return flat

    monkeypatch.setattr(C, "llama_get_embeddings", fake_get_embeddings, raising=False)

    embeddings = llm.embed(inputs, truncate=True, return_count=False)

    assert isinstance(embeddings, list) and len(embeddings) == len(inputs)
    for emb, count in zip(embeddings, token_counts):
        assert len(emb) == count
        for vec in emb:
            assert len(vec) == 3
    assert all(recorded_logits_all)


def test_embed_raises_without_truncate_when_over_batch(monkeypatch):
    llm = Llama(MODEL, vocab_only=True, embedding=True, n_batch=4, verbose=False)

    import llama_cpp.llama_cpp as C

    monkeypatch.setattr(C, "llama_get_embeddings_seq", lambda ctx, i: lambda sl: [0.0] * (sl.stop - sl.start), raising=False)
    monkeypatch.setattr(llm._ctx, "decode", lambda batch: None, raising=False)
    monkeypatch.setattr(llm._ctx, "kv_cache_clear", lambda: None, raising=False)

    def fake_tokenize(data):
        return list(range(llm.n_batch + 1))

    monkeypatch.setattr(llm, "tokenize", fake_tokenize)

    with pytest.raises(ValueError):
        llm.embed("x", truncate=False)
