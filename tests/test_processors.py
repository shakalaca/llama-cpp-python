import numpy as np
from llama_cpp.llama import Llama, LogitsProcessorList, MinTokensLogitsProcessor

MODEL = "./vendor/llama.cpp/models/ggml-vocab-llama-spm.gguf"


def test_processors_order_and_min_tokens():
    llm = Llama(MODEL, vocab_only=True, n_ctx=32, logits_all=True, verbose=False)

    # fabricate logits
    vocab = llm.n_vocab()
    logits = np.zeros((1, vocab), dtype=np.float32)
    # Make eos identifiable
    eos = llm.token_eos()

    # A processor that sets eos logit to a high value
    def favor_eos(input_ids, scores):
        scores[eos] = 1000.0
        return scores

    # MinTokens should zero out eos until min_tokens satisfied
    mt = MinTokensLogitsProcessor(min_tokens=3, token_eos=eos)

    procs = LogitsProcessorList([favor_eos, mt])
    # First call defines prompt length; simulate generation steps
    input_ids = np.array([1, 2, 3], dtype=np.intc)
    mt.prompt_tokens = len(input_ids)

    # before reaching 3 generated tokens, eos must be -inf
    s = np.copy(logits[0])
    s = procs(input_ids, s)
    assert np.isneginf(s[eos])

    # After 3 tokens, eos can be positive
    mt.prompt_tokens = len(input_ids) - 3
    s2 = np.copy(logits[0])
    s2 = procs(input_ids, s2)
    assert s2[eos] == 1000.0
