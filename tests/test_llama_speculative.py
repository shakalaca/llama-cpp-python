import numpy as np

from llama_cpp.llama_speculative import LlamaPromptLookupDecoding


def test_find_candidate_pred_tokens_returns_match():
    find_candidate_pred_tokens = LlamaPromptLookupDecoding.find_candidate_pred_tokens

    input_ids = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
    result = find_candidate_pred_tokens(input_ids, max_ngram_size=3, num_pred_tokens=2)

    assert np.array_equal(result, np.array([1, 2]))


def test_find_candidate_pred_tokens_no_match_returns_empty():
    find_candidate_pred_tokens = LlamaPromptLookupDecoding.find_candidate_pred_tokens

    input_ids = np.array([1, 2, 3, 4, 5])
    result = find_candidate_pred_tokens(input_ids, max_ngram_size=3, num_pred_tokens=2)

    assert np.array_equal(result, np.array([]))


def test_find_candidate_pred_tokens_truncates_to_available_length():
    find_candidate_pred_tokens = LlamaPromptLookupDecoding.find_candidate_pred_tokens

    input_ids = np.array([4, 4, 4, 4])
    result = find_candidate_pred_tokens(input_ids, max_ngram_size=2, num_pred_tokens=3)

    assert np.array_equal(result, np.array([4, 4]))


def test_find_candidate_pred_tokens_short_context():
    find_candidate_pred_tokens = LlamaPromptLookupDecoding.find_candidate_pred_tokens

    input_ids = np.array([7])
    result = find_candidate_pred_tokens(input_ids, max_ngram_size=2, num_pred_tokens=2)

    assert np.array_equal(result, np.array([]))


def test_prompt_lookup_decoding_uses_instance_configuration():
    decoder = LlamaPromptLookupDecoding(max_ngram_size=1, num_pred_tokens=2)
    input_ids = np.array([5, 6, 7, 6, 7])

    result = decoder(input_ids)

    assert np.array_equal(result, np.array([6, 7]))
