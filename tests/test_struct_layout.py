import multiprocessing

import llama_cpp

MODEL = "./vendor/llama.cpp/models/ggml-vocab-llama-spm.gguf"


def test_context_params_flash_attn_type_default_and_shim():
    cparams = llama_cpp.llama_context_default_params()
    # default should be AUTO
    assert cparams.flash_attn_type == llama_cpp.LLAMA_FLASH_ATTN_TYPE_AUTO

    # shim maps bool to enum
    cparams.flash_attn = True
    assert cparams.flash_attn_type == llama_cpp.LLAMA_FLASH_ATTN_TYPE_ENABLED

    cparams.flash_attn = False
    assert cparams.flash_attn_type == llama_cpp.LLAMA_FLASH_ATTN_TYPE_DISABLED

    # boolean tail fields should be present and bools
    for name in [
        "embeddings",
        "offload_kqv",
        "no_perf",
        "op_offload",
        "swa_full",
        "kv_unified",
    ]:
        assert hasattr(cparams, name), f"missing field: {name}"
        assert isinstance(getattr(cparams, name), bool), f"{name} not bool"


def test_context_params_rope_and_kv_defaults():
    cparams = llama_cpp.llama_context_default_params()
    assert cparams.rope_scaling_type == llama_cpp.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED
    assert cparams.type_k == llama_cpp.GGML_TYPE_F16
    assert cparams.type_v == llama_cpp.GGML_TYPE_F16
    assert hasattr(cparams, "kv_unified")
    assert isinstance(cparams.kv_unified, bool)


def test_context_params_attention_and_sequence_fields():
    cparams = llama_cpp.llama_context_default_params()
    assert hasattr(cparams, "attention_type")
    assert isinstance(cparams.attention_type, int)
    assert hasattr(cparams, "n_seq_max")
    assert cparams.n_seq_max >= 1


def test_context_params_attention_enum_matches_bindings():
    cparams = llama_cpp.llama_context_default_params()
    attention_constants = [
        getattr(llama_cpp, name)
        for name in dir(llama_cpp)
        if name.startswith("LLAMA_ATTENTION_TYPE_")
    ]
    assert attention_constants, "expected LLAMA_ATTENTION_TYPE_* constants"
    assert cparams.attention_type in attention_constants


def test_model_quantize_params_boolean_fields():
    qparams = llama_cpp.llama_model_quantize_default_params()
    for name in [
        "allow_requantize",
        "quantize_output_tensor",
        "only_copy",
        "pure",
        "keep_split",
    ]:
        assert hasattr(qparams, name), f"missing field: {name}"
        assert isinstance(getattr(qparams, name), bool), f"{name} not bool"


def test_model_params_has_no_host_bool():
    mparams = llama_cpp.llama_model_default_params()
    assert hasattr(mparams, "no_host")
    assert isinstance(mparams.no_host, bool)


def test_model_params_progress_callback_defaults_none():
    mparams = llama_cpp.llama_model_default_params()
    assert hasattr(mparams, "progress_callback")
    assert callable(mparams.progress_callback)


def test_opt_params_has_optimizer_type_field():
    lopt = llama_cpp.llama_opt_params()
    # should be settable and int-like
    lopt.optimizer_type = 0
    assert isinstance(lopt.optimizer_type, int)


def test_high_level_llama_flash_attn_shim_works():
    # Use tiny vocab-only model for lightweight construction
    llm = llama_cpp.Llama(
        MODEL,
        vocab_only=True,
        n_ctx=16,
        n_batch=16,
        n_ubatch=16,
        n_threads=multiprocessing.cpu_count(),
        n_threads_batch=multiprocessing.cpu_count(),
        flash_attn=True,
        verbose=False,
    )
    assert llm.context_params.flash_attn_type != llama_cpp.LLAMA_FLASH_ATTN_TYPE_DISABLED
