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


def test_model_params_has_no_host_bool():
    mparams = llama_cpp.llama_model_default_params()
    assert hasattr(mparams, "no_host")
    assert isinstance(mparams.no_host, bool)


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
