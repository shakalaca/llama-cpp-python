import ctypes
import pytest

import llama_cpp
from llama_cpp.llama import Llama

MODEL = "./vendor/llama.cpp/models/ggml-vocab-llama-spm.gguf"


def test_kv_overrides_types_and_limits():
    # string <= 128 ok, >128 raises
    ok_str = "x" * 10
    bad_str = "y" * 129

    # Construct with kv_overrides
    llm = Llama(
        MODEL,
        vocab_only=True,
        kv_overrides={
            "bool_key": True,
            "int_key": 1,
            "float_key": 1.5,
            "str_key": ok_str,
        },
        verbose=False,
    )

    arr = llm._kv_overrides_array
    # Ensure tags
    tags = {b"bool_key": llama_cpp.LLAMA_KV_OVERRIDE_TYPE_BOOL,
            b"int_key": llama_cpp.LLAMA_KV_OVERRIDE_TYPE_INT,
            b"float_key": llama_cpp.LLAMA_KV_OVERRIDE_TYPE_FLOAT,
            b"str_key": llama_cpp.LLAMA_KV_OVERRIDE_TYPE_STR}
    seen = set()
    for item in arr:
        if item.key == b"\x00":
            break
        seen.add(item.key)
        assert item.tag == tags[item.key]
    assert set(tags.keys()).issubset(seen)

    # oversize should raise
    with pytest.raises(ValueError):
        Llama(MODEL, vocab_only=True, kv_overrides={"k": bad_str}, verbose=False)
