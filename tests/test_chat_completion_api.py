import sys
import types
from typing import Dict, Iterator, List, Optional

import pytest

import llama_cpp
from llama_cpp.llama import Llama

MODEL = "./vendor/llama.cpp/models/ggml-vocab-llama-spm.gguf"


def make_fake_llm() -> Llama:
    llm = Llama(
        MODEL,
        vocab_only=True,
        n_ctx=64,
        n_batch=32,
        n_ubatch=32,
        logits_all=True,
        verbose=False,
    )
    llm.set_seed(123)
    return llm


def test_create_chat_completion_forwards_arguments():
    llm = make_fake_llm()
    captured: Dict[str, Dict[str, object]] = {}
    sentinel_processor = object()
    sentinel_grammar = object()
    sentinel_bias = {7: 1.5}

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ]
    functions = [
        {
            "name": "store_message",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
            },
        }
    ]
    function_call = {"name": "store_message"}
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]
    tool_choice = {"type": "function", "function": {"name": "search"}}
    response_format = {"type": "json_object"}

    expected_payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "alias",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Here is a joke."},
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 6,
            "total_tokens": 16,
        },
    }

    def fake_handler(**kwargs):
        captured["kwargs"] = kwargs
        return expected_payload

    llm.chat_handler = fake_handler  # type: ignore[attr-defined]

    result = llm.create_chat_completion(
        messages=messages,
        functions=functions,
        function_call=function_call,
        tools=tools,
        tool_choice=tool_choice,
        temperature=0.3,
        top_p=0.91,
        top_k=33,
        min_p=0.2,
        typical_p=0.85,
        stream=False,
        stop=["DONE"],
        seed=99,
        response_format=response_format,
        max_tokens=42,
        presence_penalty=0.1,
        frequency_penalty=0.2,
        repeat_penalty=1.25,
        tfs_z=0.7,
        mirostat_mode=1,
        mirostat_tau=4.5,
        mirostat_eta=0.3,
        model="alias",
        logits_processor=sentinel_processor,  # type: ignore[arg-type]
        grammar=sentinel_grammar,  # type: ignore[arg-type]
        logit_bias=sentinel_bias,
        logprobs=True,
        top_logprobs=3,
    )

    assert result == expected_payload
    forwarded = captured["kwargs"]
    assert forwarded["llama"] is llm
    assert forwarded["messages"] == messages
    assert forwarded["functions"] == functions
    assert forwarded["function_call"] == function_call
    assert forwarded["tools"] == tools
    assert forwarded["tool_choice"] == tool_choice
    assert forwarded["temperature"] == 0.3
    assert forwarded["top_p"] == 0.91
    assert forwarded["top_k"] == 33
    assert forwarded["min_p"] == 0.2
    assert forwarded["typical_p"] == 0.85
    assert forwarded["stream"] is False
    assert forwarded["stop"] == ["DONE"]
    assert forwarded["seed"] == 99
    assert forwarded["response_format"] == response_format
    assert forwarded["max_tokens"] == 42
    assert forwarded["presence_penalty"] == 0.1
    assert forwarded["frequency_penalty"] == 0.2
    assert forwarded["repeat_penalty"] == 1.25
    assert forwarded["tfs_z"] == 0.7
    assert forwarded["mirostat_mode"] == 1
    assert forwarded["mirostat_tau"] == 4.5
    assert forwarded["mirostat_eta"] == 0.3
    assert forwarded["model"] == "alias"
    assert forwarded["logits_processor"] is sentinel_processor
    assert forwarded["grammar"] is sentinel_grammar
    assert forwarded["logit_bias"] == sentinel_bias
    assert forwarded["logprobs"] is True
    assert forwarded["top_logprobs"] == 3


def test_create_chat_completion_streams_chunks():
    llm = make_fake_llm()
    captured: Dict[str, Dict[str, object]] = {}
    chunks = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hi"},
                    "finish_reason": None,
                    "logprobs": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": " there"},
                    "finish_reason": None,
                    "logprobs": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ]
        },
    ]

    def fake_handler(**kwargs) -> Iterator[Dict[str, object]]:
        captured["kwargs"] = kwargs
        return iter(chunks)

    llm.chat_handler = fake_handler  # type: ignore[attr-defined]

    iterator = llm.create_chat_completion(
        messages=[{"role": "user", "content": "hey"}],
        stream=True,
    )

    assert list(iterator) == chunks
    forwarded = captured["kwargs"]
    assert forwarded["stream"] is True


class _SimpleNamespace:
    def __init__(self, **data: object) -> None:
        self.__dict__.update(data)


@pytest.fixture(name="mock_openai")
def mock_openai_fixture(monkeypatch):
    module_chat = types.ModuleType("openai.types.chat")
    module_chat.ChatCompletion = type("ChatCompletion", (_SimpleNamespace,), {})
    module_chat.ChatCompletionChunk = type("ChatCompletionChunk", (_SimpleNamespace,), {})

    module_types = types.ModuleType("openai.types")
    module_types.chat = module_chat

    module_openai = types.ModuleType("openai")
    module_openai.types = module_types

    monkeypatch.setitem(sys.modules, "openai", module_openai)
    monkeypatch.setitem(sys.modules, "openai.types", module_types)
    monkeypatch.setitem(sys.modules, "openai.types.chat", module_chat)
    yield module_chat


def test_create_chat_completion_openai_v1_wraps_response(mock_openai):
    llm = make_fake_llm()

    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "alias",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "pong"},
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "total_tokens": 5,
        },
    }

    def fake_handler(**kwargs):
        return payload

    llm.chat_handler = fake_handler  # type: ignore[attr-defined]

    response = llm.create_chat_completion_openai_v1(
        messages=[{"role": "user", "content": "ping"}],
        stream=False,
    )

    assert isinstance(response, mock_openai.ChatCompletion)
    assert response.id == "chatcmpl-test"
    assert response.choices[0]["message"]["content"] == "pong"


def test_create_chat_completion_openai_v1_wraps_stream(mock_openai):
    llm = make_fake_llm()

    chunks = [
        {
            "id": "chunk-1",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "p"},
                    "finish_reason": None,
                    "logprobs": None,
                }
            ],
        },
        {
            "id": "chunk-2",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "ong"},
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
        },
    ]

    def fake_handler(**kwargs):
        return iter(chunks)

    llm.chat_handler = fake_handler  # type: ignore[attr-defined]

    stream = llm.create_chat_completion_openai_v1(
        messages=[{"role": "user", "content": "ping"}],
        stream=True,
    )

    materialized = list(stream)
    assert all(isinstance(chunk, mock_openai.ChatCompletionChunk) for chunk in materialized)
    assert [chunk.id for chunk in materialized] == ["chunk-1", "chunk-2"]
