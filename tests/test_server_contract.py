import json
import time
from typing import Dict, Iterable, Iterator, List, Optional, Union

import pytest
from fastapi.testclient import TestClient

from llama_cpp.server import app as app_module
from llama_cpp.server.app import create_app
from llama_cpp.server.settings import ModelSettings, ServerSettings


class FakeLlama:
    def __init__(self, alias: str) -> None:
        self.alias = alias

    def __call__(self, prompt: Union[str, List[str]], stream: bool = False, **kwargs):
        return self.create_completion(prompt=prompt, stream=stream, **kwargs)

    def create_completion(
        self,
        prompt: Union[str, List[str]],
        stream: bool = False,
        **kwargs,
    ):
        prompt_text = prompt if isinstance(prompt, str) else "".join(prompt)
        if prompt_text == "fail":
            raise RuntimeError("intentional failure")
        if stream:
            return self._stream_completion(prompt_text)
        created = int(time.time())
        return {
            "id": f"cmpl-{self.alias}",
            "object": "text_completion",
            "created": created,
            "model": self.alias,
            "choices": [
                {
                    "index": 0,
                    "text": f"Echo: {prompt_text}",
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_text),
                "completion_tokens": len(prompt_text) + 6,
                "total_tokens": (len(prompt_text) * 2) + 6,
            },
        }

    def _stream_completion(self, prompt_text: str) -> Iterator[Dict[str, object]]:
        created = int(time.time())
        chunks = [
            {
                "id": f"cmpl-{self.alias}",
                "object": "text_completion.chunk",
                "created": created,
                "model": self.alias,
                "choices": [
                    {
                        "index": 0,
                        "text": "Echo: ",
                        "finish_reason": None,
                        "logprobs": None,
                    }
                ],
            },
            {
                "id": f"cmpl-{self.alias}",
                "object": "text_completion.chunk",
                "created": created,
                "model": self.alias,
                "choices": [
                    {
                        "index": 0,
                        "text": prompt_text,
                        "finish_reason": None,
                        "logprobs": None,
                    }
                ],
            },
            {
                "id": f"cmpl-{self.alias}",
                "object": "text_completion.chunk",
                "created": created,
                "model": self.alias,
                "choices": [
                    {
                        "index": 0,
                        "text": "",
                        "finish_reason": "stop",
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_text),
                    "completion_tokens": len(prompt_text) + 6,
                    "total_tokens": (len(prompt_text) * 2) + 6,
                },
            },
        ]
        for chunk in chunks:
            yield chunk

    def create_chat_completion(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs):
        content = messages[-1]["content"]
        if stream:
            return self._stream_chat(content)
        created = int(time.time())
        return {
            "id": f"chatcmpl-{self.alias}",
            "object": "chat.completion",
            "created": created,
            "model": self.alias,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"Echo: {content}"},
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": len(content),
                "completion_tokens": len(content) + 6,
                "total_tokens": (len(content) * 2) + 6,
            },
        }

    def _stream_chat(self, content: str) -> Iterator[Dict[str, object]]:
        created = int(time.time())
        chunks = [
            {
                "id": f"chatcmpl-{self.alias}",
                "object": "chat.completion.chunk",
                "created": created,
                "model": self.alias,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "Echo: "},
                        "finish_reason": None,
                        "logprobs": None,
                    }
                ]
            },
            {
                "id": f"chatcmpl-{self.alias}",
                "object": "chat.completion.chunk",
                "created": created,
                "model": self.alias,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content},
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
                ],
                "id": f"chatcmpl-{self.alias}",
                "object": "chat.completion.chunk",
                "created": created,
                "model": self.alias,
                "usage": {
                    "prompt_tokens": len(content),
                    "completion_tokens": len(content) + 6,
                    "total_tokens": (len(content) * 2) + 6,
                },
            },
        ]
        for chunk in chunks:
            yield chunk

    def create_embedding(self, input: Union[str, List[str]], **kwargs):
        inputs = [input] if isinstance(input, str) else input
        data = [
            {
                "object": "embedding",
                "index": idx,
                "embedding": [float(len(item))],
            }
            for idx, item in enumerate(inputs)
        ]
        return {
            "object": "list",
            "model": self.alias,
            "data": data,
        }

    def tokenize(self, data: bytes, *args, **kwargs) -> List[int]:
        text = data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else data
        return [ord(ch) for ch in text]

    def detokenize(self, tokens: Iterable[int]) -> bytes:
        return bytes(int(token) for token in tokens)

    def token_eos(self) -> int:
        return 0


class FakeProxy:
    def __init__(self) -> None:
        self._models: Dict[str, ModelSettings] = {}
        self._instances: Dict[str, FakeLlama] = {}
        self._default_alias: Optional[str] = None

    def configure(self, models: List[ModelSettings]) -> None:
        self._models.clear()
        self._instances.clear()
        for index, settings in enumerate(models):
            alias = settings.model_alias or settings.model
            self._models[alias] = settings
            if index == 0:
                self._default_alias = alias

    def __call__(self, model: Optional[str] = None) -> FakeLlama:
        if not self._models:
            raise RuntimeError("fake proxy not configured")
        alias = model if model and model in self._models else self._default_alias
        assert alias is not None
        if alias not in self._instances:
            self._instances[alias] = FakeLlama(alias)
        return self._instances[alias]

    def __iter__(self) -> Iterator[str]:
        return iter(self._models.keys())


@pytest.fixture(name="client")
def client_fixture(monkeypatch):
    fake_proxy = FakeProxy()

    def fake_set_llama_proxy(model_settings: List[ModelSettings]) -> None:
        fake_proxy.configure(model_settings)
        app_module._llama_proxy = fake_proxy

    monkeypatch.setattr(app_module, "set_llama_proxy", fake_set_llama_proxy)

    server_settings = ServerSettings(
        api_key="test-key",
        interrupt_requests=False,
        disable_ping_events=True,
    )
    model_settings = [
        ModelSettings(
            model="fake-model.gguf",
            model_alias="fake-model",
            vocab_only=True,
        )
    ]

    app = create_app(server_settings=server_settings, model_settings=model_settings)
    test_client = TestClient(app)
    yield test_client
    test_client.close()


def _auth_headers() -> Dict[str, str]:
    return {"Authorization": "Bearer test-key"}


def test_completion_requires_auth(client: TestClient):
    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "hi"},
    )
    assert response.status_code == 401


def test_completion_returns_payload(client: TestClient):
    response = client.post(
        "/v1/completions",
        headers=_auth_headers(),
        json={"model": "fake-model", "prompt": "hi"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["text"] == "Echo: hi"


def test_chat_completion_returns_message(client: TestClient):
    response = client.post(
        "/v1/chat/completions",
        headers=_auth_headers(),
        json={
            "model": "fake-model",
            "messages": [
                {"role": "user", "content": "Hello"},
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "Echo: Hello"


def test_embedding_endpoint(client: TestClient):
    response = client.post(
        "/v1/embeddings",
        headers=_auth_headers(),
        json={"model": "fake-model", "input": "abc"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["data"][0]["embedding"] == [3.0]


def test_models_listing(client: TestClient):
    response = client.get("/v1/models", headers=_auth_headers())
    assert response.status_code == 200
    body = response.json()
    ids = [entry["id"] for entry in body["data"]]
    assert ids == ["fake-model"]


def test_tokenize_roundtrip(client: TestClient):
    payload = {"model": "fake-model", "input": "ab"}
    tok_response = client.post(
        "/extras/tokenize",
        headers=_auth_headers(),
        json=payload,
    )
    assert tok_response.status_code == 200
    tokens = tok_response.json()["tokens"]

    detok_response = client.post(
        "/extras/detokenize",
        headers=_auth_headers(),
        json={"model": "fake-model", "tokens": tokens},
    )
    assert detok_response.status_code == 200
    assert detok_response.json()["text"] == "ab"

    count_response = client.post(
        "/extras/tokenize/count",
        headers=_auth_headers(),
        json=payload,
    )
    assert count_response.status_code == 200
    assert count_response.json()["count"] == len(tokens)


def test_streaming_completion_emits_chunks(client: TestClient):
    with client.stream(
        "POST",
        "/v1/completions",
        headers=_auth_headers(),
        json={"model": "fake-model", "prompt": "hi", "stream": True},
    ) as stream:
        payloads = []
        for line in stream.iter_lines():
            if not line:
                continue
            assert line.startswith("data: ")
            data = line.removeprefix("data: ")
            if data == "[DONE]":
                break
            payloads.append(json.loads(data))

    texts = [chunk["choices"][0]["text"] for chunk in payloads]
    assert texts[:2] == ["Echo: ", "hi"]
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"


def test_error_response_is_wrapped(client: TestClient):
    response = client.post(
        "/v1/completions",
        headers=_auth_headers(),
        json={"model": "fake-model", "prompt": "fail"},
    )
    assert response.status_code == 500
    body = response.json()
    assert body["error"]["message"] == "intentional failure"
    assert body["error"]["type"] == "internal_server_error"
