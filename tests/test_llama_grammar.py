import json
import types
from typing import Dict, List, Set

import pytest

import llama_cpp
import llama_cpp.llama_grammar as llama_grammar

MODEL = "./vendor/llama.cpp/models/ggml-vocab-llama-spm.gguf"


def _parse_simple_gbnf(grammar_str: str) -> Dict[str, List[List[str]]]:
    rules: Dict[str, List[List[str]]] = {}
    for raw in grammar_str.strip().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "::=" not in line:
            continue
        name, expr = line.split("::=", 1)
        name = name.strip()
        productions: List[List[str]] = []
        for alt in expr.split("|"):
            tokens = [token for token in alt.strip().split(" ") if token]
            cleaned = [token.strip('"') for token in tokens]
            productions.append(cleaned)
        rules[name] = productions
    return rules


def _match_rule(rules: Dict[str, List[List[str]]], symbol: str, text: str, index: int) -> Set[int]:
    if symbol not in rules:
        literal = symbol
        if text.startswith(literal, index):
            return {index + len(literal)}
        return set()

    outcomes: Set[int] = set()
    for production in rules[symbol]:
        positions: Set[int] = {index}
        for part in production:
            next_positions: Set[int] = set()
            for pos in positions:
                next_positions.update(_match_rule(rules, part, text, pos))
            positions = next_positions
            if not positions:
                break
        outcomes.update(positions)
    return outcomes


def _accepts(rules: Dict[str, List[List[str]]], root: str, candidate: str) -> bool:
    return len(candidate) in _match_rule(rules, root, candidate, 0)

tree = """
leaf ::= "."
node ::= leaf | "(" node node ")"
root ::= node
"""


def test_grammar_from_string():
    grammar = llama_cpp.LlamaGrammar.from_string(tree)
    assert grammar._root == llama_grammar.LLAMA_GRAMMAR_DEFAULT_ROOT
    assert grammar._grammar.strip() == tree.strip()

    parsed = _parse_simple_gbnf(grammar._grammar)
    assert _accepts(parsed, grammar._root, ".")
    assert _accepts(parsed, grammar._root, "(..)")
    assert not _accepts(parsed, grammar._root, "(.)")


def test_composed_pydantic_grammar():
    """
    from pydantic import BaseModel

    class A(BaseModel):
        a: int

    class B(BaseModel):
        a: A
        b: int
    """

    # This schema corresponds to the grammar in the comment above.
    # We don't use the pydantic models directly to avoid the dependency.
    schema = {
        "$defs": {
            "A": {
                "properties": {"a": {"title": "A", "type": "integer"}},
                "required": ["a"],
                "title": "A",
                "type": "object",
            }
        },
        "properties": {
            "a": {"$ref": "#/$defs/A"},
            "b": {"title": "B", "type": "integer"},
        },
        "required": ["a", "b"],
        "title": "B",
        "type": "object",
    }

    grammar = llama_cpp.LlamaGrammar.from_json_schema(json.dumps(schema))
    lines = grammar._grammar.splitlines()
    assert any(line.startswith("A ::=") for line in lines)
    assert any("A-a-kv" in line and "integer" in line for line in lines)
    assert any("b-kv" in line and "integer" in line for line in lines)


def test_grammar_anyof():
    sch = {
        "properties": {
            "temperature": {
                "description": "The temperature mentioned",
                "type": "number",
            },
            "unit": {
                "anyOf": [
                    {
                        "description": "Unit for temperature",
                        "enum": ["celsius", "fahrenheit"],
                        "type": "string",
                    },
                    {"type": "null"},
                ],
            },
        },
        "type": "object",
    }

    grammar = llama_cpp.LlamaGrammar.from_json_schema(json.dumps(sch))
    lines = grammar._grammar.splitlines()
    assert any("unit ::= unit-0 | null" in line for line in lines)
    assert any('unit-0 ::= "\\"celsius\\"" | "\\"fahrenheit\\""' in line for line in lines)


def test_grammar_invalid_schema_raises():
    bad_schema = {"type": "string", "pattern": "abc"}
    with pytest.raises(AssertionError):
        llama_cpp.LlamaGrammar.from_json_schema(json.dumps(bad_schema))


def _make_stub_llm(monkeypatch):
    llm = llama_cpp.Llama(
        MODEL,
        vocab_only=True,
        n_ctx=64,
        n_batch=32,
        n_ubatch=32,
        logits_all=True,
        verbose=False,
    )

    monkeypatch.setattr(
        llm,
        "tokenize",
        lambda data, add_bos=False, special=True: [1, 2, 3],
        raising=False,
    )

    recorded = {}

    def fake_create_completion(self, prompt, **kwargs):
        recorded["prompt"] = list(prompt)
        recorded["kwargs"] = kwargs
        return {
            "id": "cmpl-test",
            "object": "text_completion",
            "created": 0,
            "model": "stub",
            "choices": [
                {
                    "text": json.dumps({"ok": True}),
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt),
                "completion_tokens": 0,
                "total_tokens": len(prompt),
            },
        }

    monkeypatch.setattr(
        llm,
        "create_completion",
        types.MethodType(fake_create_completion, llm),
        raising=False,
    )

    return llm, recorded


def test_chat_completion_passes_explicit_grammar(monkeypatch):
    llm, recorded = _make_stub_llm(monkeypatch)
    explicit = llama_cpp.LlamaGrammar.from_string('root ::= "hi"')

    result = llm.create_chat_completion(
        messages=[{"role": "user", "content": "hello"}],
        grammar=explicit,
    )

    assert recorded["kwargs"]["grammar"] is explicit
    assert result["choices"][0]["message"]["content"] == json.dumps({"ok": True})


def test_chat_completion_response_format_builds_grammar(monkeypatch):
    llm, recorded = _make_stub_llm(monkeypatch)
    schema = {
        "type": "object",
        "properties": {"foo": {"type": "integer"}},
        "required": ["foo"],
    }

    llm.create_chat_completion(
        messages=[{"role": "user", "content": "hello"}],
        response_format={"type": "json_object", "schema": schema},
    )

    grammar = recorded["kwargs"].get("grammar")
    assert isinstance(grammar, llama_cpp.LlamaGrammar)
    assert "foo" in grammar._grammar


def test_chat_completion_tool_choice_uses_tool_schema(monkeypatch):
    llm, recorded = _make_stub_llm(monkeypatch)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract",
                "parameters": {
                    "type": "object",
                    "properties": {"bar": {"type": "string"}},
                },
            },
        }
    ]

    llm.create_chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "extract"}},
    )

    grammar = recorded["kwargs"].get("grammar")
    assert isinstance(grammar, llama_cpp.LlamaGrammar)
    assert "bar" in grammar._grammar


def test_recursive_schema_conversion_handles_depth():
    depth = 16
    defs: Dict[str, Dict[str, object]] = {}
    for level in range(depth):
        obj: Dict[str, object] = {
            "type": "object",
            "properties": {
                "value": {"type": "integer"},
            },
            "required": ["value"],
            "additionalProperties": False,
        }
        if level < depth - 1:
            obj["properties"]["next"] = {"$ref": f"#/$defs/Node{level + 1}"}
            obj["required"].append("next")
        else:
            obj["properties"]["next"] = {"type": "null"}
        defs[f"Node{level}"] = obj

    schema = {
        "$defs": defs,
        "type": "object",
        "properties": {"head": {"$ref": "#/$defs/Node0"}},
        "required": ["head"],
        "additionalProperties": False,
    }

    grammar = llama_cpp.LlamaGrammar.from_json_schema(json.dumps(schema))

    for level in range(depth):
        assert f"Node{level}" in grammar._grammar
    assert grammar._grammar.count("value") >= depth
