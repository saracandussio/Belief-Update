from belief_update.shared.metrics import cover_em, em, f1, normalize_answer
from belief_update.utils import (
    extract_answer,
    native_decomposition,
    parse_tool_call,
    resolve_placeholders,
)


def test_normalization_and_em():
    assert normalize_answer("The  Beatles!") == "beatles"
    assert em("the beatles", ["Beatles"])
    assert not em(None, ["x"])
    assert cover_em("Paris, France", ["Paris"])
    assert f1("New York City", ["New York"]) > 0.7
    assert f1("yes", ["no"]) == 0.0


def test_extract_answer_after_thinking():
    text = "<think>maybe <answer> Rome </answer>?</think>\nSo <answer> Paris </answer>"
    assert extract_answer(text, "Qwen/Qwen3-8B") == "Paris"
    assert extract_answer("<think>no tags</think> Paris", "Qwen/Qwen3-8B") is None
    # no end-of-thinking marker: fall back to whole text
    assert extract_answer("<answer>Oslo</answer>", "Qwen/Qwen3-8B") == "Oslo"


def test_parse_tool_call_formats():
    native = '<think>x</think>\n<tool_call>\n{"name": "search", "arguments": {"query": "capital of Oklahoma"}}\n</tool_call>'
    assert parse_tool_call(native, "qwen_native") == "capital of Oklahoma"
    double = '<tool_call>{"name": "search", "arguments": "{\\"query\\": \\"q\\"}"}</tool_call>'
    assert parse_tool_call(double, "qwen_native") == "q"
    assert parse_tool_call("<tool_call>{bad json}</tool_call>", "qwen_native") is None
    assert parse_tool_call('<tool_call>{"name": "other", "arguments": {}}</tool_call>', "qwen_native") is None
    assert parse_tool_call("<think>a</think><search> Dallas state </search>", "searchr1") == "Dallas state"


def test_musique_placeholders():
    assert resolve_placeholders("Who founded #1 in #2?", ["Acme", "1999"]) == "Who founded Acme in 1999?"
    item = {"metadata": {"question_decomposition": [
        {"question": "Green >> performer", "answer": "Steve Hillage"},
        {"question": "#1 >> spouse", "answer": "Miquette Giraudy"},
    ]}}
    hops = native_decomposition(item)
    assert hops[1]["question"] == "Steve Hillage >> spouse"
    assert native_decomposition({"metadata": {}}) is None
