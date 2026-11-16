from belief_update.agent import AgentState, run_search_loop
from belief_update.utils import TOOL_FORMATS


def enc(text):
    return [ord(c) for c in text]


def scripted_generate(script):
    """script: {uid: [turn texts]} consumed in order per state."""
    calls = {}

    def gen(prompts):
        out = []
        for p in prompts:
            uid = chr(p[0])  # first token encodes the uid
            k = calls.get(uid, 0)
            calls[uid] = k + 1
            text = script[uid][k]
            out.append({"text": text, "token_ids": enc(text), "finish_reason": "stop"})
        return out

    return gen


def fake_retrieve(queries):
    return [[{"document": {"contents": f"Title {q}\nBody about {q}."}, "score": 1.0}] for q in queries]


CALL = '<think>need it</think><tool_call>{{"name": "search", "arguments": {{"query": "{q}"}}}}</tool_call>'


def make_state(uid):
    return AgentState(uid=uid, sample_idx=0, token_ids=enc(uid + "PROMPT"), prompt_len=7)


def test_loop_answers_after_search_and_keeps_tokens():
    script = {
        "a": [CALL.format(q="Dallas state"), "<think>ok</think><answer> Oklahoma City </answer>"],
        "b": ["<think>known</think><answer> Austin </answer>"],
        "c": ['<tool_call>{"oops"}</tool_call>'],
    }
    states = [make_state(u) for u in "abc"]
    run_search_loop(states, generate_fn=scripted_generate(script), retrieve_fn=fake_retrieve,
                    encode_fn=enc, tool_format="qwen_native", model_name="Qwen/Qwen3-8B")
    a, b, c = states
    assert (a.finish_reason, a.answer, a.n_calls) == ("answer", "Oklahoma City", 1)
    assert [s["type"] for s in a.segments] == ["model", "tool", "model"]
    assert a.segments[1]["query"] == "Dallas state"
    assert a.segments[1]["docs"][0]["title"] == "Title Dallas state"
    # token stream = prompt + turn1 + tool response + turn2, in order
    resp = TOOL_FORMATS["qwen_native"]["response_template"].format(
        result="Doc 1(Title: Title Dallas state) Body about Dallas state.")
    expected = "aPROMPT" + script["a"][0] + resp + script["a"][1]
    assert "".join(map(chr, a.token_ids)) == expected
    assert (b.finish_reason, b.n_calls) == ("answer", 0)
    assert c.finish_reason == "invalid_call"


def test_max_calls_and_searchr1_format():
    script = {"a": ["<search> q </search>"] * 5}
    s = make_state("a")
    run_search_loop([s], generate_fn=scripted_generate(script), retrieve_fn=fake_retrieve,
                    encode_fn=enc, tool_format="searchr1", model_name="x", max_calls=2)
    assert s.n_calls == 2 and s.finish_reason == "max_calls"
    assert "<information>Doc 1(Title: Title q)" in "".join(map(chr, s.token_ids))
