import json

from belief_update.agent import AgentState, run_search_loop
from belief_update.modeling import derive_response_template, make_response_formatter
from belief_update.utils import TOOL_FORMATS, extract_answer, get_tool_format, parse_tool_call


def test_default_formats():
    assert get_tool_format("Qwen/Qwen3.8-27B") == "qwen_xml"
    assert get_tool_format("Qwen/Qwen3.5-27B-FP8") == "qwen_xml"
    assert get_tool_format("Qwen/Qwen3-8B") == "qwen_native"
    assert get_tool_format("openai/gpt-oss-20b") == "harmony"
    assert get_tool_format("PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo") == "searchr1"


def test_parse_qwen_xml():
    text = ("<think>need the state</think>\n\n<tool_call>\n<function=search>\n"
            "<parameter=query>\nDallas state\n</parameter>\n</function>\n</tool_call>")
    assert parse_tool_call(text, "qwen_xml") == "Dallas state"
    other = "<tool_call>\n<function=browse>\n<parameter=query>x</parameter>\n</function>\n</tool_call>"
    assert parse_tool_call(other, "qwen_xml") is None


def test_parse_harmony():
    a = ('<|channel|>analysis<|message|>Need to search.<|end|><|start|>assistant<|channel|>'
         'commentary to=functions.search <|constrain|>json<|message|>{"query":"Dallas state"}<|call|>')
    assert parse_tool_call(a, "harmony") == "Dallas state"
    b = ('<|start|>assistant to=functions.search<|channel|>commentary json<|message|>'
         '{"query": "capital of Texas"}<|call|>')
    assert parse_tool_call(b, "harmony") == "capital of Texas"
    assert parse_tool_call('to=functions.browse<|message|>{"query":"x"}<|call|>', "harmony") is None
    final = ("<|channel|>analysis<|message|>ok<|end|><|start|>assistant<|channel|>final<|message|>"
             "<answer> Austin </answer><|return|>")
    assert extract_answer(final, "openai/gpt-oss-20b") == "Austin"


class QwenXmlTemplate:
    """Mimics the Qwen3.5+ template for the parts that matter."""

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True,
                            tokenize=False, enable_thinking=True, **kw):
        out = "<|im_start|>system\n# Tools ...<|im_end|>\n" if tools else ""
        for m in messages:
            if m["role"] == "user":
                out += f"<|im_start|>user\n{m['content']}<|im_end|>\n"
            elif m["role"] == "assistant":
                out += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                for tc in m.get("tool_calls", []):
                    f = tc["function"]
                    params = "".join(f"<parameter={k}>\n{v}\n</parameter>\n"
                                     for k, v in f["arguments"].items())
                    out += f"<tool_call>\n<function={f['name']}>\n{params}</function>\n</tool_call>"
                out += "<|im_end|>\n"
            elif m["role"] == "tool":
                out += f"<|im_start|>user\n<tool_response>\n{m['content']}\n</tool_response><|im_end|>\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n" + ("<think>\n" if enable_thinking else "")
        return out


class HarmonyTemplate:
    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True,
                            tokenize=False, **kw):
        out = "<|start|>developer<|message|># Tools ...<|end|>" if tools else ""
        for m in messages:
            if m["role"] == "user":
                out += f"<|start|>user<|message|>{m['content']}<|end|>"
            elif m["role"] == "assistant":
                f = m["tool_calls"][0]["function"]
                out += (f"<|start|>assistant to=functions.{f['name']}<|channel|>commentary json"
                        f"<|message|>{json.dumps(f['arguments'])}<|call|>")
            elif m["role"] == "tool":
                out += (f"<|start|>functions.search to=assistant<|channel|>commentary<|message|>"
                        f"{json.dumps(m['content'])}<|end|>")
        return out + ("<|start|>assistant" if add_generation_prompt else "")


def test_derive_response_template():
    tpl, js = derive_response_template(QwenXmlTemplate(), "qwen_xml", "Qwen/Qwen3.8-27B")
    # the derived wrapper keeps the generation prompt's "<think>\n", unlike the fallback
    assert tpl == TOOL_FORMATS["qwen_xml"]["response_template"] + "<think>\n" and not js
    tpl, js = derive_response_template(HarmonyTemplate(), "harmony", "openai/gpt-oss-20b")
    assert js and tpl == TOOL_FORMATS["harmony"]["response_template"]
    assert make_response_formatter(tpl, js)('Doc 1 "x"\n').startswith(
        '<|start|>functions.search to=assistant<|channel|>commentary<|message|>"Doc 1 \\"x\\"\\n"')
    assert derive_response_template(object(), "harmony", "openai/gpt-oss-20b") == (None, False)


def test_agent_harmony_loop():
    turns = iter([
        ('<|channel|>analysis<|message|>search<|end|><|start|>assistant<|channel|>commentary '
         'to=functions.search <|constrain|>json<|message|>{"query":"q"}<|call|>'),
        '<|channel|>analysis<|message|>ok<|end|><|start|>assistant<|channel|>final<|message|><answer> A </answer><|return|>',
    ])

    def gen(prompts):
        t = next(turns)
        return [{"text": t, "token_ids": [ord(c) for c in t], "finish_reason": "stop"}]

    s = AgentState(uid="u", sample_idx=0, token_ids=[ord("x")], prompt_len=1)
    run_search_loop([s], generate_fn=gen,
                    retrieve_fn=lambda qs: [[{"document": {"contents": "T\nbody"}}] for _ in qs],
                    encode_fn=lambda t: [ord(c) for c in t], tool_format="harmony",
                    model_name="openai/gpt-oss-20b")
    assert (s.finish_reason, s.answer, s.n_calls) == ("answer", "A", 1)
    assert "<|start|>functions.search to=assistant" in "".join(map(chr, s.token_ids))
