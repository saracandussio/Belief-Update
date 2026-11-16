import json
from types import SimpleNamespace as NS

from belief_update.aux_model import generate_cached, parse_json_block, strip_thinking


def test_strip_thinking():
    oss = ("<|channel|>analysis<|message|>think<|end|><|start|>assistant<|channel|>final"
           "<|message|>{\"a\": 1}<|return|>")
    assert strip_thinking(oss, "openai/gpt-oss-120b") == '{"a": 1}'
    assert strip_thinking("<|channel|>analysis<|message|>cut", "openai/gpt-oss-120b") is None
    assert strip_thinking("<think>x</think>\n\ncorrect<|im_end|>", "Qwen/Qwen3.8-27B") == "correct"
    assert parse_json_block('noise ```json\n{"chain": true}\n``` tail') == {"chain": True}
    assert parse_json_block(None) is None


class FakeTok:
    def apply_chat_template(self, messages, **kw):
        return "U:" + messages[0]["content"] + "|A:"

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(map(chr, ids))


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def generate(self, prompts, params):
        self.calls += 1
        outs = []
        for p in prompts:
            q = "".join(map(chr, p["prompt_token_ids"])).split("U:")[1].split("|A:")[0]
            t = f"<think>r</think>{json.dumps({'echo': q})}<|im_end|>"
            outs.append(NS(outputs=[NS(token_ids=[ord(c) for c in t], finish_reason="stop")]))
        return outs


def test_generate_cached_resumes(tmp_path):
    units = [{"uid": str(i), "prompt": f"p{i}"} for i in range(5)]
    cache_file = tmp_path / "c.jsonl"
    llm = FakeLLM()
    out = generate_cached(units=units[:3], model_name="Qwen/Qwen3.8-27B", cache_file=cache_file,
                          batch_size=2, llm=llm, tokenizer=FakeTok())
    assert parse_json_block(out["2"]["final"]) == {"echo": "p2"} and llm.calls == 2
    out = generate_cached(units=units, model_name="Qwen/Qwen3.8-27B", cache_file=cache_file,
                          batch_size=10, llm=llm, tokenizer=FakeTok())
    assert len(out) == 5 and llm.calls == 3  # only the 2 new units were generated
    assert sum(1 for _ in open(cache_file)) == 5
