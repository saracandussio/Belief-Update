"""vLLM / tokenizer helpers. Imported only by the generation scripts (and tests)."""

from __future__ import annotations

import json
import re
from typing import Callable, Optional

from belief_update.utils import (
    NATIVE_FORMATS,
    SEARCH_TOOL_SCHEMA,
    TOOL_FORMATS,
    supports_thinking_switch,
)

DEFAULT_GEN_KWARGS = {"temperature": 0.7, "top_p": 0.9}  # Commitment-Boundary defaults

_RESULT = "RESULT_SENTINEL_7f3a"
_QUERY = "QUERY_SENTINEL_7f3a"


def load_llm(
    model_name: str,
    tensor_parallel_size: int = 1,
    quantization: Optional[str] = None,
    max_model_len: Optional[int] = None,
    gpu_memory_utilization: float = 0.9,
    llm_kwargs: Optional[dict] = None,
):
    """`llm_kwargs` goes straight to vllm.LLM, e.g. for Qwen3.8 (which ships a
    vision tower): {"limit_mm_per_prompt": {"image": 0}}."""
    from transformers import AutoTokenizer
    from vllm import LLM

    kwargs = dict(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        quantization=quantization,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    kwargs.update(default_llm_kwargs(model_name))
    kwargs.update(llm_kwargs or {})
    llm = LLM(**kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return llm, tokenizer


def default_llm_kwargs(model_name: str) -> dict:
    """Qwen3.5+ checkpoints (Qwen3.8 included) ship a vision tower: text-only here."""
    if re.search(r"qwen3\.[5-9]", model_name.lower()):
        return {"limit_mm_per_prompt": {"image": 0, "video": 0}}
    return {}


def template_kwargs(model_name: str, thinking: bool, chat_template_kwargs: Optional[dict]) -> dict:
    kwargs = dict(chat_template_kwargs or {})
    if supports_thinking_switch(model_name):
        kwargs.setdefault("enable_thinking", thinking)
    elif not thinking:
        raise ValueError(f"{model_name}: no template switch for disabling thinking.")
    return kwargs


def build_prompt_ids(
    tokenizer,
    user_content: str,
    model_name: str,
    *,
    thinking: bool = True,
    with_search_tool: bool = False,
    chat_template_kwargs: Optional[dict] = None,
) -> list[int]:
    kwargs = template_kwargs(model_name, thinking, chat_template_kwargs)
    if with_search_tool:
        kwargs["tools"] = [SEARCH_TOOL_SCHEMA]
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        add_generation_prompt=True,
        tokenize=False,
        **kwargs,
    )
    return tokenizer.encode(text, add_special_tokens=False)


def derive_response_template(
    tokenizer,
    tool_format: str,
    model_name: str,
    chat_template_kwargs: Optional[dict] = None,
) -> tuple[Optional[str], bool]:
    """Read the tool-response wrapper off the model's own chat template.

    Renders [user, assistant tool call, tool result] + generation prompt and
    returns the text between the end of the call and the start of the next
    assistant turn, with `{result}` in place of the tool output. The second
    value is True when the template JSON-encodes tool outputs.
    Returns (None, False) if the template cannot be rendered this way.
    """
    if tool_format not in NATIVE_FORMATS:
        return TOOL_FORMATS[tool_format]["response_template"], False
    call_close = TOOL_FORMATS[tool_format]["call_close"]
    messages = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_0", "type": "function",
            "function": {"name": "search", "arguments": {"query": _QUERY}},
        }]},
        {"role": "tool", "tool_call_id": "call_0", "name": "search", "content": _RESULT},
    ]
    try:
        rendered = tokenizer.apply_chat_template(
            messages, tools=[SEARCH_TOOL_SCHEMA], add_generation_prompt=True, tokenize=False,
            **template_kwargs(model_name, True, chat_template_kwargs),
        )
    except Exception as e:  # template rejects this shape
        print(f"derive_response_template: render failed ({e})")
        return None, False
    r = rendered.find(_RESULT)
    c = rendered.rfind(call_close, 0, r)
    if r == -1 or c == -1:
        return None, False
    tail = rendered[c + len(call_close):]
    json_encoded = f'"{_RESULT}"' in tail
    tail = tail.replace("{", "{{").replace("}", "}}")
    tail = tail.replace(f'"{_RESULT}"' if json_encoded else _RESULT, "{result}")
    return tail, json_encoded


def make_response_formatter(template: str, json_encoded: bool = False) -> Callable[[str], str]:
    def fmt(result: str) -> str:
        return template.format(result=json.dumps(result) if json_encoded else result)

    return fmt


def make_generate_fn(llm, sampling_params, tokenizer):
    """Adapter: list of token-id prompts -> one continuation each.

    The returned text is decoded from the generated ids (special tokens kept),
    so it always matches what is appended to the context, including stop
    tokens such as gpt-oss's <|call|>.
    """

    def generate(prompts: list[list[int]]) -> list[dict]:
        outputs = llm.generate([{"prompt_token_ids": p} for p in prompts], sampling_params)
        res = []
        for o in outputs:
            c = o.outputs[0]
            ids = list(c.token_ids)
            res.append({
                "text": tokenizer.decode(ids, skip_special_tokens=False),
                "token_ids": ids,
                "finish_reason": c.finish_reason,
            })
        return res

    return generate
