"""Shared utilities: data loading, model-specific markers, prompts, extraction.

Mirrors `attributions/utils.py` in Commitment-Boundary: one place for the
things that depend on the dataset or on the model family.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Thinking-token configuration (same convention as Commitment-Boundary)
# ---------------------------------------------------------------------------


def get_thinking_tokens(model_name: str) -> dict[str, Any]:
    name = model_name.lower()
    if "gpt-oss" in name:
        return {
            "start_token": "<|channel|>analysis<|message|>",
            "end_token": "<|channel|>final<|message|>",
        }
    if "gemma-4" in name or "gemma4" in name:
        return {"start_token": "<|channel>thought", "end_token": "<channel|>"}
    if "ministral" in name or "mistral" in name:
        return {"start_token": "[THINK]", "end_token": "[/THINK]"}
    # Qwen, DeepSeek and default
    return {"start_token": "<think>", "end_token": "</think>"}


def supports_thinking_switch(model_name: str) -> bool:
    """Models whose chat template accepts `enable_thinking=False` (no-CoT pass
    without any answer suffix). gpt-oss has no such switch: its reasoning is
    always on, with `reasoning_effort` (low|medium|high) as chat-template kwarg."""
    return "qwen3" in model_name.lower()


# ---------------------------------------------------------------------------
# Tool-call formats
# ---------------------------------------------------------------------------
#
# The agent loop works at the token level: it appends the model's own tokens,
# then the tokenized tool response, and continues. `response_template` is the
# exact text the chat template inserts for a tool message. For new model
# families, check it with `scripts/check_template.py` before running.

SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search Wikipedia. Returns the top passages for the query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query."}},
            "required": ["query"],
        },
    },
}

TOOL_FORMATS: dict[str, dict[str, str]] = {
    # Qwen3 (Qwen3-8B, ...): Hermes-style JSON inside <tool_call>.
    "qwen_native": {
        "call_open": "<tool_call>",
        "call_close": "</tool_call>",
        "response_template": (
            "<|im_end|>\n<|im_start|>user\n<tool_response>\n{result}\n</tool_response>"
            "<|im_end|>\n<|im_start|>assistant\n"
        ),
    },
    # Qwen3.5 / 3.6 / 3.8: XML inside <tool_call> (vLLM parser qwen3_coder / qwen3_xml).
    "qwen_xml": {
        "call_open": "<tool_call>",
        "call_close": "</tool_call>",
        "response_template": (
            "<|im_end|>\n<|im_start|>user\n<tool_response>\n{result}\n</tool_response>"
            "<|im_end|>\n<|im_start|>assistant\n"
        ),
    },
    # gpt-oss (harmony):
    #   "...commentary to=functions.search <|constrain|>json<|message|>{...}<|call|>"
    "harmony": {
        "call_open": "to=functions.",
        "call_close": "<|call|>",
        "response_template": (
            "<|start|>functions.search to=assistant<|channel|>commentary<|message|>"
            "{result}<|end|><|start|>assistant"
        ),
    },
    # Search-R1 / ZeroSearch protocol: everything stays in one assistant turn.
    "searchr1": {
        "call_open": "<search>",
        "call_close": "</search>",
        "response_template": "\n\n<information>{result}</information>\n\n",
    },
}
# For the native formats, the response template actually used at run time is
# derived from the model's own chat template (modeling.derive_response_template);
# the strings above are the fallback and the reference for check_template.py.
NATIVE_FORMATS = ("qwen_native", "qwen_xml", "harmony")


def get_tool_format(model_name: str, tool_format: Optional[str] = None) -> str:
    if tool_format is not None:
        if tool_format not in TOOL_FORMATS:
            raise ValueError(f"Unknown tool_format {tool_format!r}; options: {list(TOOL_FORMATS)}")
        return tool_format
    name = model_name.lower()
    if "search-r1" in name or "searchr1" in name or "zerosearch" in name:
        return "searchr1"
    if "gpt-oss" in name:
        return "harmony"
    if re.search(r"qwen3\.[5-9]", name):
        return "qwen_xml"
    if "qwen3" in name:
        return "qwen_native"
    raise ValueError(f"No default tool format for {model_name}; pass --tool_format explicitly.")


def _query_from_json(body: str) -> Optional[str]:
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, str):  # double-encoded
        return _query_from_json(obj)
    if not isinstance(obj, dict):
        return None
    if "arguments" in obj:  # {"name": ..., "arguments": {...}}
        if obj.get("name") != "search":
            return None
        args = obj["arguments"]
        if isinstance(args, str):
            return _query_from_json(args)
        obj = args
    query = obj.get("query") if isinstance(obj, dict) else None
    return query.strip() if isinstance(query, str) and query.strip() else None


def parse_tool_call(text: str, tool_format: str) -> Optional[str]:
    """Return the search query of the last complete tool call in `text`, or None."""
    if tool_format == "harmony":
        matches = re.findall(r"to=functions\.(\w+)(.*?)<\|message\|>(.*?)<\|call\|>", text,
                             flags=re.DOTALL)
        if not matches:
            return None
        name, _header, body = matches[-1]
        if name != "search":
            return None
        q = _query_from_json(body.strip())
        if q is None and body.strip() and not body.strip().startswith("{"):
            q = body.strip()  # plain-text argument
        return q
    fmt = TOOL_FORMATS[tool_format]
    pattern = re.escape(fmt["call_open"]) + r"(.*?)" + re.escape(fmt["call_close"])
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if not matches:
        return None
    body = matches[-1].strip()
    if tool_format == "searchr1":
        return body or None
    if tool_format == "qwen_xml":
        m = re.search(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", body, flags=re.DOTALL)
        if not m or m.group(1).strip() != "search":
            return None
        p = re.search(r"<parameter=query>(.*?)(?:</parameter>|$)", m.group(2), flags=re.DOTALL)
        q = p.group(1).strip() if p else ""
        return q or None
    return _query_from_json(body)


# ---------------------------------------------------------------------------
# Prompts. The format is fixed and logged with every record: the elicitation
# format can change results (Agent Step Value), so it must never vary silently.
# ---------------------------------------------------------------------------

ANSWER_INSTR = (
    "give the final answer inside <answer> and </answer>, without detailed "
    "illustrations. For example, <answer> Beijing </answer>."
)

PROMPTS = {
    "closed_book": (
        "Answer the given question using only your own knowledge. Reason as needed, then "
        + ANSWER_INSTR + " Question: {question}"
    ),
    "search_native": (
        "Answer the given question. You can call the search tool as many times as you want "
        "to look up information. When you have enough information, "
        + ANSWER_INSTR + " Question: {question}"
    ),
    # Verbatim Search-R1 prompt (scripts/data_process/qa_search_test_merge.py).
    "search_searchr1": (
        "Answer the given question. You must conduct reasoning inside <think> and </think> "
        "first every time you get new information. After reasoning, if you find you lack some "
        "knowledge, you can call a search engine by <search> query </search> and it will return "
        "the top searched results between <information> and </information>. You can search as "
        "many times as your want. If you find no further external knowledge needed, you can "
        "directly provide the answer inside <answer> and </answer>, without detailed "
        "illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"
    ),
}


def normalize_question(q: str) -> str:
    q = q.strip()
    return q if q.endswith("?") else q + "?"


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def extract_answer(text: str, model_name: str = "") -> Optional[str]:
    """Last <answer>...</answer> after the end of thinking (falls back to the
    whole text if the end-of-thinking marker is absent). None if missing."""
    end = get_thinking_tokens(model_name)["end_token"]
    tail = text.rsplit(end, 1)[-1] if end in text else text
    matches = _ANSWER_RE.findall(tail) or _ANSWER_RE.findall(text)
    if not matches:
        return None
    ans = matches[-1].strip()
    return ans or None


# ---------------------------------------------------------------------------
# Datasets (FlashRAG versions, as used by Search-R1)
# ---------------------------------------------------------------------------

FLASHRAG_CONFIGS = {
    "hotpotqa": "hotpotqa",
    "2wiki": "2wikimultihopqa",
    "2wikimultihopqa": "2wikimultihopqa",
    "musique": "musique",
    "bamboogle": "bamboogle",
}


def get_data(
    data_name: str,
    max_questions: Optional[int] = None,
    seed: int = 0,
) -> list[dict]:
    """Load a Search-R1 benchmark from RUC-NLPIR/FlashRAG_datasets.

    Uses the test split if present, else dev (same rule as Search-R1).
    With `max_questions`, takes a seeded random subset (not the first N, which
    can be ordered by question type).

    Returns dicts: {id, question, golden_answers, metadata}.
    """
    from datasets import load_dataset

    key = data_name.lower().split("/")[-1]
    if key not in FLASHRAG_CONFIGS:
        raise ValueError(f"Unknown dataset {data_name}; options: {sorted(FLASHRAG_CONFIGS)}")
    ds = load_dataset("RUC-NLPIR/FlashRAG_datasets", FLASHRAG_CONFIGS[key])
    split = "test" if "test" in ds else "dev" if "dev" in ds else "train"
    items = [
        {
            "id": str(x["id"]),
            "question": normalize_question(x["question"]),
            "golden_answers": list(x["golden_answers"]),
            "metadata": x.get("metadata") or {},
            "split": split,
        }
        for x in ds[split]
    ]
    if max_questions is not None and max_questions < len(items):
        items = random.Random(seed).sample(items, max_questions)
    return items


def dataset_short_name(data_name: str) -> str:
    key = data_name.lower().split("/")[-1]
    return FLASHRAG_CONFIGS.get(key, key)


def model_short_name(model_name: str) -> str:
    return model_name.rstrip("/").split("/")[-1]


# ---------------------------------------------------------------------------
# Native decompositions
# ---------------------------------------------------------------------------


def native_decomposition(item: dict) -> Optional[list[dict]]:
    """Gold hop chain from the dataset itself, if it ships one.

    MuSiQue: `question_decomposition` = [{question, answer, ...}], with `#k`
    placeholders that refer to the answer of hop k. They are resolved here with
    the gold answers. Some MuSiQue steps are relation stubs ("Green >> performer")
    rather than questions, so `scripts/make_variants.py` still rewrites every hop
    into a natural standalone question, keeping these gold answers fixed.
    Other datasets: None (the strong model builds the chain).
    """
    meta = item.get("metadata") or {}
    decomp = meta.get("question_decomposition")
    if not decomp:
        return None
    hops: list[dict] = []
    for step in decomp:
        q = resolve_placeholders(step["question"], [h["answer"] for h in hops])
        hops.append({"question": q.strip(), "answer": str(step["answer"]).strip()})
    return hops


def resolve_placeholders(question: str, previous_answers: list[str]) -> str:
    def sub(m: re.Match) -> str:
        k = int(m.group(1))
        return previous_answers[k - 1] if 1 <= k <= len(previous_answers) else m.group(0)

    return re.sub(r"#(\d+)", sub, question)


def supporting_sentences(item: dict, max_chars: int = 3000) -> list[str]:
    """Gold supporting sentences (HotpotQA / 2Wiki style), as "Title: sentence".

    Accepts both layouts found in HF exports:
    - context = {"title": [...], "sentences": [[...], ...]} or [[title, [sents]], ...]
    - supporting_facts = {"title": [...], "sent_id": [...]} or [[title, sent_id], ...]
    Returns [] if the fields are missing or malformed.
    """
    meta = item.get("metadata") or {}
    ctx, sf = meta.get("context"), meta.get("supporting_facts")
    if not ctx or not sf:
        return []
    try:
        if isinstance(ctx, dict):
            paras = dict(zip(ctx["title"], ctx.get("sentences") or ctx["content"]))
        else:
            paras = {t: sents for t, sents in ctx}
        if isinstance(sf, dict):
            facts = list(zip(sf["title"], sf["sent_id"]))
        else:
            facts = [(t, i) for t, i in sf]
    except (KeyError, TypeError, ValueError):
        return []
    out, total = [], 0
    for title, idx in facts:
        sents = paras.get(title)
        if sents is None or not (0 <= int(idx) < len(sents)):
            continue
        line = f"{title}: {str(sents[int(idx)]).strip()}"
        total += len(line)
        if total > max_chars:
            break
        out.append(line)
    return out


def question_type(item: dict) -> Optional[str]:
    meta = item.get("metadata") or {}
    return meta.get("type")
