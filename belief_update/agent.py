"""Multi-turn search-agent loop at the token level.

Backend-agnostic: `generate_fn` maps a list of token-id prompts to one
continuation each, `retrieve_fn` maps queries to documents. The vLLM adapter
lives in `scripts/generate_search_traces.py`; tests use fakes.

Token ids are the source of truth (as in Commitment-Boundary): the model's own
generated ids are kept verbatim, and only tool responses are tokenized by us.
This keeps the traces exactly re-playable for later prefill / truncation runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from belief_update.retrieval import doc_record, passages_to_string
from belief_update.utils import TOOL_FORMATS, extract_answer, parse_tool_call

GenerateFn = Callable[[list[list[int]]], list[dict]]  # -> [{"text", "token_ids", "finish_reason"}]
RetrieveFn = Callable[[list[str]], list[list[dict]]]


@dataclass
class AgentState:
    uid: str
    sample_idx: int
    token_ids: list[int]
    prompt_len: int
    segments: list[dict] = field(default_factory=list)
    n_calls: int = 0
    done: bool = False
    finish_reason: Optional[str] = None
    answer: Optional[str] = None


def run_search_loop(
    states: list[AgentState],
    *,
    generate_fn: GenerateFn,
    retrieve_fn: RetrieveFn,
    encode_fn: Callable[[str], list[int]],
    tool_format: str,
    model_name: str,
    max_calls: int = 5,
    max_context_tokens: Optional[int] = None,
    format_response: Optional[Callable[[str], str]] = None,
) -> list[AgentState]:
    """Advance all states until each answers, stops, or exhausts its budget.

    `format_response` wraps retrieved passages into the tool message (default:
    the fallback template of `tool_format`; at run time it comes from the
    model's chat template, see modeling.derive_response_template).

    finish_reason: answer | no_answer | invalid_call | max_calls | length | context
    """
    fmt = TOOL_FORMATS[tool_format]
    if format_response is None:
        def format_response(result: str) -> str:
            return fmt["response_template"].format(result=result)
    for _ in range(max_calls + 1):
        active = [s for s in states if not s.done]
        if not active:
            break
        outs = generate_fn([s.token_ids for s in active])
        pending: list[tuple[AgentState, str]] = []
        for s, o in zip(active, outs):
            s.token_ids.extend(o["token_ids"])
            text = o["text"]
            s.segments.append({"type": "model", "text": text})
            ans = extract_answer(text, model_name)
            called = text.rstrip().endswith(fmt["call_close"])
            if ans is not None and not called:
                s.done, s.finish_reason, s.answer = True, "answer", ans
            elif called:
                query = parse_tool_call(text, tool_format)
                if query is None:
                    s.done, s.finish_reason = True, "invalid_call"
                elif s.n_calls >= max_calls:
                    s.done, s.finish_reason = True, "max_calls"
                else:
                    pending.append((s, query))
            elif o.get("finish_reason") == "length":
                s.done, s.finish_reason = True, "length"
            else:
                s.done, s.finish_reason = True, "no_answer"

        if pending:
            results = retrieve_fn([q for _, q in pending])
            for (s, q), docs in zip(pending, results):
                resp = format_response(passages_to_string(docs).rstrip("\n"))
                s.token_ids.extend(encode_fn(resp))
                s.n_calls += 1
                s.segments.append(
                    {"type": "tool", "query": q, "docs": [doc_record(d) for d in docs],
                     "text": resp}
                )
                if max_context_tokens is not None and len(s.token_ids) >= max_context_tokens:
                    s.done, s.finish_reason = True, "context"

    for s in states:  # loop exhausted while a call was pending
        if not s.done:
            s.done, s.finish_reason = True, "max_calls"
    return states
