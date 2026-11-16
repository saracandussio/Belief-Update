"""Offline vLLM for the auxiliary models (variants writer, judge).

Same pattern as the generation scripts and Commitment-Boundary: `vllm.LLM`
loaded in-process, prompts built with the model's chat template, generated in
batches. No server. Results are appended to a JSONL cache keyed by `uid`, so a
crashed or timed-out job resumes where it stopped.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from belief_update.utils import get_thinking_tokens

_TRAILING_SPECIALS = ("<|return|>", "<|end|>", "<|im_end|>", "<|endoftext|>", "</s>")


def strip_thinking(text: str, model_name: str) -> Optional[str]:
    """The model's answer without its reasoning. None if the reasoning never
    closed (truncated generation) for a model that always reasons."""
    end = get_thinking_tokens(model_name)["end_token"]
    if end in text:
        text = text.rsplit(end, 1)[-1]
    elif "gpt-oss" in model_name.lower():
        return None
    text = text.strip()
    changed = True
    while changed:
        changed = False
        for tok in _TRAILING_SPECIALS:
            if text.endswith(tok):
                text, changed = text[: -len(tok)].rstrip(), True
    return text


def parse_json_block(text: Optional[str]) -> Optional[dict]:
    """First JSON object in `text` (tolerates ```json fences and preambles)."""
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = [m.group(1)] if m else []
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def load_cache(cache_file: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if cache_file.exists():
        with open(cache_file) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    cache[r["uid"]] = r
    return cache


def generate_cached(
    *,
    units: list[dict],  # each: {"uid": str, "prompt": str}
    model_name: str,
    cache_file: Path,
    label: str = "",
    batch_size: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 8192,
    max_model_len: Optional[int] = 16384,
    tensor_parallel_size: int = 1,
    llm_kwargs: Optional[dict] = None,
    chat_template_kwargs: Optional[dict] = None,
    llm=None,
    tokenizer=None,
) -> dict[str, dict]:
    """Generate one completion per unit; returns {uid: {"uid", "raw", "final", "finish_reason"}}.

    `llm` / `tokenizer` can be injected (tests); otherwise they are loaded only
    if something is left to generate.
    """
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_file)
    todo = [u for u in units if u["uid"] not in cache]
    print(f"[{label}] {len(cache)} cached, {len(todo)} to generate with {model_name}.")
    if not todo:
        return cache

    from belief_update.modeling import build_prompt_ids, load_llm

    if llm is None:
        llm, tokenizer = load_llm(model_name, tensor_parallel_size, None, max_model_len,
                                  llm_kwargs=llm_kwargs)
        from vllm import SamplingParams
    else:  # injected backend (tests): plain kwargs are enough
        def SamplingParams(**kw):  # noqa: N802
            return kw

    params = SamplingParams(n=1, temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                            skip_special_tokens=False)
    with open(cache_file, "a") as f:
        for start in tqdm(range(0, len(todo), batch_size), desc=f"Batches [{label}]"):
            batch = todo[start:start + batch_size]
            prompts = [
                {"prompt_token_ids": build_prompt_ids(tokenizer, u["prompt"], model_name,
                                                      chat_template_kwargs=chat_template_kwargs)}
                for u in batch
            ]
            outputs = llm.generate(prompts, params)
            for u, o in zip(batch, outputs):
                c = o.outputs[0]
                raw = tokenizer.decode(list(c.token_ids), skip_special_tokens=False)
                rec = {"uid": u["uid"], "raw": raw, "final": strip_thinking(raw, model_name),
                       "finish_reason": c.finish_reason}
                cache[u["uid"]] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    return cache
