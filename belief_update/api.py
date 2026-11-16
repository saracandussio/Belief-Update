"""OpenAI-compatible client for the stronger model (variants) and the judge.

Works with any OpenAI-compatible endpoint: `vllm serve <model>`, OpenRouter,
OpenAI, ... Configure with OPENAI_BASE_URL / OPENAI_API_KEY.
Results are cached in a JSONL keyed by `uid`, so runs are resumable.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

from tqdm import tqdm


def make_chat_fn(model: str, temperature: float = 0.0, max_tokens: int = 2048):
    from openai import OpenAI

    client = OpenAI()

    def chat(prompt: str) -> str:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return r.choices[0].message.content or ""

    return chat


def parse_json_block(text: str) -> Optional[dict]:
    """Parse the first JSON object in `text` (tolerates ```json fences and preambles)."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = [m.group(1)] if m else []
    start = text.find("{")
    end = text.rfind("}")
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


def run_cached(
    *,
    units: list[dict],
    work_fn: Callable[[dict], dict],
    cache_file: Path,
    max_workers: int = 16,
    label: str = "",
) -> dict[str, dict]:
    """Apply `work_fn` to each unit (keyed by `uid`) with a thread pool, append
    results to `cache_file` as they complete. Returns {uid: result}."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict] = {}
    if cache_file.exists():
        with open(cache_file) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    cache[r["uid"]] = r
    todo = [u for u in units if u["uid"] not in cache]
    print(f"[{label}] {len(cache)} cached, {len(todo)} to do.")
    if not todo:
        return cache
    with open(cache_file, "a") as f, ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(work_fn, u): u for u in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=label):
            u = futures[fut]
            try:
                r = fut.result()
            except Exception as e:  # keep going; failed units are retried next run
                print(f"[{label}] {u['uid']} failed: {e}")
                continue
            r["uid"] = u["uid"]
            cache[u["uid"]] = r
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
    return cache
