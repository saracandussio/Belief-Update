"""Expand items into generation units (full question / single hop, original / paraphrase)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def make_uid(item_id: str, target: str, hop_idx: Optional[int], variant: str) -> str:
    part = "q" if target == "question" else f"h{hop_idx}"
    return f"{item_id}|{part}|{variant}"


def build_units(
    items: list[dict],
    variants: Optional[dict[str, dict]],
    target: str,
    include_paraphrases: bool = True,
    only_valid: bool = True,
) -> list[dict]:
    """target='question': the composed question (+ its paraphrases).
    target='hops': every single-hop sub-question (+ its paraphrases)."""
    if target not in {"question", "hops"}:
        raise ValueError(target)
    units: list[dict] = []
    for it in items:
        v = (variants or {}).get(it["id"])
        usable = v is not None and (v.get("valid") or not only_valid)
        if target == "question":
            qs = [("orig", it["question"])]
            if include_paraphrases and usable:
                qs += [(f"para{k}", p) for k, p in enumerate(v["paraphrases"])]
            for variant, q in qs:
                units.append(
                    {
                        "uid": make_uid(it["id"], target, None, variant),
                        "id": it["id"],
                        "target": target,
                        "hop_idx": None,
                        "variant": variant,
                        "question": q,
                        "golds": it["golden_answers"],
                    }
                )
        else:
            if not usable:
                continue
            for j, h in enumerate(v["hops"]):
                qs = [("orig", h["question"])]
                if include_paraphrases:
                    qs += [(f"para{k}", p) for k, p in enumerate(h.get("paraphrases", []))]
                for variant, q in qs:
                    units.append(
                        {
                            "uid": make_uid(it["id"], target, j, variant),
                            "id": it["id"],
                            "target": target,
                            "hop_idx": j,
                            "variant": variant,
                            "question": q,
                            "golds": [h["answer"]],
                        }
                    )
    return units


def load_inputs(
    data_dir: Path, target: str, include_paraphrases: bool, max_questions: Optional[int] = None
) -> list[dict]:
    """Read items.json (+ variants.json if present) and expand into units."""
    with open(data_dir / "items.json") as f:
        items = json.load(f)
    if max_questions is not None:
        items = items[:max_questions]
    variants = None
    vfile = data_dir / "variants.json"
    if vfile.exists():
        with open(vfile) as f:
            variants = {v["id"]: v for v in json.load(f)}
    elif target == "hops" or include_paraphrases:
        print(f"WARNING: {vfile} missing; run scripts/make_variants.py first "
              f"(hops/paraphrases will be skipped).")
    return build_units(items, variants, target, include_paraphrases=include_paraphrases)
