"""Build hop chains and paraphrases with a stronger model (OpenAI-compatible API).

Reads data/<dataset>/items.json, writes:
- data/<dataset>/variants_<writer>.jsonl  (raw cache, resumable)
- data/<dataset>/variants.json            (validated records, used downstream)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import fire

from belief_update.api import make_chat_fn, parse_json_block, run_cached
from belief_update.shared.io import dump_json
from belief_update.utils import dataset_short_name, model_short_name
from belief_update.variants import build_variants_prompt, validate_variants


def main(
    data_name: str = "2wiki",
    writer_model: str = None,  # required: a model clearly stronger than the subject
    num_paraphrases: int = 3,
    max_workers: int = 16,
    temperature: float = 0.7,
):
    if writer_model is None:
        raise ValueError("Pass --writer_model (served at OPENAI_BASE_URL).")
    data_dir = Path("./data") / dataset_short_name(data_name)
    with open(data_dir / "items.json") as f:
        items = json.load(f)
    chat = make_chat_fn(writer_model, temperature=temperature, max_tokens=4096)

    def work(unit):
        raw = chat(build_variants_prompt(unit["item"], num_paraphrases))
        return {"raw": raw}

    units = [{"uid": it["id"], "item": it} for it in items]
    cache = run_cached(
        units=units,
        work_fn=work,
        cache_file=data_dir / f"variants_{model_short_name(writer_model)}.jsonl",
        max_workers=max_workers,
        label="variants",
    )

    records = []
    for it in items:
        raw = cache.get(it["id"], {}).get("raw")
        rec = validate_variants(it, parse_json_block(raw) if raw else None, num_paraphrases)
        rec["writer_model"] = writer_model
        records.append(rec)
    dump_json(data_dir / "variants.json", records)

    n_valid = sum(r["valid"] for r in records)
    issues = {}
    for r in records:
        for i in r["issues"]:
            key = i.split("=")[0] if "=" in i else i
            issues[key] = issues.get(key, 0) + 1
    print(f"Valid: {n_valid}/{len(records)}  chain: {sum(bool(r['chain']) for r in records)}")
    print(f"Issues: {json.dumps(issues, indent=2)}")


if __name__ == "__main__":
    fire.Fire(main)
