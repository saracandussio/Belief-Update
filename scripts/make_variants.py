"""Build hop chains and paraphrases with a stronger model (offline vLLM).

Reads data/<dataset>/items.json, writes:
- data/<dataset>/variants_<writer>.jsonl  (raw generations, resumable cache)
- data/<dataset>/variants.json            (validated records, used downstream)
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import fire

from belief_update.aux_model import generate_cached, parse_json_block
from belief_update.shared.io import dump_json
from belief_update.utils import dataset_short_name, model_short_name
from belief_update.variants import build_variants_prompt, validate_variants


def build_variants(items, cache, num_paraphrases, writer_model):
    records = []
    for it in items:
        final = (cache.get(it["id"]) or {}).get("final")
        rec = validate_variants(it, parse_json_block(final), num_paraphrases)
        rec["writer_model"] = writer_model
        records.append(rec)
    return records


def main(
    data_name: str = "2wiki",
    writer_model: str = "openai/gpt-oss-120b",
    num_paraphrases: int = 3,
    temperature: float = 0.7,
    max_tokens: int = 8192,  # reasoning writers spend tokens before the JSON
    max_model_len: int = 16384,
    batch_size: int = 512,
    tensor_parallel_size: int = 1,
    chat_template_kwargs: Optional[dict] = None,
    llm_kwargs: Optional[dict] = None,
):
    data_dir = Path("./data") / dataset_short_name(data_name)
    with open(data_dir / "items.json") as f:
        items = json.load(f)

    units = [{"uid": it["id"], "prompt": build_variants_prompt(it, num_paraphrases)}
             for it in items]
    cache = generate_cached(
        units=units,
        model_name=writer_model,
        cache_file=data_dir / f"variants_{model_short_name(writer_model)}.jsonl",
        label="variants",
        batch_size=batch_size,
        temperature=temperature,
        max_tokens=max_tokens,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        llm_kwargs=llm_kwargs,
        chat_template_kwargs=chat_template_kwargs,
    )

    records = build_variants(items, cache, num_paraphrases, writer_model)
    dump_json(data_dir / "variants.json", records)

    n_valid = sum(r["valid"] for r in records)
    truncated = sum((cache.get(it["id"]) or {}).get("finish_reason") == "length" for it in items)
    issues = {}
    for r in records:
        for i in r["issues"]:
            key = i.split("=")[0] if "=" in i else i
            issues[key] = issues.get(key, 0) + 1
    print(f"Valid: {n_valid}/{len(records)}  chain: {sum(bool(r['chain']) for r in records)}  "
          f"truncated generations: {truncated}")
    print(f"Issues: {json.dumps(issues, indent=2)}")


if __name__ == "__main__":
    fire.Fire(main)
