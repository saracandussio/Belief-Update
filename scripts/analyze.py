"""Item table + summary for one model on one dataset.

Reads whatever result files exist (closed-book question/hops, CoT/no-CoT,
search) and writes:
- outputs/<model>/<dataset>/item_table.jsonl
- outputs/<model>/<dataset>/summary_<metric>.json
- outputs/<model>/<dataset>/prior_defined_ids.txt  (input set for the prefill stage)
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import fire

from belief_update.analysis import build_item_table, summarize
from belief_update.shared.io import dump_json, load_json_list, write_readable
from belief_update.utils import dataset_short_name, model_short_name


def main(
    model_name: str = "Qwen/Qwen3.8-27B",
    data_name: str = "2wiki",
    metric: str = "em",  # em | judge
    judge_model: Optional[str] = None,
    tau: float = 0.5,
    delta: float = 0.25,
):
    data_dir = Path("./data") / dataset_short_name(data_name)
    short = model_short_name(model_name)
    with open(data_dir / "items.json") as f:
        items = json.load(f)
    vfile = data_dir / "variants.json"
    variants = {v["id"]: v for v in load_json_list(vfile)} if vfile.exists() else {}

    closed_q, closed_h = {}, {}
    for mode in ("cot", "nocot"):
        q = load_json_list(data_dir / f"{short}_closed_book_question_{mode}.json")
        h = load_json_list(data_dir / f"{short}_closed_book_hops_{mode}.json")
        if q:
            closed_q[mode] = q
        if h:
            closed_h[mode] = h
    search = []
    for f in sorted(data_dir.glob(f"{short}_search_*.json")):
        search += load_json_list(f)
    print(f"closed-book question: {list(closed_q)}  hops: {list(closed_h)}  "
          f"search records: {len(search)}")

    judge_map = None
    if metric == "judge":
        if judge_model is None:
            raise ValueError("--metric judge needs --judge_model")
        jfile = data_dir / f"{short}_judge_{model_short_name(judge_model)}.jsonl"
        judge_map = {}
        with open(jfile) as f:
            for line in f:
                r = json.loads(line)
                if r.get("correct") is not None:
                    judge_map[r["uid"]] = r["correct"]

    rows, missing = build_item_table(
        variants=variants, items=items, closed_question=closed_q, closed_hops=closed_h,
        search=search, tau=tau, delta=delta, metric=metric, judge_map=judge_map,
    )
    summary = summarize(rows)
    summary.update({"model": model_name, "dataset": data_name, "metric": metric, "tau": tau,
                    "delta": delta, "judge_missing": missing.get("judge", 0)})

    out_dir = Path("./outputs") / short / dataset_short_name(data_name)
    write_readable(rows, out_dir / "item_table.jsonl")
    dump_json(out_dir / f"summary_{metric}.json", summary)
    with open(out_dir / "prior_defined_ids.txt", "w") as f:
        f.write("\n".join(r["id"] for r in rows if r["prior_defined"]))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    fire.Fire(main)
