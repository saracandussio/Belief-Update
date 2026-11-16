"""Judge the predictions that are not EM-correct (aliases, granularity), offline vLLM.

Collects every (question, golds, prediction) triple from the model's result
files, skips EM-correct ones, and judges each distinct triple once.
Output: data/<dataset>/<model>_judge_<judge>.jsonl (used by analyze --metric judge)
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import fire

from belief_update.analysis import record_answers
from belief_update.aux_model import generate_cached
from belief_update.judge import build_judge_prompt, judge_key, parse_judge
from belief_update.shared.io import load_json_list
from belief_update.shared.metrics import em
from belief_update.utils import dataset_short_name, model_short_name


def result_files(data_dir: Path, model_name: str) -> list[Path]:
    short = model_short_name(model_name)
    files = sorted(data_dir.glob(f"{short}_closed_book_*.json"))
    files += sorted(data_dir.glob(f"{short}_search_*.json"))
    return files


def main(
    model_name: str = "Qwen/Qwen3.8-27B",
    data_name: str = "2wiki",
    judge_model: str = "openai/gpt-oss-120b",
    max_tokens: int = 2048,  # room for the judge's reasoning
    max_model_len: int = 8192,
    batch_size: int = 1024,
    tensor_parallel_size: int = 1,
    chat_template_kwargs: Optional[dict] = None,
    llm_kwargs: Optional[dict] = None,
):
    data_dir = Path("./data") / dataset_short_name(data_name)
    triples = {}
    for f in result_files(data_dir, model_name):
        for r in load_json_list(f):
            for pred in record_answers(r):
                if pred is None or em(pred, r["golds"]):
                    continue
                key = judge_key(r["question"], r["golds"], pred)
                triples[key] = build_judge_prompt(r["question"], r["golds"], pred)
    print(f"{len(triples)} distinct non-EM predictions to judge")

    out_file = data_dir / (f"{model_short_name(model_name)}_judge_"
                           f"{model_short_name(judge_model)}.jsonl")
    cache = generate_cached(
        units=[{"uid": k, "prompt": p} for k, p in triples.items()],
        model_name=judge_model,
        cache_file=out_file.with_suffix(".raw.jsonl"),
        label="judge",
        batch_size=batch_size,
        temperature=0.0,
        max_tokens=max_tokens,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        llm_kwargs=llm_kwargs,
        chat_template_kwargs=chat_template_kwargs,
    )
    verdicts = {k: parse_judge(c.get("final") or "") for k, c in cache.items()}
    with open(out_file, "w") as f:
        for k, v in verdicts.items():
            f.write(json.dumps({"uid": k, "correct": v}, ensure_ascii=False) + "\n")
    vals = list(verdicts.values())
    print(f"judged={len(vals)}  correct={sum(v is True for v in vals)}  "
          f"unparsed={sum(v is None for v in vals)}")


if __name__ == "__main__":
    fire.Fire(main)
