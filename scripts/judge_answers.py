"""Judge the predictions that are not EM-correct (aliases, granularity).

Collects every (question, golds, prediction) triple from the model's result
files, skips EM-correct ones, and asks the judge once per distinct triple.
Output: data/<dataset>/<model>_judge_<judge>.jsonl (used by analyze --metric judge)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fire

from belief_update.analysis import record_answers
from belief_update.api import make_chat_fn, run_cached
from belief_update.judge import build_judge_prompt, judge_key, parse_judge
from belief_update.shared.io import load_json_list
from belief_update.shared.metrics import em
from belief_update.utils import dataset_short_name, model_short_name


def result_files(data_dir: Path, model_name: str) -> list[Path]:
    short = model_short_name(model_name)
    files = sorted(data_dir.glob(f"{short}_closed_book_*.json"))
    files += sorted(data_dir.glob(f"{short}_search_*.json"))
    return [f for f in files if not f.name.endswith("_readable.jsonl")]


def main(model_name: str = "Qwen/Qwen3.8-27B", data_name: str = "2wiki",
         judge_model: str = None, max_workers: int = 16, max_tokens: int = 2048):
    if judge_model is None:
        raise ValueError("Pass --judge_model (served at OPENAI_BASE_URL).")
    data_dir = Path("./data") / dataset_short_name(data_name)
    triples = {}
    for f in result_files(data_dir, model_name):
        for r in load_json_list(f):
            for pred in record_answers(r):
                if pred is None or em(pred, r["golds"]):
                    continue
                key = judge_key(r["question"], r["golds"], pred)
                triples[key] = {"uid": key, "question": r["question"], "golds": r["golds"],
                                "pred": pred}
    print(f"{len(triples)} distinct non-EM predictions to judge")
    chat = make_chat_fn(judge_model, temperature=0.0, max_tokens=max_tokens)

    def work(u):
        raw = chat(build_judge_prompt(u["question"], u["golds"], u["pred"]))
        return {"raw": raw, "correct": parse_judge(raw), "pred": u["pred"]}

    cache = run_cached(
        units=list(triples.values()),
        work_fn=work,
        cache_file=data_dir / (f"{model_short_name(model_name)}_judge_"
                               f"{model_short_name(judge_model)}.jsonl"),
        max_workers=max_workers,
        label="judge",
    )
    verdicts = [c["correct"] for c in cache.values()]
    print(f"judged={len(verdicts)}  correct={sum(v is True for v in verdicts)}  "
          f"unparsed={sum(v is None for v in verdicts)}")


if __name__ == "__main__":
    fire.Fire(main)
