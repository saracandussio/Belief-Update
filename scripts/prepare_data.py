"""Download a Search-R1 benchmark (FlashRAG version) and freeze the item subset.

Writes data/<dataset>/items.json. Every later stage reads this file, so the
subset is fixed once and shared across models and conditions.
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fire

from belief_update.shared.io import dump_json
from belief_update.utils import dataset_short_name, get_data, native_decomposition, question_type


def main(data_name: str = "2wiki", max_questions: Optional[int] = 500, seed: int = 0,
         overwrite: bool = False):
    out = Path("./data") / dataset_short_name(data_name) / "items.json"
    if out.exists() and not overwrite:
        print(f"{out} exists; pass --overwrite True to resample.")
        return
    items = get_data(data_name, max_questions=max_questions, seed=seed)
    dump_json(out, items)

    n_native = sum(native_decomposition(it) is not None for it in items)
    types = {}
    for it in items:
        types[str(question_type(it))] = types.get(str(question_type(it)), 0) + 1
    print(f"Saved {len(items)} items from split '{items[0]['split']}' to {out}")
    print(f"  native hop chains: {n_native}/{len(items)}")
    print(f"  question types: {types}")
    print(f"  metadata keys (first item): {sorted((items[0].get('metadata') or {}).keys())}")


if __name__ == "__main__":
    fire.Fire(main)
