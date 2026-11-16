"""Batched, resumable result files (same layout as Commitment-Boundary).

Each pass writes:
- `<final_file>`: JSON list with one record per unit (question, hop, ...);
- `<batch_dir>/batch_XXXXXX_YYYYYY.json`: per-batch safety copies;
- `<readable_file>`: JSONL without the bulky fields, for quick inspection.
Records are keyed by `uid`, so re-running a pass skips what is already done.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from tqdm import tqdm


def load_json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:  # corrupted partial file
        print(f"Could not read {path} ({e}); starting from scratch.")
        return []


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def run_batched(
    *,
    units: list[dict],
    process_batch: Callable[[list[dict]], list[dict]],
    final_file: Path,
    batch_dir: Path,
    batch_size: int,
    resume: bool = True,
    label: str = "",
    readable_file: Optional[Path] = None,
    readable_drop: tuple[str, ...] = (),
) -> list[dict]:
    """Run `process_batch` over `units` in batches, saving after each batch."""
    done = load_json_list(final_file) if resume else []
    done_uids = {r["uid"] for r in done}
    todo = [u for u in units if u["uid"] not in done_uids]
    if done:
        print(f"[{label}] {len(done)} units already processed, {len(todo)} remaining.")
    else:
        print(f"[{label}] Processing {len(todo)} units in batches of {batch_size}.")

    batch_dir.mkdir(parents=True, exist_ok=True)
    all_results = list(done)
    for start in tqdm(range(0, len(todo), batch_size), desc=f"Batches [{label}]"):
        batch = todo[start:start + batch_size]
        results = process_batch(batch)
        offset = len(done) + start
        dump_json(batch_dir / f"batch_{offset:06d}_{offset + len(results):06d}.json", results)
        all_results.extend(results)
        dump_json(final_file, all_results)

    if readable_file is not None:
        write_readable(all_results, readable_file, drop=readable_drop)
    return all_results


def write_readable(records: list[dict], path: Path, drop: tuple[str, ...] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps({k: v for k, v in r.items() if k not in drop}, ensure_ascii=False))
            f.write("\n")
