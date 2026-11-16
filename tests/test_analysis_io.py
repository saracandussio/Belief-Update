from belief_update.analysis import build_item_table, summarize
from belief_update.judge import judge_key, parse_judge
from belief_update.shared.io import run_batched


def closed(id_, target, variant, golds, answers, hop_idx=None, q="q"):
    return {"uid": f"{id_}|{target}|{variant}|{hop_idx}", "id": id_, "target": target,
            "hop_idx": hop_idx, "variant": variant, "question": q, "golds": golds,
            "extracted_answers": answers}


def test_categories_and_summary():
    items = [{"id": i, "question": "q", "golden_answers": ["A"]} for i in "1234"]
    variants = {i: {"id": i, "type": "compositional", "chain": True, "valid": True,
                    "hops": [{"answer": "B"}, {"answer": "A"}]} for i in "1234"}
    know = {"1": (True, True), "2": (False, True), "3": (True, False), "4": (False, False)}
    cq, ch = [], []
    for i, (hops_ok, q_ok) in know.items():
        cq.append(closed(i, "question", "orig", ["A"], ["A" if q_ok else "Z"] * 4))
        cq.append(closed(i, "question", "para0", ["A"], ["Z"] * 4))
        for j, g in enumerate(["B", "A"]):
            ch.append(closed(i, "hops", "orig", [g], [g if (hops_ok or j == 1) else "Z"] * 4, j))
    search = [{"id": "4", "variant": "orig", "question": "q", "golds": ["A"],
               "samples": [{"answer": "A", "n_calls": 2, "finish_reason": "answer"}] * 2}]
    rows, _ = build_item_table(variants=variants, items=items, closed_question={"cot": cq},
                               closed_hops={"cot": ch}, search=search)
    cats = {r["id"]: r["category"] for r in rows}
    assert cats == {"1": "composes", "2": "shortcut", "3": "composition_gap", "4": "unknown"}
    assert [r["id"] for r in rows if r["prior_defined"]] == ["1", "3"]
    assert rows[0]["format_sensitive"] is True  # orig 1.0 vs paraphrase 0.0
    s = summarize(rows)
    assert s["categories"]["shortcut"] == 1
    assert s["closed_book_x_search"] == {"cb_unknown/search_ok": 1}


def test_judge_metric():
    items = [{"id": "1", "question": "q", "golden_answers": ["NYC"]}]
    rec = closed("1", "question", "orig", ["NYC"], ["New York City", "Boston"])
    jm = {judge_key("q", ["NYC"], "New York City"): True}
    rows, missing = build_item_table(variants={}, items=items, closed_question={"cot": [rec]},
                                     closed_hops={}, search=[], metric="judge", judge_map=jm)
    assert rows[0]["cb_q_cot"] == 0.5 and missing["judge"] == 1
    assert parse_judge("Correct.") is True and parse_judge("incorrect") is False
    assert parse_judge("The answer is correct") is True and parse_judge("??") is None


def test_run_batched_resumes(tmp_path):
    units = [{"uid": str(i)} for i in range(5)]
    seen = []

    def proc(batch):
        seen.extend(u["uid"] for u in batch)
        return [{**u, "x": 1} for u in batch]

    f = tmp_path / "out.json"
    run_batched(units=units[:3], process_batch=proc, final_file=f, batch_dir=tmp_path / "b",
                batch_size=2, label="t")
    res = run_batched(units=units, process_batch=proc, final_file=f, batch_dir=tmp_path / "b",
                      batch_size=2, label="t", readable_file=tmp_path / "r.jsonl")
    assert seen == ["0", "1", "2", "3", "4"] and len(res) == 5
    assert len(list((tmp_path / "b").glob("*.json"))) == 3
