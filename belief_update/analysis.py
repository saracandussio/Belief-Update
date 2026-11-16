"""Item-level table and summary for the parametric-knowledge baselines.

Per item it combines:
- closed-book pass rate on the composed question (CoT and no-CoT);
- closed-book pass rate on each single hop;
- the same on paraphrases (format ablation);
- pass rate with the search tool.

Categories (closed-book, CoT, threshold tau on pass rates):
- composes:        all hops known, composed question known
- shortcut:        composed question known, some hop not known
                   (answer reached without the hop chain: memorized pair)
- composition_gap: all hops known, composed question not known
- unknown:         neither
`prior_defined` marks chain items whose every hop is known: the candidates
for the prefill / counterfactual experiments, where the prior must exist.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Optional

from belief_update.judge import judge_key
from belief_update.shared.metrics import em

CATEGORIES = ("composes", "shortcut", "composition_gap", "unknown")


def is_correct(
    pred: Optional[str],
    golds: list[str],
    question: str,
    metric: str = "em",
    judge_map: Optional[dict[str, bool]] = None,
    missing: Optional[Counter] = None,
) -> bool:
    if pred is None:
        return False
    if metric == "em":
        return em(pred, golds)
    if metric == "judge":
        if em(pred, golds):
            return True
        verdict = (judge_map or {}).get(judge_key(question, golds, pred))
        if verdict is None and missing is not None:
            missing["judge"] += 1
        return bool(verdict)
    raise ValueError(metric)


def unit_pass_rate(record: dict, **kw) -> Optional[float]:
    preds = record_answers(record)
    if not preds:
        return None
    flags = [is_correct(p, record["golds"], record["question"], **kw) for p in preds]
    return sum(flags) / len(flags)


def record_answers(record: dict) -> list[Optional[str]]:
    if "samples" in record:  # search traces
        return [s["answer"] for s in record["samples"]]
    return record["extracted_answers"]


def _mean(xs) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None


def build_item_table(
    *,
    variants: dict[str, dict],
    items: list[dict],
    closed_question: dict[str, list[dict]],  # mode -> records (target=question)
    closed_hops: dict[str, list[dict]],  # mode -> records (target=hops)
    search: list[dict],
    tau: float = 0.5,
    delta: float = 0.25,
    metric: str = "em",
    judge_map: Optional[dict[str, bool]] = None,
) -> tuple[list[dict], Counter]:
    missing: Counter = Counter()
    kw = dict(metric=metric, judge_map=judge_map, missing=missing)

    def index(records):
        by_id: dict[str, list[dict]] = defaultdict(list)
        for r in records:
            by_id[r["id"]].append(r)
        return by_id

    cq = {m: index(rs) for m, rs in closed_question.items()}
    ch = {m: index(rs) for m, rs in closed_hops.items()}
    se = index(search)

    rows = []
    for it in items:
        v = variants.get(it["id"], {})
        row: dict = {
            "id": it["id"],
            "question": it["question"],
            "golden_answers": it["golden_answers"],
            "type": v.get("type"),
            "chain": v.get("chain"),
            "variants_valid": v.get("valid"),
            "n_hops": len(v.get("hops", [])),
        }
        for mode, by_id in cq.items():
            recs = by_id.get(it["id"], [])
            orig = [unit_pass_rate(r, **kw) for r in recs if r["variant"] == "orig"]
            para = [unit_pass_rate(r, **kw) for r in recs if r["variant"] != "orig"]
            row[f"cb_q_{mode}"] = _mean(orig)
            row[f"cb_q_para_{mode}"] = _mean(para)
        for mode, by_id in ch.items():
            recs = by_id.get(it["id"], [])
            per_hop = []
            per_hop_para = []
            for j in range(row["n_hops"]):
                per_hop.append(
                    _mean(unit_pass_rate(r, **kw) for r in recs
                          if r["hop_idx"] == j and r["variant"] == "orig")
                )
                per_hop_para.append(
                    _mean(unit_pass_rate(r, **kw) for r in recs
                          if r["hop_idx"] == j and r["variant"] != "orig")
                )
            row[f"cb_hops_{mode}"] = per_hop
            row[f"cb_hops_para_{mode}"] = per_hop_para

        srecs = [r for r in se.get(it["id"], []) if r["variant"] == "orig"]
        row["search_q"] = _mean(unit_pass_rate(r, **kw) for r in srecs)
        samples = [s for r in srecs for s in r["samples"]]
        row["search_mean_calls"] = _mean(s["n_calls"] for s in samples)
        row["search_finish"] = dict(Counter(s["finish_reason"] for s in samples))

        # Categories (CoT closed-book)
        q_rate = row.get("cb_q_cot")
        hop_rates = row.get("cb_hops_cot") or []
        if not hop_rates or any(h is None for h in hop_rates):
            hops_known = None
        else:
            hops_known = all(h >= tau for h in hop_rates)
        q_known = None if q_rate is None else q_rate >= tau
        row["q_known"] = q_known
        row["hops_known"] = hops_known
        if q_known is None or hops_known is None:
            row["category"] = None
        else:
            row["category"] = {
                (True, True): "composes",
                (False, True): "shortcut",
                (True, False): "composition_gap",
                (False, False): "unknown",
            }[(hops_known, q_known)]
        para = row.get("cb_q_para_cot")
        row["format_sensitive"] = (
            None if q_rate is None or para is None else abs(q_rate - para) >= delta
        )
        row["prior_defined"] = bool(hops_known) and bool(row["chain"])
        rows.append(row)
    return rows, missing


def summarize(rows: list[dict]) -> dict:
    def avg(key):
        return _mean(r.get(key) for r in rows)

    summary: dict = {"n_items": len(rows)}
    for key in ("cb_q_cot", "cb_q_nocot", "cb_q_para_cot", "cb_q_para_nocot", "search_q",
                "search_mean_calls"):
        if any(key in r for r in rows):
            summary[key] = avg(key)
    for mode in ("cot", "nocot"):
        key = f"cb_hops_{mode}"
        if any(key in r for r in rows):
            summary[f"cb_hop_{mode}"] = _mean(h for r in rows for h in (r.get(key) or []))
    cats = Counter(r["category"] for r in rows if r["category"] is not None)
    summary["categories"] = {c: cats.get(c, 0) for c in CATEGORIES}
    summary["format_sensitive"] = sum(bool(r["format_sensitive"]) for r in rows)
    summary["prior_defined"] = sum(r["prior_defined"] for r in rows)

    # Does search add anything over the prior? (closed-book known x search correct)
    grid = Counter()
    for r in rows:
        if r.get("q_known") is None or r.get("search_q") is None:
            continue
        grid[(("cb_known" if r["q_known"] else "cb_unknown"),
              ("search_ok" if r["search_q"] >= 0.5 else "search_fail"))] += 1
    summary["closed_book_x_search"] = {f"{a}/{b}": n for (a, b), n in sorted(grid.items())}

    by_type: dict[str, dict] = {}
    for t in sorted({str(r["type"]) for r in rows}):
        rs = [r for r in rows if str(r["type"]) == t]
        by_type[t] = {
            "n": len(rs),
            "cb_q_cot": _mean(r.get("cb_q_cot") for r in rs),
            "search_q": _mean(r.get("search_q") for r in rs),
            "categories": dict(Counter(r["category"] for r in rs if r["category"])),
        }
    summary["by_type"] = by_type
    return summary
