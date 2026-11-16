"""Answer normalization and QA metrics.

EM / F1 follow the SQuAD normalization used by Search-R1 and FlashRAG, so the
numbers are comparable with the papers built on that suite. `cover_em` is the
lenient variant (a gold answer appears inside the normalized prediction).
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, Optional, Sequence


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _as_list(golds: str | Sequence[str]) -> list[str]:
    return [golds] if isinstance(golds, str) else list(golds)


def em(pred: Optional[str], golds: str | Sequence[str]) -> bool:
    if pred is None:
        return False
    p = normalize_answer(pred)
    return any(p == normalize_answer(g) for g in _as_list(golds))


def cover_em(pred: Optional[str], golds: str | Sequence[str]) -> bool:
    if pred is None:
        return False
    p = normalize_answer(pred)
    return any(normalize_answer(g) and normalize_answer(g) in p for g in _as_list(golds))


def f1(pred: Optional[str], golds: str | Sequence[str]) -> float:
    if pred is None:
        return 0.0
    best = 0.0
    p_tokens = normalize_answer(pred).split()
    for g in _as_list(golds):
        g_tokens = normalize_answer(g).split()
        # yes/no/noanswer must match exactly (SQuAD/HotpotQA convention)
        if normalize_answer(pred) in {"yes", "no", "noanswer"} or normalize_answer(g) in {
            "yes", "no", "noanswer"
        }:
            best = max(best, float(normalize_answer(pred) == normalize_answer(g)))
            continue
        common = Counter(p_tokens) & Counter(g_tokens)
        n_same = sum(common.values())
        if n_same == 0:
            continue
        prec = n_same / len(p_tokens)
        rec = n_same / len(g_tokens)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def score_answer(pred: Optional[str], golds: str | Sequence[str]) -> dict:
    return {
        "em": em(pred, golds),
        "cover_em": cover_em(pred, golds),
        "f1": f1(pred, golds),
    }


def pass_rate(flags: Iterable[bool]) -> Optional[float]:
    flags = list(flags)
    if not flags:
        return None
    return sum(bool(f) for f in flags) / len(flags)
