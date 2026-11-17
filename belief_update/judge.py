"""LLM judge for answers that exact match cannot settle (aliases, granularity).

Closed label set: correct | incorrect. Only predictions that are not already
EM-correct are sent to the judge; EM-correct ones are correct by definition.
Validate the judge on a hand-labelled sample before trusting it.
"""

from __future__ import annotations

from typing import Optional

JUDGE_PROMPT = """Decide whether the predicted answer to the question is correct, given the gold answer(s).

Correct only if the prediction identifies the same entity, date or value as a gold answer. Allowed:
- aliases, translations or transliterations of the same name, and titles or honorifics ("Sir Nicholas Bacon" vs "Nicholas Bacon");
- extra detail that is consistent with the gold ("MECC (Minnesota Educational Computing Consortium)");
- a coarser or finer form of the same date, or a small rounding of the same number ("1999" vs "May 1999", "11" vs "10.8").

Incorrect if:
- it names a different person, even when part of the name matches (a different given name, a relative with the same surname, the subject of the question instead of the person asked about);
- it is a range or a list of options, unless every option matches the gold;
- it is a more generic category than the gold ("merchant" for "wool merchant") or a different specific one;
- it contradicts the gold, or is too vague to identify it.

Question: {question}
Gold answer(s): {golds}
Predicted answer: {pred}

Reply with exactly one word: correct or incorrect."""


def build_judge_prompt(question: str, golds: list[str], pred: str) -> str:
    return JUDGE_PROMPT.format(question=question, golds=" | ".join(golds), pred=pred)


def parse_judge(text: str) -> Optional[bool]:
    t = text.strip().lower()
    if t.startswith("incorrect"):
        return False
    if t.startswith("correct"):
        return True
    # tolerate a short preamble
    last = t.split()[-1].strip(".!*") if t.split() else ""
    return {"correct": True, "incorrect": False}.get(last)


def judge_key(question: str, golds: list[str], pred: str) -> str:
    """Cache key: identical (question, golds, pred) triples are judged once."""
    return "\x1f".join([question, "|".join(golds), pred.strip().lower()])
