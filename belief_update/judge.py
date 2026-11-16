"""LLM judge for answers that exact match cannot settle (aliases, granularity).

Closed label set: correct | incorrect. Only predictions that are not already
EM-correct are sent to the judge; EM-correct ones are correct by definition.
Validate the judge on a hand-labelled sample before trusting it.
"""

from __future__ import annotations

from typing import Optional

JUDGE_PROMPT = """Decide whether the predicted answer to the question is correct, given the gold answer(s).
Count it as correct if it refers to the same entity, date or value as a gold answer, even with different wording, an alias, or a more or less specific but compatible form (e.g. "1999" vs "May 1999"). Count it as incorrect if it is a different entity, contradicts the gold answer, or is too vague to identify it.

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
