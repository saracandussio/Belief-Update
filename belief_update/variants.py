"""Sub-questions and paraphrases built by a stronger model.

For every item the stronger model returns:
- `hops`: the chain of standalone single-hop questions with short answers
  (anchored to the dataset's own chain when it has one, e.g. MuSiQue);
- `paraphrases`: rewordings of the full question that keep every hop implicit;
- per-hop paraphrases.
These feed the per-hop closed-book test (parametric prior per hop, shortcut
detection) and the format ablation.
"""

from __future__ import annotations

import json
from typing import Optional

from belief_update.shared.metrics import cover_em, em, normalize_answer
from belief_update.utils import native_decomposition, normalize_question, question_type

VARIANTS_PROMPT = """You are preparing controlled test material for a study of multi-hop question answering.

Question: {question}
Gold answer(s): {golds}
{extra}
Tasks:
1. Decide whether the question is a sequential chain, where the answer to one hop is needed to ask the next ("chain": true), or something else, e.g. a comparison between two entities ("chain": false).
2. Write "hops": the minimal list of single-hop questions that together answer the question, in order. Each hop must be a natural, standalone question that names its subject explicitly (no pronouns, no "#1"). Each hop has a short "answer" (an entity, date or short phrase, no explanation). For a chain, the answer of the last hop must be the gold answer. For a non-chain question, list the facts that need to be looked up, and the last hop is the step that produces the gold answer.{native_rule}
3. Write {k} "paraphrases" of the original question. Keep the meaning and the gold answer identical, keep every intermediate entity implicit (never reveal a bridge entity), and vary wording and syntax, not just synonyms.
4. For each hop, write {k} paraphrases in "hop_paraphrases" (a list of lists, aligned with "hops").

Return only JSON:
{{"chain": true, "hops": [{{"question": "...", "answer": "..."}}], "paraphrases": ["..."], "hop_paraphrases": [["..."]]}}"""


def build_variants_prompt(item: dict, k: int) -> str:
    extra_lines = []
    qtype = question_type(item)
    if qtype:
        extra_lines.append(f"Dataset question type: {qtype}")
    native = native_decomposition(item)
    native_rule = ""
    if native:
        chain = "\n".join(f"  {i + 1}. {h['question']} -> {h['answer']}" for i, h in enumerate(native))
        extra_lines.append(
            "Reference chain from the dataset (#k refers to the answer of hop k; some steps are "
            "relation stubs such as 'X >> relation'):\n" + chain
        )
        native_rule = (
            " Use exactly one hop per reference step, in the same order, and keep the reference "
            "answers verbatim; only rewrite the questions."
        )
    evid = (item.get("metadata") or {}).get("evidences")
    if evid:
        extra_lines.append("Evidence triples: " + json.dumps(evid, ensure_ascii=False))
    extra = ("\n".join(extra_lines) + "\n") if extra_lines else ""
    return VARIANTS_PROMPT.format(
        question=item["question"],
        golds=" | ".join(item["golden_answers"]),
        extra=extra,
        k=k,
        native_rule=native_rule,
    )


def validate_variants(item: dict, out: Optional[dict], k: int) -> dict:
    """Normalize the model output and attach `valid` + `issues`."""
    issues: list[str] = []
    native = native_decomposition(item)
    rec = {
        "id": item["id"],
        "question": item["question"],
        "golden_answers": item["golden_answers"],
        "type": question_type(item),
        "hops_source": "native" if native else "llm",
        "chain": None,
        "hops": [],
        "paraphrases": [],
    }
    if out is None:
        rec.update(valid=False, issues=["unparseable"])
        return rec

    rec["chain"] = bool(out.get("chain", True))
    hops = out.get("hops") or []
    hop_paras = out.get("hop_paraphrases") or []
    if not (1 <= len(hops) <= 4):
        issues.append(f"n_hops={len(hops)}")
    if len(hop_paras) != len(hops):
        issues.append("hop_paraphrases misaligned")
    for j, h in enumerate(hops):
        q, a = str(h.get("question", "")).strip(), str(h.get("answer", "")).strip()
        if not q or not a:
            issues.append(f"hop{j} empty")
        if "#" in q:
            issues.append(f"hop{j} has placeholder")
        paras = hop_paras[j] if j < len(hop_paras) and isinstance(hop_paras[j], list) else []
        rec["hops"].append(
            {
                "question": normalize_question(q) if q else q,
                "answer": a,
                "paraphrases": [normalize_question(p) for p in paras if str(p).strip()][:k],
            }
        )
    if native:
        if len(native) != len(hops):
            issues.append("hop count differs from native chain")
        else:
            for j, (n, h) in enumerate(zip(native, rec["hops"])):
                if not em(h["answer"], n["answer"]):
                    issues.append(f"hop{j} answer changed from native")
                h["answer"] = n["answer"]  # gold chain wins
    if rec["hops"] and rec["chain"]:
        last = rec["hops"][-1]["answer"]
        if not (em(last, item["golden_answers"]) or cover_em(last, item["golden_answers"])):
            issues.append("last hop answer != gold")
    paras = [normalize_question(p) for p in (out.get("paraphrases") or []) if str(p).strip()]
    # Drop paraphrases that reveal a bridge entity absent from the original question.
    bridges = [h["answer"] for h in rec["hops"][:-1]] if rec["chain"] else []
    q_norm = normalize_answer(item["question"])
    leaky = [
        p for p in paras
        if any(
            normalize_answer(b) and normalize_answer(b) in normalize_answer(p)
            and normalize_answer(b) not in q_norm
            for b in bridges
        )
    ]
    if leaky:
        issues.append(f"dropped {len(leaky)} paraphrases leaking a bridge entity")
    rec["paraphrases"] = [p for p in paras if p not in leaky][:k]
    if len(rec["paraphrases"]) < k:
        issues.append(f"only {len(rec['paraphrases'])} paraphrases")
    rec["issues"] = issues
    # Paraphrase shortfalls are tolerated; structural problems are not.
    rec["valid"] = not [i for i in issues if not i.startswith(("only ", "dropped "))]
    return rec
