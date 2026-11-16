"""Closed-book answers: how much of each item is already in the parameters.

Targets:
- question: the composed multi-hop question (+ paraphrases);
- hops:     every single-hop sub-question from variants.json (+ paraphrases).

Passes (as in Commitment-Boundary): CoT (thinking on) and, with --no_cot True,
an additional no-CoT pass. The no-CoT pass uses the chat template's own switch
(`enable_thinking=False`); no answer suffix is appended. The answer format is
fixed by the instruction (`<answer>...</answer>`) in both passes.

Output: data/<dataset>/<model>_closed_book_<target>_<cot|nocot>.json
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fire

from belief_update.shared.io import run_batched
from belief_update.shared.metrics import pass_rate, score_answer
from belief_update.units import load_inputs
from belief_update.utils import (
    PROMPTS,
    dataset_short_name,
    extract_answer,
    model_short_name,
    supports_thinking_switch,
)


def run_pass(*, llm, tokenizer, units, model_name, thinking, num_out, gen_kwargs,
             batch_size, resume, out_file, label, chat_template_kwargs=None):
    from vllm import SamplingParams

    from belief_update.modeling import build_prompt_ids

    params = SamplingParams(n=num_out, skip_special_tokens=False, **gen_kwargs)
    template = "closed_book"

    def process(batch):
        prompts = [
            build_prompt_ids(tokenizer, PROMPTS[template].format(question=u["question"]),
                             model_name, thinking=thinking,
                             chat_template_kwargs=chat_template_kwargs)
            for u in batch
        ]
        outputs = llm.generate([{"prompt_token_ids": p} for p in prompts], params)
        results = []
        for u, p, o in zip(batch, prompts, outputs):
            traces = [c.text for c in o.outputs]
            answers = [extract_answer(t, model_name) for t in traces]
            scores = [score_answer(a, u["golds"]) for a in answers]
            results.append({
                **u,
                "mode": "cot" if thinking else "nocot",
                "prompt_template": template,
                "chat_template_kwargs": chat_template_kwargs or {},
                "prompt_tokens": p,
                "traces": traces,
                "traces_tokens": [list(c.token_ids) for c in o.outputs],
                "finish_reasons": [c.finish_reason for c in o.outputs],
                "extracted_answers": answers,
                "scores": scores,
            })
        return results

    records = run_batched(
        units=units,
        process_batch=process,
        final_file=out_file,
        batch_dir=out_file.parent / "batches" / out_file.stem,
        batch_size=batch_size,
        resume=resume,
        label=label,
        readable_file=out_file.with_name(out_file.stem + "_readable.jsonl"),
        readable_drop=("prompt_tokens", "traces_tokens", "traces"),
    )
    sanity(records, label)
    return records


def sanity(records, label):
    """Quick EM pass rates, split by variant, like the CB sanity check."""
    for name, keep in (("orig", lambda r: r["variant"] == "orig"),
                       ("paraphrase", lambda r: r["variant"] != "orig")):
        rs = [r for r in records if keep(r)]
        if not rs:
            continue
        rates = [pass_rate(s["em"] for s in r["scores"]) for r in rs]
        cover = [pass_rate(s["cover_em"] for s in r["scores"]) for r in rs]
        no_ans = sum(a is None for r in rs for a in r["extracted_answers"])
        n_tr = sum(len(r["extracted_answers"]) for r in rs)
        print(f"[{label}] {name}: n={len(rs)}  EM pass rate={sum(rates) / len(rates):.3f}  "
              f"cover-EM={sum(cover) / len(cover):.3f}  no <answer>: {no_ans}/{n_tr}")


def main(
    model_name: str = "Qwen/Qwen3.8-27B",
    data_name: str = "2wiki",
    target: str = "question",  # question | hops
    num_out: int = 8,
    batch_size: int = 64,
    resume: bool = True,
    no_cot: bool = True,  # if True, additionally run a no-CoT pass
    include_paraphrases: bool = True,
    max_questions: Optional[int] = None,
    tensor_parallel_size: int = 1,
    quantization: Optional[str] = None,
    max_model_len: Optional[int] = 16384,  # prompt + 8192 generated tokens
    chat_template_kwargs: Optional[dict] = None,  # e.g. {"reasoning_effort": "medium"}
    llm_kwargs: Optional[dict] = None,  # passed to vllm.LLM
    **gen_kwargs,
):
    from belief_update.modeling import DEFAULT_GEN_KWARGS, load_llm

    data_dir = Path("./data") / dataset_short_name(data_name)
    units = load_inputs(data_dir, target, include_paraphrases, max_questions)
    print(f"{len(units)} units (target={target})")
    if not units:
        return

    gen_kwargs = {**DEFAULT_GEN_KWARGS, "max_tokens": 8192, **gen_kwargs}
    llm, tokenizer = load_llm(model_name, tensor_parallel_size, quantization, max_model_len,
                              llm_kwargs=llm_kwargs)
    short = model_short_name(model_name)

    run_pass(llm=llm, tokenizer=tokenizer, units=units, model_name=model_name, thinking=True,
             num_out=num_out, gen_kwargs=gen_kwargs, batch_size=batch_size, resume=resume,
             out_file=data_dir / f"{short}_closed_book_{target}_cot.json",
             label=f"closed-book {target} CoT", chat_template_kwargs=chat_template_kwargs)

    if not no_cot:
        return
    if not supports_thinking_switch(model_name):
        print(f"Skipping no-CoT pass: {model_name} has no enable_thinking switch "
              f"(and this repo does not use answer suffixes).")
        return
    nocot_kwargs = {**gen_kwargs, "max_tokens": min(gen_kwargs["max_tokens"], 1024)}
    run_pass(llm=llm, tokenizer=tokenizer, units=units, model_name=model_name, thinking=False,
             num_out=num_out, gen_kwargs=nocot_kwargs, batch_size=batch_size, resume=resume,
             out_file=data_dir / f"{short}_closed_book_{target}_nocot.json",
             label=f"closed-book {target} no-CoT", chat_template_kwargs=chat_template_kwargs)


if __name__ == "__main__":
    fire.Fire(main)
