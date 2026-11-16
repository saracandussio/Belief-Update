"""Search-agent traces on the composed questions (clean retrieval, no perturbation).

Each question is sampled `num_out` times; every sample is an independent
multi-turn trajectory (model turn -> search -> model turn ... -> <answer>).
Retrieval: Search-R1 server (E5 over Wikipedia 2018, top-3 by default).

Tool formats:
- qwen_native: Qwen3 tool calling (`<tool_call>` JSON);
- qwen_xml:    Qwen3.5/3.6/3.8 tool calling (`<tool_call><function=search>` XML);
- harmony:     gpt-oss tool calling (`to=functions.search ... <|call|>`);
- searchr1:    Search-R1 tags (`<search>`, `<information>`), verbatim prompt.
For the native formats the tool is declared through the chat template, and the
tool-response wrapper is read off the same template at start-up.

Output: data/<dataset>/<model>_search_<tool_format>.json
Each sample keeps the full token ids and a list of segments (model turns and
tool calls with retrieved docs), so trajectories can be replayed later with a
prefill or a perturbed tool output at any step.
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fire

from belief_update.shared.io import run_batched
from belief_update.shared.io import write_readable
from belief_update.shared.metrics import pass_rate, score_answer
from belief_update.units import load_inputs
from belief_update.utils import (
    PROMPTS,
    TOOL_FORMATS,
    dataset_short_name,
    get_tool_format,
    model_short_name,
)


def main(
    model_name: str = "Qwen/Qwen3.8-27B",
    data_name: str = "2wiki",
    tool_format: Optional[str] = None,  # qwen_native | searchr1 (default: from model name)
    retriever_url: str = "http://127.0.0.1:8000/retrieve",
    topk: int = 3,
    num_out: int = 8,
    max_calls: int = 5,
    max_tokens_per_turn: int = 4096,
    max_model_len: int = 32768,
    batch_size: int = 32,
    resume: bool = True,
    include_paraphrases: bool = False,
    max_questions: Optional[int] = None,
    tensor_parallel_size: int = 1,
    quantization: Optional[str] = None,
    chat_template_kwargs: Optional[dict] = None,  # e.g. {"reasoning_effort": "medium"}
    llm_kwargs: Optional[dict] = None,  # passed to vllm.LLM
    **gen_kwargs,
):
    from vllm import SamplingParams

    from belief_update.agent import AgentState, run_search_loop
    from belief_update.modeling import (
        DEFAULT_GEN_KWARGS,
        build_prompt_ids,
        derive_response_template,
        load_llm,
        make_generate_fn,
        make_response_formatter,
    )
    from belief_update.retrieval import SearchR1Retriever

    fmt_name = get_tool_format(model_name, tool_format)
    fmt = TOOL_FORMATS[fmt_name]
    native = fmt_name != "searchr1"
    template = "search_native" if native else "search_searchr1"

    data_dir = Path("./data") / dataset_short_name(data_name)
    units = load_inputs(data_dir, "question", include_paraphrases, max_questions)
    print(f"{len(units)} units, tool format {fmt_name}")

    llm, tokenizer = load_llm(model_name, tensor_parallel_size, quantization, max_model_len,
                              llm_kwargs=llm_kwargs)
    response_template, json_encoded = derive_response_template(
        tokenizer, fmt_name, model_name, chat_template_kwargs)
    if response_template is None:
        response_template, json_encoded = fmt["response_template"], False
        print(f"WARNING: could not derive the tool-response wrapper from the chat template; "
              f"using the built-in {fmt_name} fallback. Run scripts/check_template.py.")
    elif response_template != fmt["response_template"]:
        print(f"NOTE: chat-template wrapper differs from the built-in {fmt_name} fallback; "
              f"using the chat template's.")
    print(f"Tool response wrapper: {response_template!r} (json_encoded={json_encoded})")
    format_response = make_response_formatter(response_template, json_encoded)
    retriever = SearchR1Retriever(retriever_url, topk=topk)
    gen_kwargs = {**DEFAULT_GEN_KWARGS, **gen_kwargs}
    params = SamplingParams(
        n=1,
        max_tokens=max_tokens_per_turn,
        stop=[fmt["call_close"]],
        include_stop_str_in_output=True,
        skip_special_tokens=False,
        **gen_kwargs,
    )
    generate_fn = make_generate_fn(llm, params, tokenizer)

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    def process(batch):
        states = []
        for u in batch:
            content = PROMPTS[template].format(question=u["question"])
            prompt_ids = build_prompt_ids(
                tokenizer, content, model_name, thinking=True,
                with_search_tool=native, chat_template_kwargs=chat_template_kwargs,
            )
            for k in range(num_out):
                states.append(AgentState(uid=u["uid"], sample_idx=k, token_ids=list(prompt_ids),
                                         prompt_len=len(prompt_ids)))
        run_search_loop(
            states,
            generate_fn=generate_fn,
            retrieve_fn=retriever,
            encode_fn=encode,
            tool_format=fmt_name,
            model_name=model_name,
            max_calls=max_calls,
            max_context_tokens=max_model_len - max_tokens_per_turn,
            format_response=format_response,
        )
        by_uid = {}
        for s in states:
            by_uid.setdefault(s.uid, []).append(s)
        results = []
        for u in batch:
            samples = []
            for s in sorted(by_uid[u["uid"]], key=lambda s: s.sample_idx):
                samples.append({
                    "sample_idx": s.sample_idx,
                    "prompt_len": s.prompt_len,
                    "token_ids": s.token_ids,
                    "segments": s.segments,
                    "n_calls": s.n_calls,
                    "finish_reason": s.finish_reason,
                    "answer": s.answer,
                    "scores": score_answer(s.answer, u["golds"]),
                })
            results.append({**u, "mode": "cot", "tool_format": fmt_name,
                            "prompt_template": template,
                            "chat_template_kwargs": chat_template_kwargs or {},
                            "tool_response_template": response_template,
                            "retriever": {"url": retriever_url, "topk": topk},
                            "samples": samples})
        return results

    out_file = data_dir / f"{model_short_name(model_name)}_search_{fmt_name}.json"
    records = run_batched(
        units=units,
        process_batch=process,
        final_file=out_file,
        batch_dir=out_file.parent / "batches" / out_file.stem,
        batch_size=batch_size,
        resume=resume,
        label="search",
    )
    write_readable(
        [{**{k: v for k, v in r.items() if k != "samples"},
          "samples": [{"answer": s["answer"], "em": s["scores"]["em"], "n_calls": s["n_calls"],
                       "finish_reason": s["finish_reason"],
                       "queries": [g["query"] for g in s["segments"] if g["type"] == "tool"]}
                      for s in r["samples"]]}
         for r in records],
        out_file.with_name(out_file.stem + "_readable.jsonl"),
    )

    rates = [pass_rate(s["scores"]["em"] for s in r["samples"]) for r in records]
    calls = [s["n_calls"] for r in records for s in r["samples"]]
    reasons = {}
    for r in records:
        for s in r["samples"]:
            reasons[s["finish_reason"]] = reasons.get(s["finish_reason"], 0) + 1
    print(f"[search] n={len(records)}  EM pass rate={sum(rates) / max(len(rates), 1):.3f}  "
          f"mean calls={sum(calls) / max(len(calls), 1):.2f}  finish={reasons}")


if __name__ == "__main__":
    fire.Fire(main)
