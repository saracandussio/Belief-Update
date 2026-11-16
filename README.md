# Belief update in search agents: parametric baselines

First tests for the belief-update project: before injecting anything, measure
how much of each multi-hop item the model already knows, and whether that
knowledge depends on the question format.

Codebase layout and conventions follow
[Commitment-Boundary](https://github.com/DanielSc4/Commitment-Boundary):
fire CLIs in `scripts/`, shared logic in a package, a `run_pipeline.sh` driver
with `STAGES`, resumable batched JSON outputs under `data/`, analysis under
`outputs/`.

**Deliberately out of scope for now:** answer suffixes / forced elicitation and
logit-based measures. Every answer is free generation with a fixed answer
format given in the instruction (`<answer> ... </answer>`, as in Search-R1).

## Models

Default subjects (`MODELS` in `run_pipeline.sh`):

| Model | Tool calling | Thinking | Notes |
|---|---|---|---|
| `Qwen/Qwen3.8-27B` | XML in `<tool_call>` (`qwen_xml`) | on; `enable_thinking=False` for the no-CoT pass | text-only: vision inputs disabled in vLLM (`limit_mm_per_prompt` = 0) |
| `openai/gpt-oss-20b` | harmony, `to=functions.search ... <\|call\|>` (`harmony`) | always on, no switch: **no no-CoT pass** | `reasoning_effort` via `CHAT_TEMPLATE_KWARGS` |

Also supported: Qwen3 (`qwen_native`, JSON tool calls) and Search-R1 / ZeroSearch
checkpoints (`searchr1`). The format is picked from the model name; override with
`TOOL_FORMAT`.

Sampling defaults are the Commitment-Boundary ones (temperature 0.7, top-p 0.9)
for every model, so that runs are comparable. The model cards recommend other
settings (Qwen3.8 thinking: temperature 1.0, top-p 0.95, top-k 20): pass them on
the script CLI if you prefer them.

## What it measures

On the Search-R1 suite (FlashRAG versions of 2Wiki, MuSiQue, HotpotQA, Bamboogle):

| Stage | Question | Output |
|---|---|---|
| `closed` | Does the model answer the composed question without tools? (CoT and no-CoT) | pass rate over `NUM_OUT` samples |
| `hops` | Does it know each single hop? | pass rate per hop |
| `variants` | Hop chains + paraphrases written by a stronger model | `variants.json` |
| `closed` / `hops` on paraphrases | Is it overfitting the dataset's wording? | paraphrase pass rates |
| `search` | Same questions with a search tool, clean retrieval | traces + pass rate |
| `analyze` | Per-item categories | item table + summary |

Per-item categories (closed-book, CoT, pass-rate threshold `tau`):

- `composes`: every hop known, composed question known;
- `shortcut`: composed question known, some hop not (memorized pair, skips the hops);
- `composition_gap`: every hop known, composed question not;
- `unknown`: neither.

`prior_defined_ids.txt` lists chain items whose every hop is known: the input
set for the prefill / counterfactual stage, where a prior must exist.
`format_sensitive` flags items whose pass rate moves by at least `delta`
between original and paraphrased question.

## Installation

```bash
python3 -m venv .venv-vllm && source .venv-vllm/bin/activate
pip install -e ".[vllm,api,dev]"
```

**Retriever (search stage only).** Use the Search-R1 server: E5 over Wikipedia 2018,
top-3 passages. From a Search-R1 checkout, download the index and corpus
(`scripts/download.py`) and run `retrieval_launch.sh`. It listens on
`http://127.0.0.1:8000/retrieve`.

**Stronger model and judge.** Any OpenAI-compatible endpoint (e.g. `vllm serve`
on a larger model): set `OPENAI_BASE_URL` and `OPENAI_API_KEY`.

## Quick start

```bash
# 1. Inspect how each model's chat template declares the tool and wraps results
python scripts/check_template.py --model_name Qwen/Qwen3.8-27B
python scripts/check_template.py --model_name openai/gpt-oss-20b

# 2. Run the pipeline (both default models)
WRITER_MODEL=<stronger-model> \
CHAT_TEMPLATE_KWARGS='{"reasoning_effort": "medium"}' \
DATASETS=2wiki,musique \
MAX_QUESTIONS=500 \
./scripts/run_pipeline.sh
```

Smoke run:

```bash
DATASETS=bamboogle MAX_QUESTIONS=20 NUM_OUT=2 STAGES=data,variants,closed,hops,analyze \
WRITER_MODEL=<stronger-model> ./scripts/run_pipeline.sh
```

Optional judge (aliases and granularity that EM misses), then analysis on judge labels:

```bash
STAGES=judge,analyze METRIC=judge JUDGE_MODEL=<judge-model> MODEL=<subject-model> \
./scripts/run_pipeline.sh
```

Defaults: temperature `0.7`, top-p `0.9` (as in Commitment-Boundary), 8 samples
per unit, 8192 max tokens closed-book (1024 no-CoT), up to 5 search calls of
4096 tokens each. Override any sampling parameter on the script CLI, e.g.
`--temperature 0.6 --top_k 20`.

## Running individual stages

```bash
python scripts/prepare_data.py --data_name 2wiki --max_questions 500 --seed 0
python scripts/make_variants.py --data_name 2wiki --writer_model <stronger-model>
python scripts/generate_closed_book.py --model_name <m> --data_name 2wiki --target question
python scripts/generate_closed_book.py --model_name <m> --data_name 2wiki --target hops
python scripts/generate_search_traces.py --model_name <m> --data_name 2wiki
python scripts/judge_answers.py --model_name <m> --data_name 2wiki --judge_model <j>
python scripts/analyze.py --model_name <m> --data_name 2wiki --metric em
```

## Tool formats

- `qwen_xml` (Qwen3.5 / 3.6 / 3.8): `<tool_call><function=search><parameter=query>...`.
- `harmony` (gpt-oss): `<|channel|>commentary to=functions.search <|constrain|>json<|message|>{"query": ...}<|call|>`.
- `qwen_native` (Qwen3): `<tool_call>{"name": "search", "arguments": {...}}</tool_call>`.
- `searchr1` (default for Search-R1 / ZeroSearch checkpoints): the verbatim
  Search-R1 prompt, with `<search>` / `<information>` tags.

For the native formats the search tool is declared through the chat template,
and the tool-response wrapper is **read off the same template at start-up**
(`modeling.derive_response_template`). It is stored with every record; the
built-in strings in `utils.TOOL_FORMATS` are only a fallback.

The loop works on token ids: the model's generated ids are kept verbatim and
only tool responses are tokenized. Every search sample stores its full
`token_ids` and its `segments` (model turns, queries, retrieved docs), so a
trajectory can later be replayed with a prefill or a perturbed tool output at
any step.

## Repository structure

```text
belief_update/
  utils.py        datasets, thinking markers, tool formats, prompts, extraction
  agent.py        backend-agnostic multi-turn search loop
  modeling.py     vLLM / tokenizer helpers
  retrieval.py    Search-R1 retrieval-server client
  variants.py     prompts + validation for hop chains and paraphrases
  units.py        items -> generation units (question / hop x original / paraphrase)
  judge.py        LLM judge prompt and parsing
  analysis.py     item table, categories, summary
  api.py          OpenAI-compatible client with resumable cache
  shared/         metrics (EM, F1, cover-EM), batched resumable IO
scripts/          stage entry points + run_pipeline.sh + check_template.py
tests/            unit tests (no GPU needed)
docs/PIPELINE.md  what each stage does and the known caveats
```

## Outputs

```text
data/<dataset>/items.json                                  frozen item subset
data/<dataset>/variants.json                               hop chains + paraphrases
data/<dataset>/<model>_closed_book_<question|hops>_<cot|nocot>.json
data/<dataset>/<model>_search_<tool_format>.json
data/<dataset>/*_readable.jsonl                            compact views
outputs/<model>/<dataset>/item_table.jsonl
outputs/<model>/<dataset>/summary_<metric>.json
outputs/<model>/<dataset>/prior_defined_ids.txt
```

## Tests

```bash
pytest
```
