#!/usr/bin/env bash
# First tests: how much of each multi-hop item is already in the parameters.
#
# Stages (comma-separated in STAGES):
#   data      download + freeze the item subset           -> data/<ds>/items.json
#   variants  hop chains + paraphrases (stronger model)   -> data/<ds>/variants.json
#   closed    closed-book, composed question (CoT/no-CoT)
#   hops      closed-book, single hops (CoT/no-CoT)
#   search    search agent with clean retrieval
#   judge     LLM judge on non-EM answers (optional)
#   analyze   item table + summary                         -> outputs/<model>/<ds>/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VLLM_PYTHON="${VLLM_PYTHON:-python}"
API_PYTHON="${API_PYTHON:-python}"   # CPU-only stages (data, analyze)
# Subject models, comma-separated. MODEL is kept as an alias for a single model.
MODELS="${MODELS:-${MODEL:-Qwen/Qwen3.8-27B,openai/gpt-oss-20b}}"
CHAT_TEMPLATE_KWARGS="${CHAT_TEMPLATE_KWARGS:-}"   # JSON, e.g. '{"reasoning_effort": "medium"}'
LLM_KWARGS="${LLM_KWARGS:-}"                       # JSON passed to vllm.LLM
DATASETS="${DATASETS:-2wiki,musique,hotpotqa,bamboogle}"
STAGES="${STAGES:-data,variants,closed,hops,search,analyze}"
MAX_QUESTIONS="${MAX_QUESTIONS:-500}"
SEED="${SEED:-0}"
NUM_OUT="${NUM_OUT:-8}"
NUM_PARAPHRASES="${NUM_PARAPHRASES:-3}"
WRITER_MODEL="${WRITER_MODEL:-openai/gpt-oss-120b}"  # stronger model for variants (offline vLLM)
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-oss-120b}"    # judge (offline vLLM)
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
TOOL_FORMAT="${TOOL_FORMAT:-}"       # qwen_native | searchr1 (default: from model name)
TP="${TP:-1}"
METRIC="${METRIC:-em}"               # em | judge

contains_stage() { [[ ",${STAGES}," == *",$1,"* ]]; }

tool_args=()
[[ -n "$TOOL_FORMAT" ]] && tool_args=(--tool_format "$TOOL_FORMAT")
gen_args=()
[[ -n "$CHAT_TEMPLATE_KWARGS" ]] && gen_args+=(--chat_template_kwargs "$CHAT_TEMPLATE_KWARGS")
[[ -n "$LLM_KWARGS" ]] && gen_args+=(--llm_kwargs "$LLM_KWARGS")
IFS=',' read -r -a model_list <<< "$MODELS"

IFS=',' read -r -a dataset_list <<< "$DATASETS"

for dataset in "${dataset_list[@]}"; do
    if contains_stage data; then
        "$API_PYTHON" scripts/prepare_data.py --data_name "$dataset" \
            --max_questions "$MAX_QUESTIONS" --seed "$SEED"
    fi
    if contains_stage variants; then
        "$VLLM_PYTHON" scripts/make_variants.py --data_name "$dataset" \
            --writer_model "$WRITER_MODEL" --num_paraphrases "$NUM_PARAPHRASES"
    fi
    for MODEL in "${model_list[@]}"; do
        echo "=== ${MODEL} on ${dataset} ==="
        if contains_stage closed; then
            "$VLLM_PYTHON" scripts/generate_closed_book.py --model_name "$MODEL" \
                --data_name "$dataset" --target question --num_out "$NUM_OUT" \
                --tensor_parallel_size "$TP" "${gen_args[@]}"
        fi
        if contains_stage hops; then
            "$VLLM_PYTHON" scripts/generate_closed_book.py --model_name "$MODEL" \
                --data_name "$dataset" --target hops --num_out "$NUM_OUT" \
                --tensor_parallel_size "$TP" "${gen_args[@]}"
        fi
        if contains_stage search; then
            "$VLLM_PYTHON" scripts/generate_search_traces.py --model_name "$MODEL" \
                --data_name "$dataset" --retriever_url "$RETRIEVER_URL" --num_out "$NUM_OUT" \
                --tensor_parallel_size "$TP" "${tool_args[@]}" "${gen_args[@]}"
        fi
        if contains_stage judge; then
            "$VLLM_PYTHON" scripts/judge_answers.py --model_name "$MODEL" --data_name "$dataset" \
                --judge_model "$JUDGE_MODEL"
        fi
        if contains_stage analyze; then
            judge_args=()
            [[ "$METRIC" == "judge" ]] && judge_args=(--judge_model "$JUDGE_MODEL")
            "$API_PYTHON" scripts/analyze.py --model_name "$MODEL" --data_name "$dataset" \
                --metric "$METRIC" "${judge_args[@]}"
        fi
    done
done
