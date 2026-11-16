#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="bu-search"
#SBATCH --partition=lovelace
#SBATCH --gres=gpu:a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=16
#SBATCH --array=0-7
#SBATCH --output=slurm_outputs/bu-search-%A_%a.out
#SBATCH --export=ALL

# Search stage + analysis, same (model, dataset) array as demetra_closed.sh.
# Starts the Search-R1 retrieval server in the job, FAISS on CPU (flat E5
# index ~65GB, does not fit on the GPU next to the model), hence --mem=160G.
#
# One-time setup (login node):
#   git clone https://github.com/PeterGriffinJin/Search-R1.git /u/scandussio/Search-R1
#   conda create -n retriever python=3.10 -y && conda activate retriever
#   pip install torch transformers datasets faiss-cpu fastapi uvicorn huggingface_hub
#   python /u/scandussio/Search-R1/scripts/download.py --save_path /u/scandussio/search-r1-index
#   cd /u/scandussio/search-r1-index && cat part_* > e5_Flat.index && gzip -d wiki-18.jsonl.gz
# Submit:  sbatch slurm/demetra_search.sh      (single task: --array=0)

set -euo pipefail

PROJECT_PATH="/u/scandussio/Belief-Update"
RETRIEVER_REPO="${RETRIEVER_REPO:-/u/scandussio/Search-R1}"
INDEX_DIR="${INDEX_DIR:-/u/scandussio/search-r1-index}"
RET_PORT="${RET_PORT:-8000}"

MODELS=("Qwen/Qwen3.8-27B" "openai/gpt-oss-20b")
DATASETS=("2wiki" "musique" "hotpotqa" "bamboogle")
N_DATASETS=${#DATASETS[@]}
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL="${MODELS[$((TASK_ID / N_DATASETS))]}"
DATASET="${DATASETS[$((TASK_ID % N_DATASETS))]}"
NUM_OUT="${NUM_OUT:-8}"
CHAT_TEMPLATE_KWARGS="${CHAT_TEMPLATE_KWARGS:-}"

echo "Task $TASK_ID  Model: $MODEL  Dataset: $DATASET"

(
    cd "$RETRIEVER_REPO"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${RETRIEVER_ENV:-retriever}"
    CUDA_VISIBLE_DEVICES="" python search_r1/search/retrieval_server.py \
        --index_path "$INDEX_DIR/e5_Flat.index" \
        --corpus_path "$INDEX_DIR/wiki-18.jsonl" \
        --topk 3 --retriever_name e5 --retriever_model intfloat/e5-base-v2
) > "slurm_outputs/bu-retriever.${SLURM_JOB_ID:-local}.log" 2>&1 &
RET_PID=$!
trap 'kill $RET_PID 2>/dev/null || true' EXIT

cd "$PROJECT_PATH"
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"

echo "Waiting for retriever on :$RET_PORT (loading the index takes a while) ..."
for _ in $(seq 1 360); do
    curl -sf -X POST "http://127.0.0.1:${RET_PORT}/retrieve" -H 'Content-Type: application/json' \
        -d '{"queries": ["test"], "topk": 1}' >/dev/null && break
    kill -0 $RET_PID 2>/dev/null || { echo "retriever died, see retriever log"; exit 1; }
    sleep 10
done

STAGES=search,analyze MODELS="$MODEL" DATASETS="$DATASET" NUM_OUT="$NUM_OUT" \
RETRIEVER_URL="http://127.0.0.1:${RET_PORT}/retrieve" \
CHAT_TEMPLATE_KWARGS="$CHAT_TEMPLATE_KWARGS" VLLM_PYTHON=python API_PYTHON=python \
    ./scripts/run_pipeline.sh

echo "=== Done ==="
