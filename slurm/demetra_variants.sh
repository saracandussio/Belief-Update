#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="bu-variants"
#SBATCH --partition=lovelace
#SBATCH --gres=gpu:a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_outputs/bu-variants.%j.out
#SBATCH --export=ALL

# Stages `data` + `variants` for every dataset (model-independent, run once).
# Serves the writer model locally with vLLM and points the OpenAI client at it.
# External API instead: export OPENAI_BASE_URL / OPENAI_API_KEY / WRITER_MODEL
# and SERVE_WRITER=0 before sbatch.
# Submit:  sbatch slurm/demetra_variants.sh

set -euo pipefail

PROJECT_PATH="/u/scandussio/Belief-Update"
cd "$PROJECT_PATH"
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false

DATASETS="${DATASETS:-2wiki,musique,hotpotqa,bamboogle}"
MAX_QUESTIONS="${MAX_QUESTIONS:-500}"
WRITER_MODEL="${WRITER_MODEL:-openai/gpt-oss-120b}"
SERVE_WRITER="${SERVE_WRITER:-1}"
PORT="${PORT:-8011}"

echo "Project: $PROJECT_PATH  Datasets: $DATASETS  Writer: $WRITER_MODEL"

if [[ "$SERVE_WRITER" == "1" ]]; then
    vllm serve "$WRITER_MODEL" --port "$PORT" --max-model-len 16384 \
        --gpu-memory-utilization 0.9 > "slurm_outputs/bu-variants-server.${SLURM_JOB_ID}.log" 2>&1 &
    SERVER_PID=$!
    trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
    echo "Waiting for vLLM server on :$PORT ..."
    for _ in $(seq 1 180); do
        curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null && break
        kill -0 $SERVER_PID 2>/dev/null || { echo "server died, see server log"; exit 1; }
        sleep 10
    done
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null || { echo "server not ready"; exit 1; }
    export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
    export OPENAI_API_KEY="EMPTY"
fi

STAGES=data,variants DATASETS="$DATASETS" MAX_QUESTIONS="$MAX_QUESTIONS" \
WRITER_MODEL="$WRITER_MODEL" VLLM_PYTHON=python API_PYTHON=python \
    ./scripts/run_pipeline.sh

echo "=== Done ==="
