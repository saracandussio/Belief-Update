#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="bu-closed"
#SBATCH --partition=lovelace
#SBATCH --gres=gpu:a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-7
#SBATCH --output=slurm_outputs/bu-closed-%A_%a.out
#SBATCH --export=ALL

# Closed-book stages (question + hops, original + paraphrases) and analysis.
# One array task per (model, dataset): task = model_idx * 4 + dataset_idx
#   0-3: Qwen3.8-27B   on 2wiki, musique, hotpotqa, bamboogle
#   4-7: gpt-oss-20b   on the same
# Needs data/<dataset>/items.json + variants.json (demetra_variants.sh).
# Resumable: resubmitting a task skips what is already done.
# Submit:  sbatch --dependency=afterok:<variants_job_id> slurm/demetra_closed.sh
# One task: sbatch --array=4 slurm/demetra_closed.sh

set -euo pipefail
source /u/scandussio/Belief-Update/slurm/demetra_env.sh

PROJECT_PATH="/u/scandussio/Belief-Update"
cd "$PROJECT_PATH"
source .venv/bin/activate

MODELS=("Qwen/Qwen3.8-27B" "openai/gpt-oss-20b")
DATASETS=("2wiki" "musique" "hotpotqa" "bamboogle")
N_DATASETS=${#DATASETS[@]}
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL="${MODELS[$((TASK_ID / N_DATASETS))]}"
DATASET="${DATASETS[$((TASK_ID % N_DATASETS))]}"

STAGES="${STAGES:-closed,hops,analyze}"
NUM_OUT="${NUM_OUT:-8}"
CHAT_TEMPLATE_KWARGS="${CHAT_TEMPLATE_KWARGS:-}"   # e.g. '{"reasoning_effort": "medium"}'

echo "Task $TASK_ID  Model: $MODEL  Dataset: $DATASET  Stages: $STAGES"
nvidia-smi --query-gpu=name,memory.total --format=csv

STAGES="$STAGES" MODELS="$MODEL" DATASETS="$DATASET" NUM_OUT="$NUM_OUT" \
CHAT_TEMPLATE_KWARGS="$CHAT_TEMPLATE_KWARGS" VLLM_PYTHON=python API_PYTHON=python \
    ./scripts/run_pipeline.sh

echo "=== Done ==="
