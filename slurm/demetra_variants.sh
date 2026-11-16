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
# The writer model runs in-process with offline vLLM (no server).
# Submit:  sbatch slurm/demetra_variants.sh

set -euo pipefail
source /u/scandussio/Belief-Update/slurm/demetra_env.sh

PROJECT_PATH="/u/scandussio/Belief-Update"
cd "$PROJECT_PATH"
source .venv/bin/activate

DATASETS="${DATASETS:-2wiki,musique,hotpotqa,bamboogle}"
MAX_QUESTIONS="${MAX_QUESTIONS:-500}"
WRITER_MODEL="${WRITER_MODEL:-openai/gpt-oss-120b}"

echo "Project: $PROJECT_PATH  Datasets: $DATASETS  Writer: $WRITER_MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv

STAGES=data,variants DATASETS="$DATASETS" MAX_QUESTIONS="$MAX_QUESTIONS" \
WRITER_MODEL="$WRITER_MODEL" VLLM_PYTHON=python API_PYTHON=python \
    ./scripts/run_pipeline.sh

echo "=== Done ==="
