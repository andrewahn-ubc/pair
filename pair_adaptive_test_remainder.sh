#!/bin/bash
#SBATCH --job-name=pair-remainder
#SBATCH --account=def-mijungp
#SBATCH --array=0-164
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --output=logs/pair_remainder_%A_%a.out

# PAIR on adaptive_test_remainder/ (165 chunks × 5 goals) against the merged Llama-2 target:
#   $SCRATCH/merged_adaptive_l2_lam0.1_eps-0.5_ep5

module purge
module load StdEnv/2023 python/3.11 cuda

source $SCRATCH/venv/pair/bin/activate

PAIR_DIR="${SCRATCH}/pair"
cd "$PAIR_DIR" || { echo "ERROR: cannot cd to $PAIR_DIR"; exit 1; }
export PYTHONPATH="${PAIR_DIR}${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p logs results

# Fail fast if the PAIR source tree is incomplete (common cause of ModuleNotFoundError).
for f in main.py loggers.py conversers.py judges.py local_llm.py config.py common.py; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $PAIR_DIR/$f — sync the full JailbreakingLLMs repo to \$SCRATCH/pair"
    ls -la "$PAIR_DIR"
    exit 1
  fi
done

echo "cwd=$(pwd)"
echo "python=$(which python)"
echo "PYTHONPATH=$PYTHONPATH"

python - <<'PY'
import time
print("\n start time: " + str(int(time.time())))
PY

IDX=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")
INPUT_PATH="${SCRATCH}/official_data/adaptive_test_remainder/test_${IDX}.csv"
OUTPUT_PATH="${SCRATCH}/pair/results/adaptive_test_remainder_pair_merged_l2_lam0.1_eps-0.5_${IDX}.csv"
TARGET_PATH="${SCRATCH}/merged_adaptive_l2_lam0.1_eps-0.5_ep5"

echo "Running on file: $INPUT_PATH"
echo "Target model path: $TARGET_PATH"
echo "Output: $OUTPUT_PATH"

python -u "$PAIR_DIR/main.py" \
  --attack-model vicuna-13b-v1.5 \
  --target-model llama-2-7b-chat-hf \
  --judge-model llama-guard-local \
  --evaluate-locally \
  --not-jailbreakbench \
  --local-attacker-path /scratch/taegyoem/vicuna_13b \
  --local-llama-path "$TARGET_PATH" \
  --local-llama-guard-path /scratch/taegyoem/llama_guard \
  --input-path "$INPUT_PATH" \
  --output-path "$OUTPUT_PATH" \
  --n-streams "30" \
  --n-iterations "3"

python - <<'PY'
import time
print("\n end time: " + str(int(time.time())))
PY
