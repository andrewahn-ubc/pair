#!/bin/bash
#SBATCH --job-name=pair-test
#SBATCH --account=def-mijungp
#SBATCH --array=0-99
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --output=logs/pair_test_%A_%a.out

module purge
module load StdEnv/2023 python/3.11 cuda

source $SCRATCH/venv/pair/bin/activate

cd $SCRATCH/pair

mkdir -p logs results

python - <<'PY'
import time
print("\n start time: " + str(int(time.time())))
PY

IDX=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
INPUT_PATH="/home/taegyoem/links/scratch/official_data/test_${IDX}.csv"
OUTPUT_PATH="/home/taegyoem/links/scratch/pair/results/test_pair_llama3_output_${IDX}.csv"

echo "Running on file: $INPUT_PATH"

python -u main.py \
  --attack-model vicuna-13b-v1.5 \
  --target-model llama-3-8b-instruct \
  --judge-model llama-guard-local \
  --evaluate-locally \
  --not-jailbreakbench \
  --local-attacker-path /scratch/taegyoem/vicuna_13b \
  --local-llama-path /home/taegyoem/scratch/llama_3_8b_instruct \
  --local-llama-guard-path /scratch/taegyoem/llama_guard \
  --input-path "$INPUT_PATH" \
  --output-path "$OUTPUT_PATH" \
  --n-streams "30" \
  --n-iterations "3" 
  
python - <<'PY'
import time
print("\n end time: " + str(int(time.time())))
PY