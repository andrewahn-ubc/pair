#!/bin/bash
#SBATCH --job-name=pair_eval
#SBATCH --account=rrg-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=logs/pair_eval_%j.out

module purge
module load StdEnv/2023 python/3.11 cuda

source $SCRATCH/venv/pair/bin/activate

cd $SCRATCH/pair

mkdir -p logs results

python - <<'PY'
import time
print("\n start time: " + str(int(time.time())))
PY

python3 -u main.py \
    --attack-model "llama-2-7b-chat-hf" \
    --target-model "llama-2-7b-chat-hf" \
    --judge-model "llama-guard-local" \
    --evaluate-locally \
    --local-llama-path "/home/taegyoem/scratch/llama2_7b_chat_hf" \
    --local-llama-guard-path "/home/taegyoem/scratch/llama_guard_7b" \
    --local-attacker-path "/home/taegyoem/scratch/llama2_7b_chat_hf" \
    --input-path "/home/taegyoem/scratch/dp-llm-experiments/official_data/test_00.csv" \
    --output-path "/home/taegyoem/scratch/pair/results/test_00_pair_output.csv" \
    --n-streams "30" \
    --n-iterations "3" 

python - <<'PY'
import time
print("\n end time: " + str(int(time.time())))
PY