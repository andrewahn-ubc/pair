#!/bin/bash
#SBATCH --job-name=pair_eval
#SBATCH --account=rrg-mijungp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
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

python -u main.py \
  --attack-model wizard-vicuna-13b-uncensored \
  --target-model llama-2-7b-chat-hf \
  --judge-model llama-guard-local \
  --evaluate-locally \
  --not-jailbreakbench \
  --local-attacker-path /home/taegyoem/scratch/wizard_vicuna_13b \
  --local-llama-path /home/taegyoem/scratch/llama2_7b \
  --local-llama-guard-path /home/taegyoem/scratch/llama_guard \
  --input-path "/home/taegyoem/scratch/official_data/test_00.csv" \
  --output-path "/home/taegyoem/scratch/pair/results/test_00_pair_output.csv" \
  --n-streams "30" \
  --n-iterations "3" 
  
python - <<'PY'
import time
print("\n end time: " + str(int(time.time())))
PY