import csv, subprocess, os, shlex
import time

csv_path = "/home/taegyoem/scratch/dp-llm-experiments/official_data/test_00.csv"
rows = []
with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

for i, row in enumerate(rows):
    print(f"starting iteration {i} at {str(time.time())}")
    goal = row["goal"]
    target = row["target"]

    cmd = [
        "python3", "main.py",
        "--attack-model", "llama-2-7b-chat-hf",
        "--target-model", "llama-2-7b-chat-hf",
        "--judge-model", "llama-guard-local",
        "--evaluate-locally",
        "--local-llama-path", "/home/taegyoem/scratch/llama2_7b_chat_hf",
        "--local-llama-guard-path", "/home/taegyoem/scratch/llama_guard_3_1b",
        "--goal", goal,
        "--target-str", target,
        "--n-streams", "2",
        "--n-iterations", "2",
    ]

    print(f"\n=== Running row {i} ===")
    print(" ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)