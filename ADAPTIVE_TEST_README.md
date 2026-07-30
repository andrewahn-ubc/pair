# Adaptive test subset (GCG / AutoDAN / PAIR)

Created from `official/splits/test/` without modifying those originals.

## adaptive_test/ (50 chunks × 5 prompts = 250 goals)
- advbench: 100
- harmbench: 100
- jailbreakbench: 50

Chunk order: advbench (`test_00`–`test_19`), harmbench (`test_20`–`test_39`), jailbreakbench (`test_40`–`test_49`).

Copy to Narval, e.g.:
```bash
rsync -av official/splits/adaptive_test/ \
  $SCRATCH/dp-llm-experiments/official_data/adaptive_test/
```

## adaptive_test_remainder/ (165 chunks, 825 goals)
Everything not in the adaptive subset, also in 5-prompt chunks.
