# Marketing benchmarks (private)

This folder contains benchmark profiles intended for **internal** marketing demos (sales decks, PDF summaries).

## Key rule: don’t publish results

These scripts default to writing JSON artifacts into `/tmp` so results don’t accidentally end up in `invert-package/results/` (which is used by the documentation leaderboard build).

You can override the output directory via:

- `--out-dir /path/to/private/output`
- or `INVERT_MARKETING_BENCH_DIR=/path/to/private/output`

## Run

Use the project environment so imports resolve to this repo:

```bash
uv run --project invert-package python invert-package/scripts/marketing/run_marketing_benchmarks.py \
  --profile low-channel-neurofeedback
```

Profiles:

- `low-channel-neurofeedback`: 8-channel EEG (neurofeedback/BCI scenarios)
- `epilepsy-software`: 64-channel EEG, focal + noisy scenarios, curated solver mix
- `platform-leaderboard`: broader solver-family run (method-selection story)

