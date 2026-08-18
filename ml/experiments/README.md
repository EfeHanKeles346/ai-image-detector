# Experiment scripts

Two kinds of script live here, and the split is the point:

- **This folder** holds the scripts that are still *run*: the current evaluation protocol
  and anything the plan (`../../PLAN.md`) calls for next.
- **[`archive/`](archive/README.md)** holds the scripts that produced finished, written-up
  results (E7–E18). They are frozen evidence for `../EXPERIMENTS.md` and are not maintained.

Run from `ml/` with `PYTHONPATH=src`:

```bash
PYTHONPATH=src .venv/bin/python experiments/e20_tile_model_shootout.py --help
```

| Script | Experiment | Purpose |
|---|---|---|
| `e20_tile_model_shootout.py` | E20 / E20-v2 | Three model families on identical native tiles, under the hardened protocol: per-image score persistence, disjoint calibration/evaluation halves, aggregation controls, source-transfer FP reporting, three-seed default. |
| `e21_external_detector_benchmark.py` | E21 | Runs an official external detector through the same protocol — the go/no-go gate before paying for any further training of our own. Two arms: `--detector bfree` (GRIP-UNINA checkout, non-commercial licence acknowledgement required) and `--detector community-forensics` (ViT-S, MIT; local snapshot or `--allow-download`, ~83 MB). |

Feature extraction caches to `../artifacts/features/`, so re-running anything after the
first time costs seconds rather than minutes.

**Requires the datasets in `../../DATASETS.md` to be present at the paths listed there.**
Scripts fail fast with a clear error if a dataset folder is missing.
