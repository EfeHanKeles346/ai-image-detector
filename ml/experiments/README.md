# Experiment scripts

One script per numbered experiment in [`../EXPERIMENTS.md`](../EXPERIMENTS.md). These are the
code that produced the numbers quoted in `EXPERIMENTS.md` and `ROADMAP.md` — kept in the repo so
every claim in the report is reproducible rather than remembered.

Run from `ml/` with `PYTHONPATH=src`:

```bash
PYTHONPATH=src .venv/bin/python experiments/e11_tile_grid_search.py
```

| Script | Experiment | Produces |
|---|---|---|
| `e07_patch_vs_resize.py` | E7 control | Native-resolution patches vs downscaling, per generator. Showed the mechanism was real *and* that patch inference cannot be bolted onto a resize-trained CNN (96% false positives on real photos). |
| `e08_cnn_vs_features.py` | E8 | ResNet-18 vs the feature model on identical training data and test sets — the controlled method-vs-method comparison. |
| `e09_ensemble.py` | E9 | Eight blending rules (mean, weighted, max, min, and rank-normalised variants). The negative result that decided the demo shows methods separately. |
| `e10_archive1_format.py` | E10 | Format control: re-encode the AI half to JPEG, then both halves, and re-measure. Tests whether the JPEG/PNG split was doing the work. |
| `e10_archive1_aspect.py` | E10 | Aspect control: centre-crop both classes square at native resolution. Also reports the metadata-only ceiling (AUC 1.000 from width/height alone). |
| `e11_tile_aggregation.py` | E11 | Six aggregation rules over a tile grid, plus the fraction of tiles below the texture floor. |
| `e11_tile_grid_search.py` | E11 | Grid-size sweep (2×2 … 6×6) × aggregation rule, optimised on high-resolution generators. Produced the 6×6 + top-3 choice and the ~700px crossover. |

Feature extraction caches to `../artifacts/features/`, so re-running any of these after the first
time costs seconds rather than minutes.

**Requires the datasets in `ROADMAP.md` §1 to be present at the paths listed there.** Scripts fail
fast with a clear error if a dataset folder is missing.
