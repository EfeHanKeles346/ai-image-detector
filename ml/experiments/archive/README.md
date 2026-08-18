# Archived experiment scripts — frozen evidence

**Frozen 2026-08-18.** Each script here produced a finished, written-up result in
[`../../EXPERIMENTS.md`](../../EXPERIMENTS.md) and will not be run again. They are kept so
every number in the log and the internship report stays reproducible in principle, but they
are **not maintained**: they were written against the repo as it stood on their date, and
later refactors (e.g. modules retired to `pixelproof/archive/`) are reflected only in their
imports, nothing more.

| Script | Experiment | What it established |
|---|---|---|
| `e07_patch_vs_resize.py` | E7 control | Downscaling deletes the evidence; patch inference cannot be bolted onto a resize-trained CNN (96% false positives). |
| `e08_cnn_vs_features.py` | E8 | 68 hand-crafted statistics vs the ResNet on identical training data — the specialist result. |
| `e09_ensemble.py` | E9 | Eight blending rules; best beat the ResNet by +0.002 — the negative result behind the unblended demo. |
| `e10_archive1_format.py` / `e10_archive1_aspect.py` | E10 | archive1 is maximally confounded (metadata AUC 1.000) but the resize pipeline made the CNNs immune. |
| `e11_tile_aggregation.py` / `e11_tile_grid_search.py` | E11 | 6×6 grid + top-3 mean; SDXL 0.948; the ~700px crossover. |
| `e12_statistics_v1_vs_v2.py` | E12 | Ten times the data did not help where it mattered; the compression domain gap. |
| `e13_tile_false_positives.py` | E13 | The tile model calls 79% of real photographs "AI" — AUC is not an operating point. |
| `e14_real_class_diversity.py` | E14 | The narrow-real-class finding — the project's most important measurement. |
| `e15_balanced_multisource.py` | E15 | The balanced multi-source real half: fixes unseen-real-source FP, does not move Defactify. |
| `e16_backbone_probe.py` | E16 | Frozen DINOv2 whole-image probe. Its falsification was overturned by the label fix (E19c). |
| `e17_module2_first_measurement.py` | E17 | Tile localisation works on diffusion inpainting only; nine classic sets sit at chance. |
| `e18_ela_vs_tiles.py` | E18 | ELA fails on the PNG-flattened compilation and passes its hand-made positive control — the splice line closed. |
