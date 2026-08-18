# =============================================================================
# pixelproof.archive — WHAT THIS PACKAGE IS
# -----------------------------------------------------------------------------
# Retired modules, frozen 2026-08-18. Each one produced a finished, written-up
# result (see ml/EXPERIMENTS.md) and nothing in the live path imports it:
#
#   analyze.py           E3  — k-means / t-SNE embedding analysis
#   classical.py         E2  — classical classifiers on frozen CNN embeddings
#   embeddings.py        E2  — embedding extraction for the above
#   learning_curve.py    E4  — the data-size ablation
#   ela.py               E18 — Error Level Analysis baseline (splice line closed)
#   backbone_features.py E16 — frozen DINOv2 feature extraction
#
# They are kept importable (as pixelproof.archive.<name>) so the archived
# experiment scripts remain runnable in principle, but they receive no
# maintenance.
# =============================================================================
