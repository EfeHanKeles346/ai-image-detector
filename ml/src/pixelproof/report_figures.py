# =============================================================================
# report_figures.py — WHAT THIS FILE DOES
# -----------------------------------------------------------------------------
# Turns the evaluator's own output into the figures used to report results.
# Every number is read from artifacts/e20/results.json and the per-image score
# JSONL that produced it — nothing is retyped, so a figure cannot drift away
# from the measurement it claims to show.
#
# WHY THIS EXISTS
# -----------------------------------------------------------------------------
# A chart typed into a slide deck is a second, unversioned copy of a result. It
# stays correct only until the experiment is re-run, and nothing warns you when
# it stops being correct. Generating figures from the saved artefacts makes the
# deck a view of the data rather than a claim about it: re-run the evaluator,
# re-run this, and every figure is current.
#
# WHAT IT DRAWS
# -----------------------------------------------------------------------------
# arms.png       AI recall at the false-positive budget, one bar per model arm
# roc.png        ROC for the selected arm, with the chosen operating point marked
# scores.png     Score distributions of both classes plus the threshold
# sources.png    False positives per authentic-photograph source, same threshold
# generators.png Recall per generator for the selected arm
#
# Usage:  PYTHONPATH=src python -m pixelproof.report_figures
# =============================================================================

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pixelproof.evaluation_protocol import (aggregate_tile_scores,
                                            stable_calibration_split,
                                            threshold_at_fpr)

NAVY = "#1E2761"
ALERT = "#C1272D"
OK = "#2E7D5B"
WARN = "#D98324"
GREY = "#5A6270"
MUTED = "#9AA8C4"

ARM_LABEL = {"stats": "68 İstatistik\n(el yapımı)",
             "resnet18": "ResNet-18\n(ön-eğitimli)",
             "small_cnn": "SmallCNN\n(sıfırdan)"}


def _frame(ax, grid_axis: str | None = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#D5DAE4")
    ax.tick_params(colors=GREY, labelsize=8.5)
    if grid_axis:
        ax.grid(axis=grid_axis, alpha=0.25, lw=0.7)


def load_scores(path: Path, rule: str) -> list[dict]:
    """One aggregated image score per record, plus the fields we group by."""
    out = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        out.append({"path": record["path"], "dataset": record["dataset"],
                    "source": record["source"], "label": record["label"],
                    "score": aggregate_tile_scores(record["tile_scores"], rule),
                    "tiles": record["tile_count"]})
    return out


def split_defactify(records: list[dict], fraction: float, seed: int):
    """Same calibration/evaluation split the evaluator used, per source and class."""
    groups = defaultdict(list)
    for r in records:
        if r["dataset"] == "defactify":
            groups[(r["source"], r["label"])].append(r)
    calibration, evaluation = [], []
    for group in groups.values():
        cal, ev = stable_calibration_split(group, fraction, seed)
        calibration += cal
        evaluation += ev
    return calibration, evaluation


def figure_arms(results: dict, out: Path, budget: float) -> None:
    """One bar per arm: AI recall at the budget, on untouched evaluation images."""
    import matplotlib.pyplot as plt

    arms, values = [], []
    for arm, runs in results["runs"].items():
        recalls = [r["aggregations"][r["selected_aggregation"]]["defactify_evaluation_recall"]
                   for r in runs]
        arms.append(ARM_LABEL.get(arm, arm))
        values.append(100 * float(np.mean(recalls)))
    order = np.argsort(values)
    arms = [arms[i] for i in order]
    values = [values[i] for i in order]
    colours = [MUTED] * len(values)
    colours[-1] = NAVY

    fig, ax = plt.subplots(figsize=(6.6, 4.5), dpi=200)
    ax.bar(arms, values, color=colours, width=0.58)
    for i, v in enumerate(values):
        ax.text(i, v + 1.4, f"%{v:.1f}", ha="center", fontsize=13,
                color=NAVY, weight="bold")
    ax.set_ylim(0, max(values) * 1.28)
    ax.set_ylabel("yakalanan AI görsel oranı", fontsize=9.5, color=GREY)
    ax.set_title(f"Aynı parçalar, aynı veri — sadece model değişti\n"
                 f"%{int(budget * 100)} yanlış alarm bütçesinde",
                 fontsize=11.5, color=NAVY, weight="bold", pad=14)
    _frame(ax)
    fig.tight_layout()
    fig.savefig(out / "arms.png", facecolor="white")
    plt.close(fig)


def figure_roc(real: np.ndarray, ai: np.ndarray, threshold: float, auc: float,
               out: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(np.r_[np.zeros(len(real)), np.ones(len(ai))], np.r_[real, ai])
    here_fpr = float((real >= threshold).mean())
    here_tpr = float((ai >= threshold).mean())

    fig, ax = plt.subplots(figsize=(6.0, 4.6), dpi=200)
    ax.plot(fpr, tpr, color=NAVY, lw=2.6)
    ax.plot([0, 1], [0, 1], "--", color="#B9C0CE", lw=1.3)
    ax.scatter([here_fpr], [here_tpr], s=150, color=ALERT, zorder=5,
               edgecolor="white", linewidth=2)
    ax.annotate(f"seçilen eşik\n%{100 * here_tpr:.0f} yakalama / %{100 * here_fpr:.0f} yanlış alarm",
                (here_fpr, here_tpr), xytext=(here_fpr + 0.14, here_tpr - 0.24),
                fontsize=9, color=ALERT,
                arrowprops=dict(arrowstyle="->", color=ALERT, lw=1.4))
    ax.set_xlabel("Yanlış alarm oranı  (masum fotoğrafa \"AI\" deme)", fontsize=9.5, color=GREY)
    ax.set_ylabel("Yakalama oranı  (AI'ı doğru bulma)", fontsize=9.5, color=GREY)
    ax.set_title(f"Eşik kaydıkça iki hata takas edilir   (AUC {auc:.3f})",
                 fontsize=11, color=NAVY, weight="bold", pad=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _frame(ax, grid_axis="both")
    fig.tight_layout()
    fig.savefig(out / "roc.png", facecolor="white")
    plt.close(fig)


def figure_scores(real: np.ndarray, ai: np.ndarray, threshold: float, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 3.5), dpi=200)
    bins = np.linspace(0, 1, 44)
    ax.hist(real, bins=bins, color=OK, alpha=0.72, label="gerçek fotoğraf")
    ax.hist(ai, bins=bins, color=ALERT, alpha=0.62, label="AI görsel")
    ax.axvline(threshold, color=NAVY, lw=2.2)
    ax.text(threshold - 0.02, ax.get_ylim()[1] * 0.92, f"eşik {threshold:.3f}",
            ha="right", fontsize=9, color=NAVY, weight="bold")
    ax.set_xlabel("Modelin verdiği puan", fontsize=9.5, color=GREY)
    ax.set_ylabel("görsel sayısı", fontsize=9.5, color=GREY)
    ax.set_title("İki sınıf büyük ölçüde üst üste biniyor",
                 fontsize=11, color=NAVY, weight="bold", pad=10)
    ax.legend(fontsize=9, frameon=False)
    _frame(ax, grid_axis=None)
    fig.tight_layout()
    fig.savefig(out / "scores.png", facecolor="white")
    plt.close(fig)


def figure_sources(source_fp: dict, budget: float, out: Path) -> None:
    """False positives per authentic source — the evaluator's own numbers."""
    import matplotlib.pyplot as plt

    rows = sorted(((k, 100 * v) for k, v in source_fp.items()), key=lambda kv: kv[1])
    names = [k for k, _ in rows]
    values = [v for _, v in rows]
    colours = [ALERT if v >= 50 else (WARN if v >= 30 else OK) for v in values]

    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=200)
    ax.barh(names, values, color=colours, height=0.66)
    for i, v in enumerate(values):
        ax.text(v + 1.4, i, f"%{v:.0f}", va="center", fontsize=9, color=NAVY, weight="bold")
    ax.axvline(100 * budget, color=NAVY, ls="--", lw=1.4)
    ax.text(100 * budget + 1.4, -0.85, f"hedef %{int(100 * budget)}", fontsize=8.5, color=NAVY)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Gerçek fotoğrafa yanlışlıkla \"AI\" deme oranı", fontsize=9.5, color=GREY)
    ax.set_title("Aynı eşik, farklı kamera kaynakları\nHepsi GERÇEK fotoğraf",
                 fontsize=11, color=NAVY, weight="bold", pad=12)
    _frame(ax, grid_axis="x")
    fig.tight_layout()
    fig.savefig(out / "sources.png", facecolor="white")
    plt.close(fig)


def figure_generators(per_generator: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    rows = sorted(((k, 100 * v["evaluation_recall"]) for k, v in per_generator.items()),
                  key=lambda kv: kv[1])
    names = [k for k, _ in rows]
    values = [v for _, v in rows]

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    ax.barh(names, values, color=[ALERT if v < 30 else (WARN if v < 60 else OK) for v in values],
            height=0.6)
    for i, v in enumerate(values):
        ax.text(v + 1.4, i, f"%{v:.0f}", va="center", fontsize=9.5, color=NAVY, weight="bold")
    ax.set_xlim(0, 108)
    ax.set_xlabel("yakalanan AI görsel oranı", fontsize=9.5, color=GREY)
    ax.set_title("Üreteç başına başarı — aynı model, aynı eşik",
                 fontsize=11, color=NAVY, weight="bold", pad=12)
    _frame(ax, grid_axis="x")
    fig.tight_layout()
    fig.savefig(out / "generators.png", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw report figures from saved results.")
    parser.add_argument("--results", type=Path, default=Path("artifacts/e20/results.json"))
    parser.add_argument("--arm", default="resnet18",
                        help="which arm the per-image figures describe")
    parser.add_argument("--output", type=Path, default=Path("artifacts/figures"))
    args = parser.parse_args()

    results = json.loads(args.results.read_text())
    protocol = results["protocol"]
    budget = protocol["fp_budget"]
    args.output.mkdir(parents=True, exist_ok=True)

    figure_arms(results, args.output, budget)
    print(f"  arms.png          {len(results['runs'])} arm")

    run = results["runs"][args.arm][0]
    rule = run["selected_aggregation"]
    records = load_scores(Path(run["raw_scores"]), rule)
    calibration, evaluation = split_defactify(records, protocol["calibration_fraction"],
                                              protocol["split_seed"])
    cal_real = np.array([r["score"] for r in calibration if r["label"] == 0])
    ev_real = np.array([r["score"] for r in evaluation if r["label"] == 0])
    ev_ai = np.array([r["score"] for r in evaluation if r["label"] == 1])
    threshold = threshold_at_fpr(cal_real, budget)
    measured = run["aggregations"][rule]
    auc = measured["defactify_evaluation_auc"]

    figure_roc(ev_real, ev_ai, threshold, auc, args.output)
    figure_scores(ev_real, ev_ai, threshold, args.output)
    figure_sources(measured["forensics_source_fp"], budget, args.output)
    figure_generators(measured["per_generator"], args.output)
    for name in ("roc", "scores", "sources", "generators"):
        print(f"  {name + '.png':<18}{args.arm}, rule={rule}, threshold={threshold:.4f}")

    confusion = {"TP": int((ev_ai >= threshold).sum()), "FN": int((ev_ai < threshold).sum()),
                 "FP": int((ev_real >= threshold).sum()), "TN": int((ev_real < threshold).sum()),
                 "threshold": float(threshold), "auc": float(auc), "arm": args.arm, "rule": rule}
    (args.output / "confusion.json").write_text(json.dumps(confusion, indent=1))
    print(f"  confusion.json    TP {confusion['TP']}  FN {confusion['FN']}  "
          f"FP {confusion['FP']}  TN {confusion['TN']}")


if __name__ == "__main__":
    main()
