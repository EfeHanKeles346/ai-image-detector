"""Phase 2a: train classical ML classifiers on frozen CNN embeddings.

Uses the same seeded 90k/10k train/validation split as the CNN so the
comparison against the CNN's own classification head is fair.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import LinearSVC

CLASSIFIERS = {
    "logistic_regression": lambda seed: LogisticRegression(max_iter=2000, random_state=seed),
    "linear_svm": lambda seed: LinearSVC(random_state=seed),
    "random_forest": lambda seed: RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed),
    "hist_gradient_boosting": lambda seed: HistGradientBoostingClassifier(random_state=seed),
}


def scores(classifier, embeddings: np.ndarray) -> np.ndarray:
    if hasattr(classifier, "predict_proba"):
        return classifier.predict_proba(embeddings)[:, 1]
    return classifier.decision_function(embeddings)


def report_row(name: str, labels: np.ndarray, continuous: np.ndarray, threshold: float) -> str:
    predictions = (continuous >= threshold).astype(int)
    return (f"| {name} | {accuracy_score(labels, predictions):.4f} | "
            f"{f1_score(labels, predictions):.4f} | {roc_auc_score(labels, continuous):.4f} |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, default=Path("artifacts/embeddings"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train = np.load(args.embeddings / "train.npz")
    test = np.load(args.embeddings / "test.npz")
    external_path = args.embeddings / "external.npz"
    external = np.load(external_path) if external_path.exists() else None

    # Reproduce data.py's seeded permutation: first 10% is validation, rest is train.
    indices = torch.randperm(len(train["labels"]), generator=torch.Generator().manual_seed(args.seed)).tolist()
    train_indices = indices[int(len(indices) * 0.1):]
    x_train, y_train = train["embeddings"][train_indices], train["labels"][train_indices]

    print(f"training on {len(y_train)} embeddings ({x_train.shape[1]}-dim)\n")
    fitted = {}
    for name, build in CLASSIFIERS.items():
        fitted[name] = build(args.seed)
        fitted[name].fit(x_train, y_train)
        print(f"fitted {name}")

    for split_name, split in (("test", test), ("external", external)):
        if split is None:
            continue
        print(f"\n### {split_name} set ({len(split['labels'])} images)")
        print("| model | accuracy | f1 | roc_auc |\n|---|---|---|---|")
        print(report_row("cnn_head (reference)", split["labels"], split["probabilities"], 0.5))
        for name, classifier in fitted.items():
            threshold = 0.5 if hasattr(classifier, "predict_proba") else 0.0
            print(report_row(name, split["labels"], scores(classifier, split["embeddings"]), threshold))


if __name__ == "__main__":
    main()
