"""Phase 2b: what did the CNN actually learn?

k-means clustering + t-SNE projection of the frozen embeddings, plus error
analysis. Clustering is used here as an exploration tool on top of a
supervised model — not as a classifier.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score

LABEL_NAMES = {0: "real", 1: "AI"}


def cluster_report(embeddings: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
    kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=seed)
    assignments = kmeans.fit_predict(embeddings)
    # Clusters are unordered; purity = how label-homogeneous each cluster is.
    purity = sum(max(np.sum((assignments == c) & (labels == l)) for l in (0, 1)) for c in (0, 1)) / len(labels)
    print(f"k-means (k=2, k-means++ init) on {len(labels)} embeddings:")
    print(f"  cluster purity            = {purity:.4f}")
    print(f"  adjusted rand index (ARI) = {adjusted_rand_score(labels, assignments):.4f}")
    for c in (0, 1):
        real, ai = np.sum((assignments == c) & (labels == 0)), np.sum((assignments == c) & (labels == 1))
        print(f"  cluster {c}: {real} real / {ai} AI")
    return assignments


def plot_tsne(embeddings, labels, correct, output: Path, seed: int, sample_size: int = 5000) -> None:
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(labels), size=min(sample_size, len(labels)), replace=False)
    projected = TSNE(n_components=2, random_state=seed, init="pca", perplexity=30).fit_transform(embeddings[sample])

    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    for label, color in ((0, "tab:blue"), (1, "tab:orange")):
        mask = labels[sample] == label
        axes[0].scatter(*projected[mask].T, s=3, alpha=0.5, c=color, label=LABEL_NAMES[label])
    axes[0].set_title("t-SNE of CNN embeddings, colored by true label")
    axes[0].legend(markerscale=4)

    for is_correct, color, name in ((True, "lightgray", "correct"), (False, "tab:red", "misclassified")):
        mask = correct[sample] == is_correct
        axes[1].scatter(*projected[mask].T, s=3 if is_correct else 8, alpha=0.5, c=color, label=name)
    axes[1].set_title("Same projection, CNN errors highlighted")
    axes[1].legend(markerscale=4)

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    print(f"saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, default=Path("artifacts/embeddings"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/figures"))
    args = parser.parse_args()

    data = np.load(args.embeddings / f"{args.split}.npz")
    embeddings, labels, probabilities = data["embeddings"], data["labels"], data["probabilities"]
    predictions = (probabilities >= 0.5).astype(int)
    correct = predictions == labels

    print(f"=== {args.split} split ===")
    assignments = cluster_report(embeddings, labels, args.seed)

    errors = ~correct
    print(f"\nerror analysis: {errors.sum()} misclassified ({errors.mean():.2%})")
    for c in (0, 1):
        in_cluster = assignments == c
        print(f"  cluster {c}: error rate {errors[in_cluster].mean():.2%}")
    borderline = np.abs(probabilities - 0.5) < 0.1
    print(f"  borderline predictions (|p-0.5|<0.1): {borderline.sum()} images, "
          f"error rate {errors[borderline].mean():.2%} (vs {errors[~borderline].mean():.2%} elsewhere)")

    args.output.mkdir(parents=True, exist_ok=True)
    plot_tsne(embeddings, labels, correct, args.output / f"tsne_{args.split}.png", args.seed)


if __name__ == "__main__":
    main()
