# ResNet-18 vs feature model vs their combinations — same training data, same test sets.
import numpy as np, torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from pixelproof.evaluate import eval_transform, FolderPairDataset, collect_predictions
from pixelproof.models import create_model
from pixelproof.feature_experiment import stack, DEFACTIFY_GENERATORS, SOURCE_RESOLUTION

CACHE = Path("artifacts/features")
GENS = DEFACTIFY_GENERATORS
H = Path.home() / "Desktop"


def rank(scores):
    """Convert to within-set percentile ranks — removes probability-scale mismatch."""
    order = scores.argsort().argsort()
    return order / max(len(scores) - 1, 1)


def main():
    # ---------- feature model (128px native crop — the stronger of the two modes) ----------
    xtr, ytr, _ = stack(["genimage_train_real", "genimage_train_ai"], 128, CACHE)
    gb = make_pipeline(StandardScaler(), HistGradientBoostingClassifier(random_state=42)).fit(xtr, ytr)

    def fscore(names):
        x, _, _ = stack(names, 128, CACHE)
        return gb.predict_proba(x)[:, 1]

    feat = {"genimage_real": fscore(["genimage_test_real"]), "genimage_ai": fscore(["genimage_test_ai"]),
            "archive1_real": fscore(["archive1_real"]), "archive1_ai": fscore(["archive1_ai"]),
            "defactify_real": fscore(["defactify_real"])}
    for g in GENS:
        feat[g] = fscore([f"defactify_{g}"])

    # ---------- ResNet ----------
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load("artifacts/best_genimage.pt", map_location=dev, weights_only=False)
    cfg = ck["config"]
    net = create_model(cfg["model"]["name"], dropout=cfg["model"]["dropout"]).to(dev)
    net.load_state_dict(ck["model"]); net.eval()
    tf = eval_transform(cfg["data"]["image_size"], cfg["data"].get("normalization", "default"))

    def run(folder):
        ds = FolderPairDataset(folder, folder, tf)
        ds.samples = [(p, 0) for p, _ in ds.samples[:len(ds.samples) // 2]]
        _, pr = collect_predictions(net, DataLoader(ds, batch_size=128, num_workers=0), dev)
        return np.array(pr)

    res = {"genimage_real": run(H/"genimage_split/test/REAL"), "genimage_ai": run(H/"genimage_split/test/FAKE"),
           "archive1_real": run(H/"archive1/real_dataset"), "archive1_ai": run(H/"archive1/Ai_generated_dataset"),
           "defactify_real": run(H/"defactify_test/real")}
    for g in GENS:
        res[g] = run(H/"defactify_test/ai"/g)

    # ---------- combination strategies ----------
    def combos(r, f):
        """r, f = ResNet and feature probabilities over the SAME set of images."""
        rr, fr_ = rank(r), rank(f)
        return {
            "ResNet alone":        r,
            "Features alone":      f,
            "mean 50/50":          (r + f) / 2,
            "weighted 75/25":      0.75 * r + 0.25 * f,
            "max":                 np.maximum(r, f),
            "min":                 np.minimum(r, f),
            "rank mean":           (rr + fr_) / 2,
            "rank weighted 75/25": 0.75 * rr + 0.25 * fr_,
        }

    def evaluate(real_r, real_f, ai_r, ai_f):
        y = np.r_[np.zeros(len(real_r)), np.ones(len(ai_r))]
        out = {}
        for name, s in combos(np.r_[real_r, ai_r], np.r_[real_f, ai_f]).items():
            out[name] = roc_auc_score(y, s)
        return out

    SETS = {
        "GenImage test (in-distribution)": ("genimage_real", "genimage_ai"),
        "archive1 (different source)":     ("archive1_real", "archive1_ai"),
    }

    print("\n" + "=" * 92)
    print("ENSEMBLE TEST — ResNet-18 + feature model (both trained on the same GenImage split)")
    print("=" * 92)

    all_rows = {}
    for title, (rk, ak) in SETS.items():
        all_rows[title] = evaluate(res[rk], feat[rk], res[ak], feat[ak])
    allai_r = np.concatenate([res[g] for g in GENS])
    allai_f = np.concatenate([feat[g] for g in GENS])
    all_rows["Defactify (5 unseen modern)"] = evaluate(res["defactify_real"], feat["defactify_real"], allai_r, allai_f)

    names = list(next(iter(all_rows.values())).keys())
    print(f"\n{'strategy':<22}" + "".join(f"{t.split(' (')[0][:16]:>18}" for t in all_rows) + f"{'mean':>9}")
    print("-" * 92)
    for n in names:
        vals = [all_rows[t][n] for t in all_rows]
        star = "  <-- best" if n != "ResNet alone" and np.mean(vals) > np.mean([all_rows[t]["ResNet alone"] for t in all_rows]) else ""
        print(f"{n:<22}" + "".join(f"{v:18.3f}" for v in vals) + f"{np.mean(vals):9.3f}{star}")

    print(f"\n--- PER GENERATOR on Defactify (AUC) ---")
    print(f"{'generator':<13}{'px':>6}" + "".join(f"{n[:15]:>17}" for n in names))
    for g in GENS:
        row = evaluate(res["defactify_real"], feat["defactify_real"], res[g], feat[g])
        print(f"{g:<13}{SOURCE_RESOLUTION[g]:>6}" + "".join(f"{row[n]:17.3f}" for n in names))


if __name__ == "__main__":
    main()
