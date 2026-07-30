# ResNet-18 vs feature model — same training data, same test sets, same metrics.
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


def m(y, s):
    return roc_auc_score(y, s), float(((s >= .5).astype(int) == y).mean()), float((s[y == 1] >= .5).mean())


def pair(real, gen):
    return np.r_[np.zeros(len(real)), np.ones(len(gen))], np.r_[real, gen]


def main():
    # ---------- feature models ----------
    feat = {}
    for mode, crop in (("full", None), ("crop", 128)):
        xtr, ytr, _ = stack(["genimage_train_real", "genimage_train_ai"], crop, CACHE)
        gb = make_pipeline(StandardScaler(), HistGradientBoostingClassifier(random_state=42)).fit(xtr, ytr)
        scores = {}
        for key, names in (("genimage", ["genimage_test_real", "genimage_test_ai"]),
                           ("archive1", ["archive1_real", "archive1_ai"])):
            x, y, _ = stack(names, crop, CACHE)
            scores[key] = (y, gb.predict_proba(x)[:, 1])
        xr, _, _ = stack(["defactify_real"], crop, CACHE)
        scores["defactify_real"] = gb.predict_proba(xr)[:, 1]
        for g in GENS:
            xg, _, _ = stack([f"defactify_{g}"], crop, CACHE)
            scores[g] = gb.predict_proba(xg)[:, 1]
        feat[mode] = scores

    # ---------- ResNet ----------
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load("artifacts/best_genimage.pt", map_location=dev, weights_only=False)
    cfg = ck["config"]
    net = create_model(cfg["model"]["name"], dropout=cfg["model"]["dropout"]).to(dev)
    net.load_state_dict(ck["model"]); net.eval()
    tf = eval_transform(cfg["data"]["image_size"], cfg["data"].get("normalization", "default"))

    def run(folder, label):
        ds = FolderPairDataset(folder, folder, tf)
        ds.samples = [(p, label) for p, _ in ds.samples[:len(ds.samples) // 2]]
        _, pr = collect_predictions(net, DataLoader(ds, batch_size=128, num_workers=0), dev)
        return np.array(pr)

    res = {}
    for key, (fr, fa) in (("genimage", (H/"genimage_split/test/REAL", H/"genimage_split/test/FAKE")),
                          ("archive1", (H/"archive1/real_dataset", H/"archive1/Ai_generated_dataset"))):
        r, a = run(fr, 0), run(fa, 1)
        res[key] = (np.r_[np.zeros(len(r)), np.ones(len(a))], np.r_[r, a])
    res["defactify_real"] = run(H/"defactify_test/real", 0)
    for g in GENS:
        res[g] = run(H/"defactify_test/ai"/g, 1)

    # ---------- report ----------
    print("\n" + "=" * 86)
    print("TRAINING DATA — identical for every model below: GenImage train, 9,917 images")
    print("   4,954 real (ImageNet nature photos)")
    print("   4,963 AI  (7 generators: ADM, BigGAN, Midjourney-old, VQDM, GLIDE, SD 1.5, Wukong)")
    print("=" * 86)

    rows = [("ResNet-18 (resize 224)", res), ("Features full-image", feat["full"]), ("Features 128px crop", feat["crop"])]
    for title, key in (("GenImage test — 1,742 img — SAME generators as training (in-distribution)", "genimage"),
                       ("archive1 — 995 img — different source, unseen generators", "archive1")):
        print(f"\n--- {title} ---")
        print(f"{'model':<26}{'AUC':>8}{'acc':>8}{'AI recall':>11}")
        for name, store in rows:
            y, s = store[key]
            a, ac, rc = m(y, s)
            print(f"{name:<26}{a:8.3f}{ac:8.3f}{rc*100:10.1f}%")

    print("\n--- Defactify — 16,875 img — 5 MODERN generators, NONE seen in training ---")
    print(f"{'model':<26}{'AUC':>8}{'acc':>8}{'AI recall':>11}{'real->AI err':>14}")
    for name, store in rows:
        allai = np.concatenate([store[g] for g in GENS])
        y, s = pair(store["defactify_real"], allai)
        a, ac, rc = m(y, s)
        print(f"{name:<26}{a:8.3f}{ac:8.3f}{rc*100:10.1f}%{float((store['defactify_real'] >= .5).mean())*100:13.1f}%")

    print("\n--- PER GENERATOR (AUC) ---")
    print(f"{'generator':<14}{'src px':>8}{'ResNet':>10}{'feat full':>11}{'feat crop':>11}{'winner':>14}")
    for g in GENS:
        vals = [roc_auc_score(*pair(store["defactify_real"], store[g])) for _, store in rows]
        winner = ["ResNet", "feat full", "feat crop"][int(np.argmax(vals))]
        print(f"{g:<14}{SOURCE_RESOLUTION[g]:>8}{vals[0]:10.3f}{vals[1]:11.3f}{vals[2]:11.3f}{winner:>14}")

    print("\n--- ENSEMBLE CHECK: mean of ResNet + feature-crop probabilities ---")
    print(f"{'generator':<14}{'ResNet':>10}{'feat crop':>11}{'ensemble':>11}{'gain':>9}")
    for g in GENS:
        y, sr = pair(res["defactify_real"], res[g])
        _, sf = pair(feat["crop"]["defactify_real"], feat["crop"][g])
        ar, af, ae = roc_auc_score(y, sr), roc_auc_score(y, sf), roc_auc_score(y, (sr + sf) / 2)
        print(f"{g:<14}{ar:10.3f}{af:11.3f}{ae:11.3f}{ae - max(ar, af):+9.3f}")
    y, sr = pair(res["defactify_real"], np.concatenate([res[g] for g in GENS]))
    _, sf = pair(feat["crop"]["defactify_real"], np.concatenate([feat["crop"][g] for g in GENS]))
    ar, af, ae = roc_auc_score(y, sr), roc_auc_score(y, sf), roc_auc_score(y, (sr + sf) / 2)
    print(f"{'ALL':<14}{ar:10.3f}{af:11.3f}{ae:11.3f}{ae - max(ar, af):+9.3f}")


if __name__ == "__main__":
    main()
