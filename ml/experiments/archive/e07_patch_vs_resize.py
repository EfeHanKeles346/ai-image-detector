# Küçültme vs parça-kesme karşılaştırması.
# Aynı resimlerde iki yöntemi yan yana ölçer:
#   A) küçült  : resmi 224x224'e sıkıştır (mevcut yöntem)
#   B) parça   : resimden orijinal kalitede 224x224'lük parçalar kes, ortalamasını al

import random
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms

from pixelproof.models import create_model
from pixelproof.project_paths import LEGACY_DATA_ROOT

PATCH = 224
N_PER_CLASS = 1000
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
shrink = transforms.Compose([transforms.Resize((PATCH, PATCH)), transforms.ToTensor(),
                             transforms.Normalize(MEAN, STD)])

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ck = torch.load("artifacts/best_genimage.pt", map_location=device, weights_only=False)
model = create_model(ck["config"]["model"]["name"], dropout=ck["config"]["model"]["dropout"]).to(device)
model.load_state_dict(ck["model"])
model.eval()


def patches(image):
    """Orijinal kalitede 224'lük parçalar: merkez + 4 köşe. Küçültme yok."""
    w, h = image.size
    if w < PATCH or h < PATCH:                       # parça sığmıyorsa en kısa kenarı 224 yap
        scale = PATCH / min(w, h)
        image = image.resize((max(PATCH, int(w * scale)), max(PATCH, int(h * scale))), Image.BICUBIC)
        w, h = image.size
    cx, cy = (w - PATCH) // 2, (h - PATCH) // 2
    spots = [(cx, cy), (0, 0), (w - PATCH, 0), (0, h - PATCH), (w - PATCH, h - PATCH)]
    return [image.crop((x, y, x + PATCH, y + PATCH)) for x, y in dict.fromkeys(spots)]


@torch.no_grad()
def score(files):
    shrink_p, patch_p = [], []
    for i in range(0, len(files), 32):
        chunk = [Image.open(f).convert("RGB") for f in files[i:i + 32]]
        batch = torch.stack([shrink(im) for im in chunk]).to(device)
        shrink_p += torch.sigmoid(model(batch)).cpu().tolist()
        for im in chunk:                              # her resmin parçaları, ortalaması alınır
            crops = torch.stack([to_tensor(c) for c in patches(im)]).to(device)
            patch_p.append(torch.sigmoid(model(crops)).mean().item())
    return shrink_p, patch_p


root = LEGACY_DATA_ROOT / "defactify_test"
random.seed(42)
folders = {"real": root / "real", **{g: root / "ai" / g for g in
           ["dalle3", "midjourney", "sd21", "sd3", "sdxl"]}}

results = {}
for name, folder in folders.items():
    files = sorted(folder.glob("*.jpg"))
    random.shuffle(files)
    results[name] = score(files[:N_PER_CLASS])
    s, p = results[name]
    print(f"{name:12s} n={len(s):4d}  küçültme p_ai={sum(s)/len(s):.3f}   parça p_ai={sum(p)/len(p):.3f}")

torch.save(results, "/private/tmp/claude-501/-Users-efehankeles-Desktop-ai-image-detector/"
                    "238e1a7f-1cab-4e73-92da-2333ce2ae064/scratchpad/patch_results.pt")

print(f"\n{'araç':<13}{'KÜÇÜLTME':>20}{'PARÇA':>20}")
print(f"{'':13}{'AUC':>10}{'recall':>10}{'AUC':>10}{'recall':>10}")
print("-" * 53)
real_s, real_p = results["real"]
for g in ["dalle3", "midjourney", "sd21", "sd3", "sdxl"]:
    gs, gp = results[g]
    y = [0] * len(real_s) + [1] * len(gs)
    a1, a2 = roc_auc_score(y, real_s + gs), roc_auc_score(y, real_p + gp)
    r1 = sum(v >= .5 for v in gs) / len(gs)
    r2 = sum(v >= .5 for v in gp) / len(gp)
    print(f"{g:<13}{a1:10.3f}{r1*100:9.1f}%{a2:10.3f}{r2*100:9.1f}%")

alls = [v for g in folders if g != "real" for v in results[g][0]]
allp = [v for g in folders if g != "real" for v in results[g][1]]
y = [0] * len(real_s) + [1] * len(alls)
print("-" * 53)
print(f"{'TÜMÜ':<13}{roc_auc_score(y, real_s + alls):10.3f}"
      f"{sum(v>=.5 for v in alls)/len(alls)*100:9.1f}%"
      f"{roc_auc_score(y, real_p + allp):10.3f}"
      f"{sum(v>=.5 for v in allp)/len(allp)*100:9.1f}%")
print(f"\ngerçek fotoğrafa yanlışlıkla 'AI' deme oranı: "
      f"küçültme %{sum(v>=.5 for v in real_s)/len(real_s)*100:.1f}  ->  "
      f"parça %{sum(v>=.5 for v in real_p)/len(real_p)*100:.1f}")
