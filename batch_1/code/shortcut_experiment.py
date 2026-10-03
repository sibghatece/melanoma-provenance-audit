"""
Source-shortcut experiment on the two mirrors whose labels are almost fully
predictable from image provenance (Javid 10,605 and ISIC 2019+2020 11,400).

Question: does a standard classifier trained on such a mirror learn to
recognise the SOURCE of an image rather than the lesion?

Features: ImageNet-pretrained EfficientNet-B0 (NOT the HAM10000-trained v3
backbones, which have seen every HAM melanoma in these mirrors), 320 px, the
same four-stage pooled features and 2-view TTA as the main pipeline.
Classifier: XGBoost with the v3 settings.

For each mirror (its own train/test split):
  A. Melanoma/malignancy classifier trained on the mirror's train split,
     scored on the whole test split (what a published paper would report).
  B. The SAME model scored inside one source only: test images that are
     ISIC 2020 copies (the only source that contains both classes in
     useful numbers). If performance drops sharply here, the high overall
     score came largely from telling sources apart.
  C. A source classifier (ISIC 2020 copy vs not) trained on the same
     features: how easy is the shortcut to learn?
  D. Control: a classifier trained ONLY on ISIC 2020-source training images
     and scored on the ISIC 2020-source test images. This is what lesion
     discrimination looks like when source cannot help.
  E. Image properties (width, height, mean brightness) per source and class,
     a first look at what cue could carry the source.

Source labels come from results_mirror_audit/matches_<mirror>.csv (d <= 10).

Run:
  export MELANOMA_ROOT=/path/to/data
  caffeinate -dimsu python3 shortcut_experiment.py 2>&1 | tee shortcut_log.txt
Features are cached; expect roughly 15-30 minutes on first run.
Outputs in $MELANOMA_ROOT/results_shortcut/
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# MELANOMA_ROOT if set, otherwise the data/ folder of this repository
ROOT = Path(os.environ.get("MELANOMA_ROOT", Path(__file__).resolve().parents[1] / "data")).expanduser()
AUDIT = ROOT / "results_mirror_audit"
OUT = ROOT / "results_shortcut"
MIRRORS = {
    "Javid_10605": ROOT / "Javed_melanoma_cancer_dataset",
    "ISIC19_20_malig_benign_11400": ROOT / "skin-cancer-isic-2019-2020-malignant-or-benign",
}
PRIMARY = 10
SEED = 42
N_BOOT = 2000
IMG_SIZE = 320
STAGE_INDICES = [5, 6, 7, 8]


# ------------------------------------------------------------------ data
def load_mirror(name, root):
    m = pd.read_csv(AUDIT / f"matches_{name}.csv")
    m = m[m["split"].isin(["train", "test"]) &
          m["class"].str.lower().isin(["benign", "malignant"])].reset_index(drop=True)
    m["path"] = [str(root / r) for r in m["rel"]]
    m["y"] = (m["class"].str.lower() == "malignant").astype(int)
    m["source"] = np.select([m["HAM10000_nn_dist"] <= PRIMARY,
                             m["ISIC2020_nn_dist"] <= PRIMARY],
                            ["HAM10000", "ISIC2020"], "unmatched")
    m["is_isic2020"] = (m["source"] == "ISIC2020").astype(int)
    missing = [p for p in m["path"].head(20) if not Path(p).exists()]
    if missing:
        raise SystemExit(f"Image paths not found, e.g. {missing[0]}")
    return m


def image_properties(df):
    rows = []
    for p in df["path"]:
        with Image.open(p) as im:
            w, h = im.size
            small = np.asarray(im.convert("L").resize((64, 64)), dtype=np.float32)
        rows.append((w, h, small.mean()))
    out = df[["split", "class", "source"]].copy()
    out[["width", "height", "mean_brightness"]] = rows
    return out


# ------------------------------------------------------------------ features
def extract_features(paths, cache):
    if cache.exists():
        return np.load(cache)
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class DS(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            return tf(Image.open(paths[i]).convert("RGB"))

    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1).to(device).eval()
    feats = []
    with torch.no_grad():
        for bi, x in enumerate(DataLoader(DS(), batch_size=32, num_workers=0)):
            x = x.to(device)
            views = []
            for v in (x, torch.flip(x, [3])):
                h, parts = v, []
                for idx, block in enumerate(model.features):
                    h = block(h)
                    if idx in STAGE_INDICES:
                        parts.append(nn.functional.adaptive_avg_pool2d(h, 1).flatten(1))
                views.append(torch.cat(parts, 1))
            feats.append(torch.stack(views).mean(0).cpu().numpy())
            if bi % 100 == 0:
                print(f"    features {bi * 32}/{len(paths)}")
    X = np.concatenate(feats)
    np.save(cache, X)
    return X


# ------------------------------------------------------------------ models / stats
def xgb(y):
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                         eval_metric="logloss", random_state=SEED, n_jobs=-1)


def boot_ci(y, p, rng):
    vals = []
    n = len(y)
    while len(vals) < N_BOOT:
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        vals.append(roc_auc_score(y[i], p[i]))
    return np.percentile(vals, [2.5, 97.5])


def score(tag, y, p, rng, thr=0.5):
    lo, hi = boot_ci(y, p, rng)
    pred = (p >= thr).astype(int)
    return {"evaluation": tag, "n": len(y), "n_positive": int(y.sum()),
            "auroc": roc_auc_score(y, p), "auroc_ci_lo": lo, "auroc_ci_hi": hi,
            "accuracy": accuracy_score(y, pred),
            "balanced_accuracy": balanced_accuracy_score(y, pred)}


# ------------------------------------------------------------------ main
def run_mirror(name, root, rng):
    print(f"\n=== {name} ===")
    df = load_mirror(name, root)
    print(df.groupby(["split", "class", "source"]).size().unstack(fill_value=0).to_string())

    props = image_properties(df)
    prop_summary = (props.groupby(["source", "class"])[["width", "height", "mean_brightness"]]
                    .agg(["median", "min", "max"]).round(1))
    prop_summary.to_csv(OUT / f"image_properties_{name}.csv")
    sizes = (props.assign(size=props.width.astype(int).astype(str) + "x"
                          + props.height.astype(int).astype(str))
             .groupby(["source", "class"])["size"]
             .agg(lambda s: s.value_counts().head(3).to_dict()))
    print("\nMost common image sizes by source and class:")
    print(sizes.to_string())

    X = extract_features(df["path"].tolist(), OUT / f"features_{name}.npy")
    tr = (df["split"] == "train").values
    te = (df["split"] == "test").values
    y, s = df["y"].values, df["is_isic2020"].values
    sc = StandardScaler().fit(X[tr])
    Xs = sc.transform(X)

    rows = []
    # A. standard classifier
    clf = xgb(y[tr]).fit(Xs[tr], y[tr])
    p = clf.predict_proba(Xs)[:, 1]
    rows.append(score("A_full_test", y[te], p[te], rng))
    # B. same model inside the ISIC 2020 source only
    b = te & (s == 1)
    if len(np.unique(y[b])) == 2:
        rows.append(score("B_same_model_ISIC2020_source_only", y[b], p[b], rng))
    # C. how learnable is the source?
    src = xgb(s[tr]).fit(Xs[tr], s[tr])
    ps = src.predict_proba(Xs)[:, 1]
    rows.append(score("C_source_classifier_(ISIC2020_vs_not)", s[te], ps[te], rng))
    # D. control: train and test within the ISIC 2020 source only
    dtr = tr & (s == 1)
    if len(np.unique(y[dtr])) == 2 and len(np.unique(y[b])) == 2:
        sc_d = StandardScaler().fit(X[dtr])
        ctrl = xgb(y[dtr]).fit(sc_d.transform(X[dtr]), y[dtr])
        pd_ = ctrl.predict_proba(sc_d.transform(X[b]))[:, 1]
        rows.append(score("D_control_trained_and_tested_within_ISIC2020_source",
                          y[b], pd_, rng))

    res = pd.DataFrame(rows)
    res.insert(0, "mirror", name)
    print("\n" + res.round(4).to_string(index=False))
    return res


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    allres = [run_mirror(n, r, rng) for n, r in MIRRORS.items()]
    pd.concat(allres).to_csv(OUT / "shortcut_results.csv", index=False)
    print(f"\nAll outputs in {OUT}")


if __name__ == "__main__":
    main()
