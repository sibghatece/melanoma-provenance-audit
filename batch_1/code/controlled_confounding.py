"""
Experiment D: controlled source confounding with large numbers and no leakage.

The Javid mirror's structure (melanomas from one source, benign lesions from
another) is rebuilt deliberately from clean data, so the shortcut can be
measured with lesion- and patient-disjoint splits and hundreds of melanomas.

Sources
  HAM10000  : melanoma (1,113 images) and other diagnoses; grouped by lesion
  ISIC 2020 : melanoma (581 images) and other diagnoses; grouped by patient
Features
  ImageNet-pretrained EfficientNet-B0 without fine-tuning, 320 px, the same
  four-stage pooled features and 2-view TTA as the other experiments.
  (No model trained on HAM10000 is used, so no image has been seen before.)

Each of 5 repetitions draws a new 70/30 split, by lesion for HAM10000 and
by patient for ISIC 2020, and trains XGBoost on three training sets:
  CONFOUNDED  melanomas only from HAM10000, non-melanomas only from ISIC 2020
              (balanced 1:1, like the Javid mirror)
  CONTROL     both classes from both sources, 1:1 within each source, so the
              source carries no information about the label
  SOURCE      same images as CONTROL, label = source (HAM10000 or ISIC 2020)
and scores held-out images from unseen lesions and patients on:
  T_confounded  HAM10000 melanomas vs ISIC 2020 non-melanomas (1:1)
  T_ISIC        ISIC 2020 melanomas vs ISIC 2020 non-melanomas (within source)
  T_HAM         HAM10000 melanomas vs HAM10000 non-melanomas (within source)

Run:
  export MELANOMA_ROOT=/path/to/data
  caffeinate -dimsu python3 controlled_confounding.py 2>&1 | tee confounding_log.txt
Feature extraction (about 43,000 images) takes roughly 30-45 min once and is
cached; the experiment itself then takes a few minutes.
Outputs in $MELANOMA_ROOT/results_confounding/
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# MELANOMA_ROOT if set, otherwise the data/ folder of this repository
ROOT = Path(os.environ.get("MELANOMA_ROOT", Path(__file__).resolve().parents[1] / "data")).expanduser()
OUT = ROOT / "results_confounding"
N_REPEATS = 5
TEST_FRACTION = 0.30
IMG_SIZE = 320
STAGE_INDICES = [5, 6, 7, 8]


# ------------------------------------------------------------------ features
def extract(paths, cache):
    if cache.exists():
        X = np.load(cache)
        if len(X) == len(paths):
            return X
    import torch
    import torch.nn as nn
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    class DS(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            return tf(Image.open(paths[i]).convert("RGB"))

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1).to(device).eval()
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
            if bi % 200 == 0:
                print(f"    {cache.stem}: {bi * 32}/{len(paths)}")
    X = np.concatenate(feats)
    np.save(cache, X)
    return X


def find_ground_truth():
    env = os.environ.get("ISIC_GT")
    if env and Path(env).exists():
        return Path(env)
    pats = ["*2020*GroundTruth*.csv", "*GroundTruth*2020*.csv", "train.csv",
            "*2020*[Mm]etadata*.csv", "*[Mm]etadata*2020*.csv"]
    for base in [ROOT] + [d for d in ROOT.iterdir() if d.is_dir()]:
        for pat in pats:
            for f in sorted(base.glob(pat)):
                try:
                    cols = pd.read_csv(f, nrows=2).columns
                except Exception:
                    continue
                if "patient_id" in cols and "image_name" in cols:
                    return f
    raise SystemExit("ISIC 2020 ground-truth CSV (image_name, patient_id) not found; set ISIC_GT")


def xgb(y, seed):
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
                         random_state=seed, n_jobs=-1)


def fit_predict(Xtr, ytr, tests, seed):
    sc = StandardScaler().fit(Xtr)
    clf = xgb(ytr, seed).fit(sc.transform(Xtr), ytr)
    return {k: clf.predict_proba(sc.transform(X))[:, 1] for k, X in tests.items()}


# ------------------------------------------------------------------ one repetition
def repetition(r, Xh, yh, gh, Xi, yi, gi):
    rng = np.random.default_rng(r)
    h_tr, h_te = next(GroupShuffleSplit(1, test_size=TEST_FRACTION, random_state=r).split(Xh, yh, gh))
    i_tr, i_te = next(GroupShuffleSplit(1, test_size=TEST_FRACTION, random_state=r).split(Xi, yi, gi))
    assert not set(gh[h_tr]) & set(gh[h_te]), "lesion overlap"
    assert not set(gi[i_tr]) & set(gi[i_te]), "patient overlap"

    hm_tr, hn_tr = h_tr[yh[h_tr] == 1], h_tr[yh[h_tr] == 0]
    hm_te, hn_te = h_te[yh[h_te] == 1], h_te[yh[h_te] == 0]
    im_tr, in_tr = i_tr[yi[i_tr] == 1], i_tr[yi[i_tr] == 0]
    im_te, in_te = i_te[yi[i_te] == 1], i_te[yi[i_te] == 0]
    sub = lambda a, n: rng.choice(a, size=min(n, len(a)), replace=False)

    # training sets
    conf_neg = sub(in_tr, len(hm_tr))
    X_conf = np.vstack([Xh[hm_tr], Xi[conf_neg]])
    y_conf = np.r_[np.ones(len(hm_tr)), np.zeros(len(conf_neg))]

    ctrl_hn, ctrl_in = sub(hn_tr, len(hm_tr)), sub(in_tr, len(im_tr))
    X_ctrl = np.vstack([Xh[hm_tr], Xh[ctrl_hn], Xi[im_tr], Xi[ctrl_in]])
    y_ctrl = np.r_[np.ones(len(hm_tr)), np.zeros(len(ctrl_hn)), np.ones(len(im_tr)), np.zeros(len(ctrl_in))]
    s_ctrl = np.r_[np.ones(len(hm_tr) + len(ctrl_hn)), np.zeros(len(im_tr) + len(ctrl_in))]

    # test sets (unseen lesions and patients only)
    tconf_neg = sub(in_te, len(hm_te))
    tests = {
        "T_confounded": np.vstack([Xh[hm_te], Xi[tconf_neg]]),
        "T_ISIC": np.vstack([Xi[im_te], Xi[in_te]]),
        "T_HAM": np.vstack([Xh[hm_te], Xh[hn_te]]),
    }
    labels = {
        "T_confounded": np.r_[np.ones(len(hm_te)), np.zeros(len(tconf_neg))],
        "T_ISIC": np.r_[np.ones(len(im_te)), np.zeros(len(in_te))],
        "T_HAM": np.r_[np.ones(len(hm_te)), np.zeros(len(hn_te))],
    }
    src_neg = sub(in_te, len(h_te))
    src_test = {"T_source": np.vstack([Xh[h_te], Xi[src_neg]])}
    src_lab = np.r_[np.ones(len(h_te)), np.zeros(len(src_neg))]

    rows = []
    for model_name, (Xtr, ytr) in {"CONFOUNDED": (X_conf, y_conf), "CONTROL": (X_ctrl, y_ctrl)}.items():
        preds = fit_predict(Xtr, ytr, tests, r)
        for t, p in preds.items():
            rows.append({"repeat": r, "model": model_name, "test": t,
                         "n": len(labels[t]), "n_melanoma": int(labels[t].sum()),
                         "auroc": roc_auc_score(labels[t], p)})
    p_src = fit_predict(X_ctrl, s_ctrl, src_test, r)["T_source"]
    rows.append({"repeat": r, "model": "SOURCE", "test": "T_source", "n": len(src_lab),
                 "n_melanoma": np.nan, "auroc": roc_auc_score(src_lab, p_src)})
    return rows


# ------------------------------------------------------------------ main
def main():
    print(f"[Data folder] {ROOT}")
    if not (ROOT / "manifest_ham10000_384.csv").exists():
        raise SystemExit(f"manifest_ham10000_384.csv not found in {ROOT}. Run code/prepare_data.py "
                         "first, or set MELANOMA_ROOT to your data folder.")
    OUT.mkdir(parents=True, exist_ok=True)
    ham = pd.read_csv(ROOT / "manifest_ham10000_384.csv")
    isic = pd.read_csv(ROOT / "manifest_isic2020_384.csv")
    gt = pd.read_csv(find_ground_truth())
    pid = dict(zip(gt["image_name"].astype(str), gt["patient_id"].astype(str)))
    gi = np.array([pid.get(Path(p).stem) for p in isic["image_path"]], dtype=object)
    if np.mean([g is not None for g in gi]) < 0.99:
        raise SystemExit("Fewer than 99% of ISIC 2020 images matched a patient_id.")
    print(f"HAM10000: {len(ham)} images, {int(ham['label'].sum())} melanomas, "
          f"{ham['lesion_group'].nunique()} lesions | ISIC 2020: {len(isic)} images, "
          f"{int(isic['label'].sum())} melanomas, {len(set(gi))} patients")

    print("Extracting ImageNet features (cached after the first run)...")
    Xh = extract(ham["image_path"].tolist(), OUT / "features_ham10000.npy")
    Xi = extract(isic["image_path"].tolist(), OUT / "features_isic2020.npy")
    yh, gh = ham["label"].values.astype(int), ham["lesion_group"].values
    yi = isic["label"].values.astype(int)

    rows = []
    for r in range(N_REPEATS):
        rr = repetition(r, Xh, yh, gh, Xi, yi, gi)
        rows += rr
        print(f"  repeat {r}: " + " | ".join(f"{x['model'][:4]} {x['test']} {x['auroc']:.3f}" for x in rr))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "confounding_per_repeat.csv", index=False)
    summ = (df.groupby(["model", "test"])
              .agg(n=("n", "mean"), n_melanoma=("n_melanoma", "mean"),
                   auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
                   auroc_min=("auroc", "min"), auroc_max=("auroc", "max"))
              .reset_index())
    summ.to_csv(OUT / "confounding_summary.csv", index=False)
    print("\n=== Controlled confounding, mean over "
          f"{N_REPEATS} lesion- and patient-disjoint repetitions ===")
    print(summ.round(4).to_string(index=False))
    print(f"\nAll outputs in {OUT}")


if __name__ == "__main__":
    main()
