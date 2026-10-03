"""
Step 2: what a contaminated "external" test looks like, measured with a
paired design.

Idea
  Each Kaggle image that duplicates a HAM10000 image belongs to one HAM
  lesion. In the lesion-grouped v3 run that lesion sat in the TEST partition
  of exactly one fold (the fold model never saw it) and in the training or
  inner-validation partition of the other folds. So the same Kaggle image is
  scored by models that saw its twin and by one model that did not. Image
  content, label, and the Kaggle resize/re-compression shift are identical;
  only prior exposure differs.

Labels
  Taken from the matched HAM10000 image (melanoma vs other). Kaggle's own
  "malignant" folder mixes melanoma with other cancers, so it is used only in
  a secondary, clearly labelled row.

Models
  Backbones: results_v3/model_fold1..5.pt (no retraining).
  Shallow stage: XGBoost on all 1,904 multi-stage features, refitted per fold
  on that fold's training features with the v3 settings (the best v3
  configuration). Features use the v3 eval transform and 2-view TTA.

Run
  export MELANOMA_ROOT=/path/to/data
  export KAGGLE_ROOT=/path/to/data/melanoma-skin-cancer
  caffeinate -dimsu python3 kaggle_seen_unseen.py 2>&1 | tee kaggle_seen_unseen_log.txt
Resumable: hashes and per-fold features are cached.

Outputs in $MELANOMA_ROOT/results_kaggle_twins/
  kaggle_ham_matches.csv        every Kaggle image, nearest HAM image, distance
  per_fold_exposure.csv          AUROC per fold on seen-in-train / seen-in-val / unseen twins
  paired_seen_unseen.csv         pooled paired comparison with cluster-bootstrap CI
  scores_per_image.csv           per-image probabilities by exposure status
"""

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# MELANOMA_ROOT if set, otherwise the data/ folder of this repository
ROOT = Path(os.environ.get("MELANOMA_ROOT", Path(__file__).resolve().parents[1] / "data")).expanduser()
KAGGLE_ROOT = Path(os.environ.get("KAGGLE_ROOT", ROOT / "melanoma-skin-cancer")).expanduser()
V3 = ROOT / "results_v3"
OUT = ROOT / "results_kaggle_twins"
CACHE = OUT / "cache"

IMG_SIZE = 320
STAGE_INDICES = [5, 6, 7, 8]
N_FOLDS, SEED, VAL_FRACTION = 5, 42, 0.15
HASH_SIZE = 16            # 256-bit phash, as in the original audit
MATCH_THRESHOLD = 10      # Hamming distance used in the paper
N_BOOT = 2000

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ------------------------------------------------------------------ hashing
def phash_matrix(paths, tag):
    cache = CACHE / f"phash_{tag}.npy"
    if cache.exists():
        return np.load(cache)
    import imagehash
    rows = []
    for i, p in enumerate(paths):
        h = imagehash.phash(Image.open(p).convert("RGB"), hash_size=HASH_SIZE)
        rows.append(h.hash.flatten().astype(np.float32))
        if (i + 1) % 2500 == 0:
            print(f"    hashed {i + 1}/{len(paths)} {tag}")
    mat = np.vstack(rows)
    np.save(cache, mat)
    return mat


def nearest_neighbours(A, B, chunk=512):
    """Nearest B row for each A row under Hamming distance on binary vectors."""
    b_sum = B.sum(axis=1)[None, :]
    nn_idx = np.empty(len(A), dtype=np.int64)
    nn_d = np.empty(len(A), dtype=np.float32)
    for s in range(0, len(A), chunk):
        a = A[s:s + chunk]
        D = a.sum(axis=1)[:, None] + b_sum - 2.0 * (a @ B.T)
        nn_idx[s:s + chunk] = D.argmin(axis=1)
        nn_d[s:s + chunk] = D.min(axis=1)
    return nn_idx, nn_d


def kaggle_files():
    rows = []
    for split in ["train", "test"]:
        for cls in ["benign", "malignant"]:
            d = KAGGLE_ROOT / split / cls
            if not d.exists():
                raise SystemExit(f"Missing Kaggle folder: {d}")
            for p in sorted(d.glob("*.jpg")):
                rows.append({"kaggle_path": str(p), "kaggle_split": split,
                             "kaggle_label_malignant": int(cls == "malignant")})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ model / features
class PathDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return eval_tf(Image.open(self.paths[i]).convert("RGB"))


def load_fold_model(fold):
    m = models.efficientnet_b0(weights=None)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, 2))
    m.load_state_dict(torch.load(V3 / f"model_fold{fold}.pt", map_location="cpu"))
    return m.to(DEVICE).eval()


def features(model, paths, tag):
    cache = CACHE / f"feat_{tag}.npy"
    if cache.exists():
        return np.load(cache)
    loader = DataLoader(PathDataset(paths), batch_size=32, shuffle=False, num_workers=0)
    out = []
    with torch.no_grad():
        for bi, x in enumerate(loader):
            x = x.to(DEVICE)
            views = [x, torch.flip(x, [3])]
            f = []
            for v in views:
                h, parts = v, []
                for idx, block in enumerate(model.features):
                    h = block(h)
                    if idx in STAGE_INDICES:
                        parts.append(nn.functional.adaptive_avg_pool2d(h, 1).flatten(1))
                f.append(torch.cat(parts, 1))
            out.append(torch.stack(f).mean(0).cpu().numpy())
            if bi % 100 == 0:
                print(f"    [{tag}] {bi * 32}/{len(paths)}")
    X = np.concatenate(out)
    np.save(cache, X)
    return X


def v3_partitions(y, groups):
    """Rebuild the v3 lesion-grouped train / inner-val / test partitions."""
    idx_all = np.arange(len(y))
    parts = []
    for tr_idx, te_idx in StratifiedGroupKFold(N_FOLDS, shuffle=True,
                                               random_state=SEED).split(idx_all, y, groups):
        gss = GroupShuffleSplit(1, test_size=VAL_FRACTION, random_state=SEED)
        i_tr, i_va = next(gss.split(tr_idx, y[tr_idx], groups[tr_idx]))
        parts.append((tr_idx[i_tr], tr_idx[i_va], te_idx))
    return parts


def xgb(y):
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                         eval_metric="logloss", random_state=SEED, n_jobs=-1)


# ------------------------------------------------------------------ stats
def auc_or_nan(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def paired_bootstrap(y, p_a, p_b, clusters, rng):
    codes, uniq = pd.factorize(clusters)
    members = [np.where(codes == c)[0] for c in range(len(uniq))]
    deltas = []
    while len(deltas) < N_BOOT:
        idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return lo, hi, max(p, 1.0 / N_BOOT)


# ------------------------------------------------------------------ main
def main():
    OUT.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    ham = pd.read_csv(ROOT / "manifest_ham10000_384.csv")
    y = ham["label"].values.astype(int)
    groups = ham["lesion_group"].values
    parts = v3_partitions(y, groups)

    # the rebuilt test partitions must match the saved v3 run
    for f in range(N_FOLDS):
        saved = np.load(V3 / "probs" / f"fold{f + 1}.npz")["test_index"]
        if not np.array_equal(np.sort(saved), np.sort(parts[f][2])):
            raise SystemExit(f"Rebuilt fold {f + 1} does not match results_v3; stop.")
    print("v3 partitions rebuilt and verified against saved test indices.")

    # ---- 1. match Kaggle images to HAM10000
    kag = kaggle_files()
    print(f"Kaggle images: {len(kag)} "
          f"(train {int((kag.kaggle_split == 'train').sum())}, "
          f"test {int((kag.kaggle_split == 'test').sum())})")
    print("Hashing (cached after first run)...")
    Hk = phash_matrix(kag["kaggle_path"].tolist(), "kaggle")
    Hh = phash_matrix(ham["image_path"].tolist(), "ham")
    nn_idx, nn_d = nearest_neighbours(Hk, Hh)
    kag["ham_index"] = nn_idx
    kag["hamming_distance"] = nn_d
    kag["matched"] = nn_d <= MATCH_THRESHOLD
    kag["ham_lesion"] = groups[nn_idx]
    kag["ham_label_melanoma"] = y[nn_idx]
    kag.to_csv(OUT / "kaggle_ham_matches.csv", index=False)
    for s in ["test", "train"]:
        sub = kag[kag.kaggle_split == s]
        print(f"  Kaggle {s}: {int(sub.matched.sum())}/{len(sub)} matched at d<={MATCH_THRESHOLD}")

    m = kag[kag.matched].reset_index(drop=True)
    print(f"Matched images used: {len(m)} "
          f"({int(m.ham_label_melanoma.sum())} melanoma by HAM label, "
          f"{m.ham_lesion.nunique()} HAM lesions)")

    # ---- 2. per-fold scoring
    status = np.empty((len(m), N_FOLDS), dtype=object)
    probs = np.full((len(m), N_FOLDS), np.nan)
    per_fold = []
    for f in range(N_FOLDS):
        fold = f + 1
        t0 = time.time()
        tr, va, te = parts[f]
        lesion_part = {}
        for part, idx in [("train", tr), ("val", va), ("test", te)]:
            for g in np.unique(groups[idx]):
                lesion_part[g] = part
        status[:, f] = [lesion_part[g] for g in m.ham_lesion]

        model = load_fold_model(fold)
        Xtr = features(model, ham["image_path"].values[tr], f"ham_train_fold{fold}")
        Xk = features(model, m["kaggle_path"].values, f"kaggle_matched_fold{fold}")
        sc = StandardScaler().fit(Xtr)
        clf = xgb(y[tr]).fit(sc.transform(Xtr), y[tr])
        probs[:, f] = clf.predict_proba(sc.transform(Xk))[:, 1]
        del model
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

        yk = m.ham_label_melanoma.values
        row = {"fold": fold}
        for st in ["train", "val", "test"]:
            sel = status[:, f] == st
            row[f"n_{st}"] = int(sel.sum())
            row[f"auroc_twin_in_{st}"] = auc_or_nan(yk[sel], probs[sel, f])
        per_fold.append(row)
        print(f"  fold {fold}: twin-in-train AUROC={row['auroc_twin_in_train']:.4f} "
              f"(n={row['n_train']}) | twin-in-test AUROC={row['auroc_twin_in_test']:.4f} "
              f"(n={row['n_test']})  [{time.time() - t0:.0f}s]")
    pd.DataFrame(per_fold).to_csv(OUT / "per_fold_exposure.csv", index=False)

    # ---- 3. pooled paired comparison on identical images
    yk = m.ham_label_melanoma.values
    p_unseen = np.array([probs[i, list(status[i]).index("test")] for i in range(len(m))])
    seen_mask = status == "train"
    p_seen = np.array([probs[i, seen_mask[i]].mean() for i in range(len(m))])
    keep = seen_mask.any(axis=1)
    rng = np.random.default_rng(SEED)
    lo, hi, p = paired_bootstrap(yk[keep], p_seen[keep], p_unseen[keep],
                                 m.ham_lesion.values[keep], rng)
    res = {
        "n_images": int(keep.sum()), "n_melanoma": int(yk[keep].sum()),
        "n_lesions": int(m.ham_lesion[keep].nunique()),
        "auroc_twin_seen_in_training": auc_or_nan(yk[keep], p_seen[keep]),
        "auroc_twin_never_seen": auc_or_nan(yk[keep], p_unseen[keep]),
    }
    res["delta"] = res["auroc_twin_seen_in_training"] - res["auroc_twin_never_seen"]
    res.update({"ci_lo": lo, "ci_hi": hi, "p_bootstrap": p})
    pd.DataFrame([res]).to_csv(OUT / "paired_seen_unseen.csv", index=False)

    scores = m[["kaggle_path", "kaggle_split", "kaggle_label_malignant",
                "ham_index", "ham_lesion", "ham_label_melanoma", "hamming_distance"]].copy()
    scores["p_twin_never_seen"] = p_unseen
    scores["p_twin_seen_mean"] = p_seen
    for f in range(N_FOLDS):
        scores[f"status_fold{f + 1}"] = status[:, f]
        scores[f"p_fold{f + 1}"] = probs[:, f]
    scores.to_csv(OUT / "scores_per_image.csv", index=False)

    print("\n=== Paired comparison on identical Kaggle images (HAM melanoma labels) ===")
    print(f"  images={res['n_images']}  melanoma={res['n_melanoma']}  lesions={res['n_lesions']}")
    print(f"  AUROC, twin seen in training : {res['auroc_twin_seen_in_training']:.4f}")
    print(f"  AUROC, twin never seen       : {res['auroc_twin_never_seen']:.4f}")
    print(f"  delta = {res['delta']:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  p = {p:.4f}")

    # secondary: Kaggle's own labels, fold-ensemble, matched images only
    ens = probs.mean(axis=1)
    kl = m.kaggle_label_malignant.values
    print(f"\n  Secondary (Kaggle malignant label, 5-fold ensemble, matched images): "
          f"AUROC={auc_or_nan(kl, ens):.4f}  [label is any malignancy, not melanoma]")
    print(f"\nAll outputs in {OUT}")


if __name__ == "__main__":
    main()
