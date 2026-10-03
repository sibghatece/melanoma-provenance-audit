"""
Leakage-magnitude experiment for the melanoma provenance paper.

Runs the v3 pipeline UNCHANGED (EfficientNet-B0 @ 320px, same hyperparameters,
same seed, same stages, same SHAP distillation, same classifiers) under a
chosen partitioning scheme, so the only difference from results_v3 is how
HAM10000 is split.

  --split image   ordinary StratifiedKFold on images (what most papers do);
                  inner validation is an image-wise StratifiedShuffleSplit.
  --split lesion  StratifiedGroupKFold on lesion_group (identical to v3);
                  only needed if you want a fresh reference run.

What it adds over v3:
  * Leakage accounting per fold: how many test images have another image of
    the same lesion in the training partition ("leaked" images).
  * OOF AUROC split into leaked vs clean test images.
  * All operating-point metrics (SEN/SPE/ACC/BAC) use the threshold fitted on
    each fold's inner validation split, never on evaluation data. Threshold-
    free metrics (AUROC, AUPRC, sens@fixed-spec) are reported as before.
  * ISIC 2020 external evaluation, so internal inflation can be contrasted
    with external performance.

Setup (same layout as the repository):
  export MELANOMA_ROOT=/path/to/data
Run:
  caffeinate -dimsu python3 run_leakage_magnitude.py --split image 2>&1 | tee leakage_magnitude_log.txt
Resumable: completed folds are skipped.

Outputs in $MELANOMA_ROOT/results_split_<mode>/:
  per_fold_results.csv, leakage_accounting.csv, oof_results.csv,
  isic_ensemble_results.csv, probs/fold*.npz, model_fold*.pt
"""

import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold,
                                     StratifiedKFold, StratifiedShuffleSplit)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- paths
# MELANOMA_ROOT if set, otherwise the data/ folder of this repository
ROOT = Path(os.environ.get("MELANOMA_ROOT", Path(__file__).resolve().parents[1] / "data")).expanduser()
HAM_MANIFEST = ROOT / "manifest_ham10000_384.csv"
ISIC_MANIFEST = ROOT / "manifest_isic2020_384.csv"

# ---------------------------------------------------------------- v3 settings (unchanged)
IMG_SIZE = 320
BATCH_SIZE = 16
N_FOLDS = 5
EPOCHS = 30
PATIENCE = 6
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.05
SEED = 42
VAL_FRACTION = 0.15
LR_BACKBONE_EARLY = 1e-5
LR_BACKBONE_LATE = 5e-5
LR_HEAD = 5e-4
STAGE_INDICES = [5, 6, 7, 8]
K_SWEEP = [256, 512, None]
SHAP_SUBSAMPLE = 2000
TTA_VIEWS = 2
SPEC_POINTS = [0.80, 0.90, 0.95]
CLASSIFIERS = ["SVM-RBF", "XGBoost", "RandomForest", "KNN", "Stacking"]

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


# ---------------------------------------------------------------- data / model
class ManifestDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = list(paths), list(labels), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def build_model():
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, 2))
    return m.to(DEVICE)


def build_optimizer(model):
    blocks = list(model.features)
    split = len(blocks) // 2
    early = [p for b in blocks[:split] for p in b.parameters()]
    late = [p for b in blocks[split:] for p in b.parameters()]
    return torch.optim.AdamW([
        {"params": early, "lr": LR_BACKBONE_EARLY},
        {"params": late, "lr": LR_BACKBONE_LATE},
        {"params": list(model.classifier.parameters()), "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)


def train_fold(tr_paths, tr_y, va_paths, va_y, fold):
    tr_loader = DataLoader(ManifestDataset(tr_paths, tr_y, train_tf),
                           batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    va_loader = DataLoader(ManifestDataset(va_paths, va_y, eval_tf),
                           batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model = build_model()
    n_pos = int(np.sum(tr_y))
    w = torch.tensor([1.0, (len(tr_y) - n_pos) / max(n_pos, 1)],
                     dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=LABEL_SMOOTH)
    optimizer = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_auc, best_state, patience, best_epoch = -1.0, None, 0, 0
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        for x, y in tr_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        probs, ys = [], []
        with torch.no_grad():
            for x, y in va_loader:
                probs.append(torch.softmax(model(x.to(DEVICE)), 1)[:, 1].cpu().numpy())
                ys.append(y.numpy())
        va_auc = roc_auc_score(np.concatenate(ys), np.concatenate(probs))
        print(f"  [Fold {fold}] Epoch {epoch}/{EPOCHS}  val_auc={va_auc:.4f}  "
              f"({time.time() - t0:.0f}s)")
        if va_auc > best_auc:
            best_auc, patience, best_epoch = va_auc, 0, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"  [Fold {fold}] Early stop @ {epoch} (best={best_auc:.4f})")
                break
    model.load_state_dict(best_state)
    return model, best_auc, best_epoch


def multistage_features(model, x):
    feats, h = [], x
    for idx, block in enumerate(model.features):
        h = block(h)
        if idx in STAGE_INDICES:
            feats.append(nn.functional.adaptive_avg_pool2d(h, 1).flatten(1))
    return torch.cat(feats, dim=1)


def extract_features(model, paths, labels, tag=""):
    loader = DataLoader(ManifestDataset(paths, labels, eval_tf),
                        batch_size=32, shuffle=False, num_workers=0)
    model.eval()
    all_f, all_y = [], []
    with torch.no_grad():
        for bi, (x, y) in enumerate(loader):
            x = x.to(DEVICE)
            views = [x] + ([torch.flip(x, [3])] if TTA_VIEWS > 1 else [])
            f = torch.stack([multistage_features(model, v) for v in views]).mean(0)
            all_f.append(f.cpu().numpy())
            all_y.append(y.numpy())
            if tag and bi % 200 == 0:
                print(f"    [{tag}] {bi * 32}/{len(paths)}")
    return np.concatenate(all_f), np.concatenate(all_y)


def shap_feature_ranking(X, y):
    sur = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                        subsample=0.8, colsample_bytree=0.8,
                        scale_pos_weight=float((y == 0).sum() / max((y == 1).sum(), 1)),
                        eval_metric="logloss", random_state=SEED, n_jobs=-1)
    sur.fit(X, y)
    import shap
    n = min(SHAP_SUBSAMPLE, X.shape[0])
    idx = np.random.RandomState(SEED).choice(X.shape[0], n, replace=False)
    vals = shap.TreeExplainer(sur).shap_values(X[idx])
    if isinstance(vals, list):
        vals = vals[1]
    imp = np.abs(vals).mean(axis=0)
    return np.argsort(imp)[::-1], imp


def base_learners(y):
    """The four v3 base learners with identical settings."""
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    return {
        "SVM-RBF": SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                       class_weight="balanced", random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 scale_pos_weight=spw, eval_metric="logloss",
                                 random_state=SEED, n_jobs=-1),
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                               random_state=SEED, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=11, weights="distance"),
    }


def make_classifiers(y):
    """Base learners plus the stacking ensemble (sklearn clones the base
    estimators internally, exactly as in v3)."""
    clfs = base_learners(y)
    clfs["Stacking"] = StackingClassifier(
        estimators=list(base_learners(y).items()),
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
        cv=3, n_jobs=-1)
    return clfs


# ---------------------------------------------------------------- metrics
def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def sens_at_spec(y, p, target_spec):
    fpr, tpr, _ = roc_curve(y, p)
    ok = (1 - fpr) >= target_spec
    return float(tpr[ok].max()) if ok.any() else float("nan")


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def metrics_row(y, p, thr, **meta):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    row = dict(meta)
    row.update({
        "threshold": thr,
        "threshold_source": "inner_validation",
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "sensitivity": recall_score(y, pred, pos_label=1, zero_division=0),
        "specificity": recall_score(y, pred, pos_label=0, zero_division=0),
        "f1_melanoma": f1_score(y, pred, pos_label=1, zero_division=0),
        "auroc": safe_auc(y, p),
        "auprc": average_precision_score(y, p),
        "prevalence": float(np.mean(y)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_samples": int(len(y)),
    })
    for s in SPEC_POINTS:
        row[f"sens_at_spec{int(s * 100)}"] = sens_at_spec(y, p, s)
    return row


# ---------------------------------------------------------------- splitting
def make_outer_splits(split, paths, y, groups):
    if split == "lesion":
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        return list(cv.split(paths, y, groups))
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(cv.split(paths, y))


def make_inner_split(split, tr_idx, y, groups):
    if split == "lesion":
        gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED)
        i_tr, i_va = next(gss.split(tr_idx, y[tr_idx], groups[tr_idx]))
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED)
        i_tr, i_va = next(sss.split(tr_idx, y[tr_idx]))
    return tr_idx[i_tr], tr_idx[i_va]


def leakage_accounting(fold, tr_i, va_i, te_idx, groups, y):
    train_lesions = set(groups[tr_i])
    val_lesions = set(groups[va_i])
    leaked = np.array([g in train_lesions for g in groups[te_idx]])
    leaked_val = np.array([g in val_lesions for g in groups[te_idx]])
    return leaked, {
        "fold": fold,
        "n_test": int(len(te_idx)),
        "n_test_melanoma": int(y[te_idx].sum()),
        "test_images_with_lesion_in_train": int(leaked.sum()),
        "pct_test_leaked": 100.0 * leaked.mean(),
        "melanoma_test_images_leaked": int((leaked & (y[te_idx] == 1)).sum()),
        "test_images_with_lesion_in_inner_val": int(leaked_val.sum()),
        "shared_lesions_train_test": int(len(train_lesions & set(groups[te_idx]))),
        "shared_lesions_train_val": int(len(train_lesions & val_lesions)),
    }


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["image", "lesion"], required=True)
    args = ap.parse_args()
    split = args.split

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    out_dir = ROOT / f"results_split_{split}"
    prob_dir = out_dir / "probs"
    prob_dir.mkdir(parents=True, exist_ok=True)
    per_fold_csv = out_dir / "per_fold_results.csv"
    leak_csv = out_dir / "leakage_accounting.csv"

    print(f"[Device] {DEVICE}  [Input] {IMG_SIZE}px  [Split] {split}")
    print(f"[Root] {ROOT}")
    ham = pd.read_csv(HAM_MANIFEST)
    isic = pd.read_csv(ISIC_MANIFEST)
    paths, y_all, groups = (ham["image_path"].values, ham["label"].values.astype(int),
                            ham["lesion_group"].values)
    isic_paths, isic_y = isic["image_path"].values, isic["label"].values.astype(int)
    print(f"HAM10000: {len(ham)} imgs, {y_all.sum()} melanoma, "
          f"{len(np.unique(groups))} lesions")
    print(f"ISIC2020: {len(isic)} imgs, {isic_y.sum()} melanoma")

    splits = make_outer_splits(split, paths, y_all, groups)
    for fold, (tr_idx, te_idx) in enumerate(splits, 1):
        marker = out_dir / f"fold{fold}.done"
        if marker.exists():
            print(f"\n=== FOLD {fold}/{N_FOLDS} === [skipped]")
            continue
        print(f"\n=== FOLD {fold}/{N_FOLDS} ({split}-wise) ===")

        tr_i, va_i = make_inner_split(split, tr_idx, y_all, groups)
        if split == "lesion":
            assert not (set(groups[tr_i]) & set(groups[te_idx])), "lesion leak train/test"
            assert not (set(groups[va_i]) & set(groups[te_idx])), "lesion leak val/test"

        leaked, leak_row = leakage_accounting(fold, tr_i, va_i, te_idx, groups, y_all)
        print(f"  train={len(tr_i)} val={len(va_i)} test={len(te_idx)} | "
              f"test images whose lesion is in train: "
              f"{leak_row['test_images_with_lesion_in_train']} "
              f"({leak_row['pct_test_leaked']:.1f}%)")

        model, best_val_auc, best_epoch = train_fold(
            paths[tr_i], y_all[tr_i], paths[va_i], y_all[va_i], fold)
        leak_row.update({"best_val_auroc": best_val_auc, "best_epoch": best_epoch})
        torch.save(model.state_dict(), out_dir / f"model_fold{fold}.pt")

        print("  Extracting features...")
        Xtr, ytr = extract_features(model, paths[tr_i], y_all[tr_i], tag="train")
        Xva, yva = extract_features(model, paths[va_i], y_all[va_i], tag="val")
        Xte, yte = extract_features(model, paths[te_idx], y_all[te_idx], tag="test")
        Xex, yex = extract_features(model, isic_paths, isic_y, tag="isic")

        sc = StandardScaler().fit(Xtr)
        Xtr, Xva, Xte, Xex = (sc.transform(Xtr), sc.transform(Xva),
                              sc.transform(Xte), sc.transform(Xex))

        print("  SHAP ranking...")
        order, imp = shap_feature_ranking(Xtr, ytr)
        np.save(out_dir / f"importance_fold{fold}.npy", imp)

        rows, store, thresholds = [], {}, {}
        for k in K_SWEEP:
            sel = order if k is None else order[:k]
            ktag = "all" if k is None else str(k)
            print(f"  --- k={ktag} ---")
            Atr, Ava, Ate, Aex = Xtr[:, sel], Xva[:, sel], Xte[:, sel], Xex[:, sel]
            for name, clf in make_classifiers(ytr).items():
                clf.fit(Atr, ytr)
                thr = youden_threshold(yva, clf.predict_proba(Ava)[:, 1])
                p_te = clf.predict_proba(Ate)[:, 1]
                p_ex = clf.predict_proba(Aex)[:, 1]
                meta = dict(split=split, fold=fold, classifier=name, k_features=ktag)
                rows.append(metrics_row(yte, p_te, thr, dataset="HAM10000_test", **meta))
                rows.append(metrics_row(yex, p_ex, thr, dataset="ISIC2020_external", **meta))
                if leaked.any() and (~leaked).any():
                    rows[-2]["auroc_leaked_subset"] = safe_auc(yte[leaked], p_te[leaked])
                    rows[-2]["auroc_clean_subset"] = safe_auc(yte[~leaked], p_te[~leaked])
                store[f"te__{ktag}__{name}"] = p_te
                store[f"ex__{ktag}__{name}"] = p_ex
                thresholds[f"thr__{ktag}__{name}"] = np.array(thr)
                print(f"    {name}: HAM auroc={rows[-2]['auroc']:.4f} | "
                      f"ISIC auroc={rows[-1]['auroc']:.4f}")

        np.savez_compressed(prob_dir / f"fold{fold}.npz", test_index=te_idx,
                            test_y=yte, isic_y=yex, leaked=leaked,
                            **store, **thresholds)
        pd.DataFrame(rows).to_csv(per_fold_csv, mode="a",
                                  header=not per_fold_csv.exists(), index=False)
        pd.DataFrame([leak_row]).to_csv(leak_csv, mode="a",
                                        header=not leak_csv.exists(), index=False)
        marker.touch()
        print(f"  [Checkpoint] fold {fold} saved.")

    aggregate(out_dir, prob_dir, split, y_all)


def aggregate(out_dir, prob_dir, split, y_all):
    fold_files = sorted(prob_dir.glob("fold*.npz"))
    if len(fold_files) < N_FOLDS:
        print(f"\nOnly {len(fold_files)}/{N_FOLDS} folds present; rerun to finish.")
        return
    print("\n=== AGGREGATION ===")
    data = [np.load(f, allow_pickle=True) for f in fold_files]
    oof_rows, ens_rows = [], []

    for k in K_SWEEP:
        ktag = "all" if k is None else str(k)
        for name in CLASSIFIERS:
            key = f"{ktag}__{name}"
            # HAM10000 OOF: each fold's predictions binarised with that fold's
            # inner-validation threshold, then pooled.
            oof_p = np.full(len(y_all), np.nan)
            oof_pred = np.full(len(y_all), -1)
            oof_leak = np.zeros(len(y_all), dtype=bool)
            for d in data:
                idx = d["test_index"]
                oof_p[idx] = d[f"te__{key}"]
                oof_pred[idx] = (d[f"te__{key}"] >= float(d[f"thr__{key}"])).astype(int)
                oof_leak[idx] = d["leaked"]
            yy, pp, pr = y_all, oof_p, oof_pred
            tn, fp, fn, tp = confusion_matrix(yy, pr, labels=[0, 1]).ravel()
            row = {
                "split": split, "classifier": name, "k_features": ktag,
                "dataset": "HAM10000_OOF", "threshold_source": "per-fold inner validation",
                "auroc": safe_auc(yy, pp), "auprc": average_precision_score(yy, pp),
                "balanced_accuracy": balanced_accuracy_score(yy, pr),
                "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
                "accuracy": accuracy_score(yy, pr),
                "n_leaked_images": int(oof_leak.sum()),
                "pct_leaked_images": 100.0 * oof_leak.mean(),
            }
            for s in SPEC_POINTS:
                row[f"sens_at_spec{int(s * 100)}"] = sens_at_spec(yy, pp, s)
            if oof_leak.any() and (~oof_leak).any():
                row["auroc_leaked_subset"] = safe_auc(yy[oof_leak], pp[oof_leak])
                row["auroc_clean_subset"] = safe_auc(yy[~oof_leak], pp[~oof_leak])
            oof_rows.append(row)

            # ISIC 2020: probabilities averaged over folds; threshold-free
            # metrics only (no threshold is fitted on external data).
            ex_p = np.mean([d[f"ex__{key}"] for d in data], axis=0)
            ex_y = data[0]["isic_y"]
            erow = {"split": split, "classifier": name, "k_features": ktag,
                    "dataset": "ISIC2020_external_ENSEMBLE",
                    "auroc": safe_auc(ex_y, ex_p),
                    "auprc": average_precision_score(ex_y, ex_p)}
            for s in SPEC_POINTS:
                erow[f"sens_at_spec{int(s * 100)}"] = sens_at_spec(ex_y, ex_p, s)
            ens_rows.append(erow)
            print(f"  k={ktag} {name}: HAM-OOF auroc={row['auroc']:.4f} | "
                  f"ISIC-ens auroc={erow['auroc']:.4f}")

    pd.DataFrame(oof_rows).to_csv(out_dir / "oof_results.csv", index=False)
    pd.DataFrame(ens_rows).to_csv(out_dir / "isic_ensemble_results.csv", index=False)
    print(f"\nSaved: {out_dir}/oof_results.csv, isic_ensemble_results.csv, "
          f"leakage_accounting.csv")


if __name__ == "__main__":
    main()
