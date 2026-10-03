"""
v3 pipeline: EfficientNet-B0 at 320px, lesion-grouped 5-fold CV on HAM10000,
SHAP-guided feature distillation, fold-ensembled external validation on ISIC 2020.

Changes from v2 and why:
  - 320px input (from the 384px cache) instead of 224. Fine dermoscopic
    structures are the main melanoma cue and were being destroyed at 224.
  - B0 is kept rather than moving to B3: the paper's efficiency claim depends
    on the small backbone, and B3 at 384 is ~6-8x the compute.
  - FOLD ENSEMBLING:
      * HAM10000 -> out-of-fold (OOF) aggregate over all 10,015 images.
        Each image is predicted exactly once, by the fold that did not train
        on it. This is the correct way to ensemble when test sets differ.
      * ISIC 2020 -> probabilities averaged across all 5 fold pipelines.
  - EXTERNAL REPORTING FIX: a Youden threshold picked at 11% prevalence does
    not transfer to 1.78% prevalence. External results are therefore reported
    threshold-free (AUROC/AUPRC) plus sensitivity at FIXED specificity
    (0.80/0.90/0.95). The transferred-threshold row is still emitted for
    completeness, clearly labelled.
  - k-sweep trimmed to [256, 512, None]: v2 showed 64/128 were strictly worse,
    so those cost runtime without informing the ablation.

Run: caffeinate -dimsu python3 run_pipeline_v3.py 2>&1 | tee pipeline_v3_log.txt
Resumable: completed folds are skipped; ensembling reads the saved per-fold
probability files, so it can be re-run alone once all folds exist.
"""

import json
import time
import warnings
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_auc_score,
                             average_precision_score, confusion_matrix, recall_score,
                             f1_score, roc_curve)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# MELANOMA_ROOT if set, otherwise the data/ folder of this repository
ROOT = Path(os.environ.get("MELANOMA_ROOT", Path(__file__).resolve().parents[1] / "data")).expanduser()
HAM_MANIFEST = ROOT / "manifest_ham10000_384.csv"
ISIC_MANIFEST = ROOT / "manifest_isic2020_384.csv"
OUT_DIR = ROOT / "results_v3"
PROB_DIR = OUT_DIR / "probs"
OUT_DIR.mkdir(exist_ok=True)
PROB_DIR.mkdir(exist_ok=True)

IMG_SIZE = 320
BATCH_SIZE = 16            # smaller: 320px activations are ~2x the memory of 224
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
TTA_VIEWS = 2              # original + horizontal flip
SPEC_POINTS = [0.80, 0.90, 0.95]

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

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


class ManifestDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = list(paths), list(labels), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB")), self.labels[i]


def build_model():
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, 2))
    return m.to(DEVICE)


def build_optimizer(model):
    n = len(model.features)
    split = n // 2
    early = [p for b in list(model.features)[:split] for p in b.parameters()]
    late = [p for b in list(model.features)[split:] for p in b.parameters()]
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

    best_auc, best_state, patience = -1.0, None, 0
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
              f"({time.time()-t0:.0f}s)")

        if va_auc > best_auc:
            best_auc, patience = va_auc, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"  [Fold {fold}] Early stop @ {epoch} (best={best_auc:.4f})")
                break
    model.load_state_dict(best_state)
    return model


def multistage_features(model, x):
    feats, h = [], x
    for idx, block in enumerate(model.features):
        h = block(h)
        if idx in STAGE_INDICES:
            feats.append(nn.functional.adaptive_avg_pool2d(h, 1).flatten(1))
    return torch.cat(feats, dim=1)


def extract_features(model, paths, labels, batch_size=32, tag=""):
    loader = DataLoader(ManifestDataset(paths, labels, eval_tf),
                        batch_size=batch_size, shuffle=False, num_workers=0)
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
                print(f"    [{tag}] {bi*batch_size}/{len(paths)}")
    return np.concatenate(all_f), np.concatenate(all_y)


def shap_feature_ranking(X, y):
    sur = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                        subsample=0.8, colsample_bytree=0.8,
                        scale_pos_weight=float((y == 0).sum() / max((y == 1).sum(), 1)),
                        eval_metric="logloss", random_state=SEED, n_jobs=-1)
    sur.fit(X, y)
    try:
        import shap
        n = min(SHAP_SUBSAMPLE, X.shape[0])
        idx = np.random.RandomState(SEED).choice(X.shape[0], n, replace=False)
        vals = shap.TreeExplainer(sur).shap_values(X[idx])
        if isinstance(vals, list):
            vals = vals[1]
        imp, method = np.abs(vals).mean(axis=0), "shap"
    except Exception as e:
        print(f"    SHAP failed ({e}); using XGBoost gain importance.")
        imp, method = sur.feature_importances_, "xgb_gain"
    return np.argsort(imp)[::-1], imp, method


def make_classifiers(y):
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    return {
        "SVM-RBF": SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                       class_weight="balanced", random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 scale_pos_weight=spw, eval_metric="logloss",
                                 random_state=SEED, n_jobs=-1),
        "RandomForest": RandomForestClassifier(n_estimators=300,
                                               class_weight="balanced",
                                               random_state=SEED, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=11, weights="distance"),
    }


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def sens_at_spec(y, p, target_spec):
    """Sensitivity at a fixed specificity -- the prevalence-robust way to
    report an operating point on an external set."""
    fpr, tpr, _ = roc_curve(y, p)
    spec = 1 - fpr
    ok = spec >= target_spec
    return float(tpr[ok].max()) if ok.any() else float("nan")


def metrics_row(y, p, thr, **meta):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    row = dict(meta)
    row.update({
        "threshold": thr,
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "sensitivity": recall_score(y, pred, pos_label=1, zero_division=0),
        "specificity": recall_score(y, pred, pos_label=0, zero_division=0),
        "f1_melanoma": f1_score(y, pred, pos_label=1, zero_division=0),
        "auroc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
        "prevalence": float(np.mean(y)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_samples": int(len(y)),
    })
    for s in SPEC_POINTS:
        row[f"sens_at_spec{int(s*100)}"] = sens_at_spec(y, p, s)
    return row


def main():
    print(f"[Device] {DEVICE}  [Input] {IMG_SIZE}px")
    ham = pd.read_csv(HAM_MANIFEST)
    isic = pd.read_csv(ISIC_MANIFEST)
    print(f"HAM10000: {len(ham)} imgs, {int(ham['label'].sum())} melanoma, "
          f"{ham['lesion_group'].nunique()} lesions")
    print(f"ISIC2020: {len(isic)} imgs, {int(isic['label'].sum())} melanoma "
          f"({100*isic['label'].mean():.2f}% prevalence)")

    paths, y_all, groups = (ham["image_path"].values, ham["label"].values,
                            ham["lesion_group"].values)
    isic_paths, isic_y = isic["image_path"].values, isic["label"].values
    per_fold_csv = OUT_DIR / "per_fold_results.csv"

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, te_idx) in enumerate(sgkf.split(paths, y_all, groups), 1):
        marker = OUT_DIR / f"fold{fold}.done"
        if marker.exists():
            print(f"\n=== FOLD {fold}/{N_FOLDS} === [skipped]")
            continue
        print(f"\n=== FOLD {fold}/{N_FOLDS} ===")

        gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED)
        i_tr, i_va = next(gss.split(paths[tr_idx], y_all[tr_idx], groups[tr_idx]))
        tr_i, va_i = tr_idx[i_tr], tr_idx[i_va]
        assert not (set(groups[tr_i]) & set(groups[te_idx])), "lesion leak train/test"
        assert not (set(groups[va_i]) & set(groups[te_idx])), "lesion leak val/test"
        print(f"  train={len(tr_i)} val={len(va_i)} test={len(te_idx)} "
              f"(melanoma in test={int(y_all[te_idx].sum())})")

        model = train_fold(paths[tr_i], y_all[tr_i], paths[va_i], y_all[va_i], fold)
        torch.save(model.state_dict(), OUT_DIR / f"model_fold{fold}.pt")

        print("  Extracting features...")
        Xtr, ytr = extract_features(model, paths[tr_i], y_all[tr_i], tag="train")
        Xva, yva = extract_features(model, paths[va_i], y_all[va_i], tag="val")
        Xte, yte = extract_features(model, paths[te_idx], y_all[te_idx], tag="test")
        Xex, yex = extract_features(model, isic_paths, isic_y, tag="isic")
        print(f"  Feature dim: {Xtr.shape[1]}")

        sc = StandardScaler().fit(Xtr)
        Xtr, Xva, Xte, Xex = (sc.transform(Xtr), sc.transform(Xva),
                              sc.transform(Xte), sc.transform(Xex))

        print("  SHAP ranking...")
        order, imp, method = shap_feature_ranking(Xtr, ytr)
        np.save(OUT_DIR / f"importance_fold{fold}.npy", imp)
        print(f"    method={method}")

        rows, store = [], {}
        for k in K_SWEEP:
            sel = order if k is None else order[:k]
            ktag = "all" if k is None else str(k)
            print(f"  --- k={ktag} ---")
            Atr, Ava, Ate, Aex = Xtr[:, sel], Xva[:, sel], Xte[:, sel], Xex[:, sel]

            clfs = make_classifiers(ytr)
            clfs["Stacking"] = StackingClassifier(
                estimators=[(n, c) for n, c in clfs.items()],
                final_estimator=LogisticRegression(max_iter=2000,
                                                   class_weight="balanced"),
                cv=3, n_jobs=-1)

            for name, clf in clfs.items():
                clf.fit(Atr, ytr)
                thr = youden_threshold(yva, clf.predict_proba(Ava)[:, 1])
                p_te = clf.predict_proba(Ate)[:, 1]
                p_ex = clf.predict_proba(Aex)[:, 1]

                rows.append(metrics_row(yte, p_te, thr, fold=fold,
                                        dataset="HAM10000_test",
                                        classifier=name, k_features=ktag))
                rows.append(metrics_row(yex, p_ex, thr, fold=fold,
                                        dataset="ISIC2020_external",
                                        classifier=name, k_features=ktag))
                store[f"te__{ktag}__{name}"] = p_te
                store[f"ex__{ktag}__{name}"] = p_ex
                print(f"    {name}: HAM auroc={rows[-2]['auroc']:.4f} | "
                      f"ISIC auroc={rows[-1]['auroc']:.4f} "
                      f"sens@spec90={rows[-1]['sens_at_spec90']:.4f}")

        np.savez_compressed(PROB_DIR / f"fold{fold}.npz",
                            test_index=te_idx, test_y=yte, isic_y=yex, **store)
        pd.DataFrame(rows).to_csv(per_fold_csv, mode="a",
                                  header=not per_fold_csv.exists(), index=False)
        marker.touch()
        print(f"  [Checkpoint] fold {fold} saved.")

    # ---------------- FOLD ENSEMBLING ----------------
    fold_files = sorted(PROB_DIR.glob("fold*.npz"))
    if len(fold_files) < N_FOLDS:
        print(f"\nOnly {len(fold_files)}/{N_FOLDS} folds present -- "
              f"rerun to finish, then ensembling will run automatically.")
        return

    print("\n=== FOLD ENSEMBLING ===")
    data = {f.stem: np.load(f, allow_pickle=True) for f in fold_files}
    ens_rows = []

    for k in K_SWEEP:
        ktag = "all" if k is None else str(k)
        for name in ["SVM-RBF", "XGBoost", "RandomForest", "KNN", "Stacking"]:
            # HAM10000: out-of-fold aggregate (each image predicted once, by the
            # fold that never trained on it)
            oof_p = np.full(len(y_all), np.nan)
            for fkey, d in data.items():
                oof_p[d["test_index"]] = d[f"te__{ktag}__{name}"]
            mask = ~np.isnan(oof_p)
            oof_thr = youden_threshold(y_all[mask], oof_p[mask])
            ens_rows.append(metrics_row(y_all[mask], oof_p[mask], oof_thr,
                                        fold="OOF_ensemble",
                                        dataset="HAM10000_OOF",
                                        classifier=name, k_features=ktag))

            # ISIC 2020: average probabilities across the 5 fold pipelines
            ex_p = np.mean([d[f"ex__{ktag}__{name}"] for d in data.values()], axis=0)
            ex_y = list(data.values())[0]["isic_y"]
            ens_rows.append(metrics_row(ex_y, ex_p, youden_threshold(ex_y, ex_p),
                                        fold="fold_ensemble",
                                        dataset="ISIC2020_external_ENSEMBLE",
                                        classifier=name, k_features=ktag))
            print(f"  k={ktag} {name}: "
                  f"HAM-OOF auroc={ens_rows[-2]['auroc']:.4f} | "
                  f"ISIC-ens auroc={ens_rows[-1]['auroc']:.4f} "
                  f"auprc={ens_rows[-1]['auprc']:.4f} "
                  f"sens@spec90={ens_rows[-1]['sens_at_spec90']:.4f}")

    pd.DataFrame(ens_rows).to_csv(OUT_DIR / "ensemble_results.csv", index=False)

    res = pd.read_csv(per_fold_csv)
    res.groupby(["dataset", "k_features", "classifier"])[
        ["auroc", "auprc", "balanced_accuracy", "sensitivity",
         "specificity", "accuracy"]].agg(["mean", "std"]).to_csv(
        OUT_DIR / "summary_v3.csv")

    print(f"\nSaved: {OUT_DIR}/summary_v3.csv and ensemble_results.csv")
    print("NOTE: the ISIC ensemble threshold is fitted on ISIC itself and is "
          "therefore optimistic -- quote AUROC/AUPRC and sens@fixed-spec as the "
          "external headline numbers, not that row's sensitivity/specificity.")


if __name__ == "__main__":
    main()
