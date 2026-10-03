"""
Recompute the HAM10000 out-of-fold operating-point metrics of the v3 run
using each fold's INNER-VALIDATION threshold, instead of a threshold fitted
on the pooled out-of-fold predictions.

Why: run_pipeline_v3.py fitted the OOF Youden threshold on the OOF
predictions themselves, so SEN/SPE/ACC/balanced accuracy in Table 4 were
evaluation-fitted and optimistic. AUROC, AUPRC and sensitivity at fixed
specificity are threshold-free and do not change.

No retraining. Reads only:
  $MELANOMA_ROOT/results_v3/per_fold_results.csv   (holds the inner-val thresholds)
  $MELANOMA_ROOT/results_v3/probs/fold1..5.npz      (holds the OOF probabilities)
  $MELANOMA_ROOT/manifest_ham10000_384.csv

For ISIC 2020 it also writes per-fold operating-point metrics (mean and SD
over folds, each fold using its own inner-val threshold), which were already
computed correctly in per_fold_results.csv and are summarised here.

Run:
  export MELANOMA_ROOT=/path/to/data
  python3 recompute_v3_thresholds.py
Output:
  $MELANOMA_ROOT/results_v3/table4_corrected_thresholds.csv
  $MELANOMA_ROOT/results_v3/table5_isic_perfold_operating_points.csv
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             roc_auc_score, roc_curve)

# MELANOMA_ROOT if set, otherwise the data/ folder of this repository
ROOT = Path(os.environ.get("MELANOMA_ROOT", Path(__file__).resolve().parents[1] / "data")).expanduser()
V3 = ROOT / "results_v3"
CLASSIFIERS = ["SVM-RBF", "XGBoost", "RandomForest", "KNN", "Stacking"]
KTAGS = ["256", "512", "all"]
SPEC_POINTS = [0.80, 0.90, 0.95]


def sens_at_spec(y, p, target):
    fpr, tpr, _ = roc_curve(y, p)
    ok = (1 - fpr) >= target
    return float(tpr[ok].max()) if ok.any() else float("nan")


def youden(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def main():
    ham = pd.read_csv(ROOT / "manifest_ham10000_384.csv")
    y_all = ham["label"].values.astype(int)
    per_fold = pd.read_csv(V3 / "per_fold_results.csv", dtype={"k_features": str})
    fold_files = sorted((V3 / "probs").glob("fold*.npz"))
    if len(fold_files) != 5:
        raise SystemExit(f"Expected 5 fold files in {V3/'probs'}, found {len(fold_files)}")
    data = {int(f.stem.replace("fold", "")): np.load(f, allow_pickle=True)
            for f in fold_files}

    ham_rows = per_fold[per_fold["dataset"] == "HAM10000_test"]
    out = []
    for ktag in KTAGS:
        for name in CLASSIFIERS:
            p = np.full(len(y_all), np.nan)
            pred = np.full(len(y_all), -1)
            for fold, d in data.items():
                sel = ham_rows[(ham_rows["fold"] == fold) &
                               (ham_rows["classifier"] == name) &
                               (ham_rows["k_features"] == ktag)]
                if len(sel) != 1:
                    raise SystemExit(f"Threshold not found: fold {fold} {name} k={ktag}")
                thr = float(sel["threshold"].iloc[0])
                idx = d["test_index"]
                probs = d[f"te__{ktag}__{name}"]
                p[idx] = probs
                pred[idx] = (probs >= thr).astype(int)
            assert not np.isnan(p).any() and (pred >= 0).all(), "OOF coverage incomplete"

            tn, fp, fn, tp = confusion_matrix(y_all, pred, labels=[0, 1]).ravel()
            # old, evaluation-fitted numbers for side-by-side comparison
            old_thr = youden(y_all, p)
            old_pred = (p >= old_thr).astype(int)
            row = {
                "classifier": name, "k": ktag,
                "AUROC": roc_auc_score(y_all, p),
                "AUPRC": average_precision_score(y_all, p),
                "BAC": balanced_accuracy_score(y_all, pred),
                "SEN": tp / (tp + fn), "SPE": tn / (tn + fp),
                "ACC": accuracy_score(y_all, pred),
                "TP": int(tp), "FN": int(fn), "TN": int(tn), "FP": int(fp),
                "BAC_old_evalfitted": balanced_accuracy_score(y_all, old_pred),
                "ACC_old_evalfitted": accuracy_score(y_all, old_pred),
            }
            for s in SPEC_POINTS:
                row[f"Sens@Sp{int(s*100)}"] = sens_at_spec(y_all, p, s)
            out.append(row)
            print(f"k={ktag:>3} {name:>12}: AUROC={row['AUROC']:.4f}  "
                  f"BAC {row['BAC_old_evalfitted']:.4f} -> {row['BAC']:.4f}  "
                  f"SEN={row['SEN']:.4f} SPE={row['SPE']:.4f}")

    t4 = pd.DataFrame(out).sort_values("AUROC", ascending=False)
    t4.to_csv(V3 / "table4_corrected_thresholds.csv", index=False)

    isic = per_fold[per_fold["dataset"] == "ISIC2020_external"]
    cols = ["balanced_accuracy", "sensitivity", "specificity", "accuracy"]
    t5 = isic.groupby(["classifier", "k_features"])[cols].agg(["mean", "std"])
    t5.to_csv(V3 / "table5_isic_perfold_operating_points.csv")

    print(f"\nSaved:\n  {V3/'table4_corrected_thresholds.csv'}\n"
          f"  {V3/'table5_isic_perfold_operating_points.csv'}")


if __name__ == "__main__":
    main()
