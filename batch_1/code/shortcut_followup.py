"""
Follow-up to shortcut_experiment.py. No feature extraction: reuses the cached
ImageNet features in results_shortcut/.

1. Patient overlap inside the ISIC 2020 portion of each mirror.
   ISIC 2020 has several lesions per patient. The mirrors were split without
   regard to patient, so a test image may share a patient with training
   images. For each mirror we count this, then recompute
     B (the full-mirror model scored on ISIC 2020-source test images) and
     D (the model trained and tested within the ISIC 2020 source)
   on the patient-clean subset only (test patients never seen in training).
   Patient IDs come from the official ISIC 2020 ground-truth CSV (column
   patient_id), matched through the image name (ISIC_xxxxxxx). The file is
   found automatically in $MELANOMA_ROOT or one folder below it; set
   ISIC_GT=/path/to/file.csv to point at it directly.

2. Image-size rule. For each binary mirror, the most common class per image
   size (width x height) is fitted on the train split and scored on the test
   split, the same way as the provenance rule.

Run:
  export MELANOMA_ROOT=/path/to/data
  export KAGGLE_ROOT=/path/to/data/melanoma-skin-cancer
  python3 shortcut_followup.py 2>&1 | tee shortcut_followup_log.txt
Outputs in $MELANOMA_ROOT/results_shortcut/
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# MELANOMA_ROOT if set, otherwise the data/ folder of this repository
ROOT = Path(os.environ.get("MELANOMA_ROOT", Path(__file__).resolve().parents[1] / "data")).expanduser()
KAGGLE_ROOT = Path(os.environ.get("KAGGLE_ROOT", ROOT / "melanoma-skin-cancer")).expanduser()
AUDIT = ROOT / "results_mirror_audit"
SC = ROOT / "results_shortcut"
PRIMARY, SEED, N_BOOT = 10, 42, 2000

FEATURE_MIRRORS = {
    "Javid_10605": ROOT / "Javed_melanoma_cancer_dataset",
    "ISIC19_20_malig_benign_11400": ROOT / "skin-cancer-isic-2019-2020-malignant-or-benign",
}
SIZE_MIRRORS = dict(FEATURE_MIRRORS, Fanconi_3297=KAGGLE_ROOT)


def load(name, root):
    """Same row selection and order as shortcut_experiment.py."""
    m = pd.read_csv(AUDIT / f"matches_{name}.csv")
    m = m[m["split"].isin(["train", "test"]) &
          m["class"].str.lower().isin(["benign", "malignant"])].reset_index(drop=True)
    m["path"] = [str(root / r) for r in m["rel"]]
    m["y"] = (m["class"].str.lower() == "malignant").astype(int)
    m["is_isic2020"] = ((m["HAM10000_nn_dist"] > PRIMARY) &
                        (m["ISIC2020_nn_dist"] <= PRIMARY)).astype(int)
    return m


def xgb(y):
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                         eval_metric="logloss", random_state=SEED, n_jobs=-1)


def auc_ci(y, p, rng):
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    vals = []
    while len(vals) < N_BOOT:
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(roc_auc_score(y[i], p[i]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return roc_auc_score(y, p), lo, hi


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
    return None


def patient_ids():
    """patient_id for every row of manifest_isic2020_384.csv."""
    isic = pd.read_csv(ROOT / "manifest_isic2020_384.csv")
    gt_path = find_ground_truth()
    if gt_path is None:
        raise SystemExit("ISIC 2020 ground-truth CSV with columns image_name and "
                         "patient_id not found. Set ISIC_GT=/path/to/file.csv")
    gt = pd.read_csv(gt_path)
    lookup = dict(zip(gt["image_name"].astype(str), gt["patient_id"].astype(str)))
    names = [Path(p).stem for p in isic["image_path"]]
    pid = np.array([lookup.get(n) for n in names], dtype=object)
    found = np.mean([x is not None for x in pid])
    print(f"ISIC 2020 ground truth: {gt_path.name} | manifest images mapped to a "
          f"patient: {100 * found:.2f}% | distinct patients: "
          f"{len(set(x for x in pid if x is not None))}")
    if found < 0.99:
        raise SystemExit("Fewer than 99% of manifest images matched a patient_id; "
                         "check that the manifest file names are ISIC image names.")
    return pid


def patient_analysis(rng):
    patient = patient_ids()
    rows = []
    for name, root in FEATURE_MIRRORS.items():
        feat = SC / f"features_{name}.npy"
        if not feat.exists():
            print(f"  {name}: cached features missing; run shortcut_experiment.py first")
            continue
        df = load(name, root)
        X = np.load(feat)
        if len(X) != len(df):
            raise SystemExit(f"{name}: feature rows ({len(X)}) != images ({len(df)})")
        df["patient"] = np.where(df["is_isic2020"] == 1,
                                 patient[df["ISIC2020_nn_index"].values], None)
        tr = (df["split"] == "train").values
        te = (df["split"] == "test").values
        s, y = df["is_isic2020"].values == 1, df["y"].values
        train_patients = set(df.loc[tr & s, "patient"])
        seen = np.array([p in train_patients for p in df["patient"]])
        b_all = te & s
        b_clean = b_all & ~seen
        print(f"\n=== {name} ===")
        print(f"  ISIC 2020-source test images: {b_all.sum()} "
              f"({int(y[b_all].sum())} malignant); patient also in train: "
              f"{int((b_all & seen).sum())} ({100 * (b_all & seen).sum() / max(b_all.sum(), 1):.1f}%)")
        print(f"  patient-clean subset: {b_clean.sum()} images, "
              f"{int(y[b_clean].sum())} malignant")

        sc = StandardScaler().fit(X[tr])
        p_full = xgb(y[tr]).fit(sc.transform(X[tr]), y[tr]).predict_proba(sc.transform(X))[:, 1]
        dtr = tr & s
        sc_d = StandardScaler().fit(X[dtr])
        p_ctrl = xgb(y[dtr]).fit(sc_d.transform(X[dtr]), y[dtr]).predict_proba(sc_d.transform(X))[:, 1]

        for tag, mask in [("all ISIC2020-source test", b_all),
                          ("patient-clean only", b_clean),
                          ("patient seen in train", b_all & seen)]:
            for model, p in [("B_full_mirror_model", p_full), ("D_within_source_control", p_ctrl)]:
                a, lo, hi = auc_ci(y[mask], p[mask], rng)
                rows.append({"mirror": name, "subset": tag, "model": model,
                             "n": int(mask.sum()), "n_malignant": int(y[mask].sum()),
                             "auroc": a, "ci_lo": lo, "ci_hi": hi})
                print(f"  {model:<26} {tag:<26} n={mask.sum():>4} "
                      f"mal={int(y[mask].sum()):>3}  AUROC={a:.4f} [{lo:.4f}, {hi:.4f}]")
    pd.DataFrame(rows).to_csv(SC / "patient_overlap_results.csv", index=False)


def size_rule():
    rows = []
    print("\n=== Image-size rule (fitted on train, scored on test) ===")
    for name, root in SIZE_MIRRORS.items():
        df = load(name, root)
        sizes = []
        for p in df["path"]:
            with Image.open(p) as im:
                sizes.append(f"{im.size[0]}x{im.size[1]}")
        df["size"] = sizes
        tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
        overall_major = tr["y"].mode()[0]
        mapping = tr.groupby("size")["y"].agg(lambda s: s.mode()[0])
        pred = te["size"].map(mapping).fillna(overall_major).astype(int)
        acc = float((pred == te["y"]).mean())
        bac = float(balanced_accuracy_score(te["y"], pred))
        n_sizes = df["size"].nunique()
        rows.append({"mirror": name, "n_test": len(te), "distinct_sizes": n_sizes,
                     "test_accuracy_from_size_only": acc,
                     "test_balanced_accuracy_from_size_only": bac,
                     "test_majority_class_accuracy": float(te["y"].value_counts(normalize=True).max())})
        print(f"  {name:<30} distinct sizes={n_sizes:>4}  size-only accuracy={100 * acc:.1f}%  "
              f"balanced={100 * bac:.1f}%  (majority class {100 * rows[-1]['test_majority_class_accuracy']:.1f}%)")
        tab = df.groupby(["split", "class", "size"]).size().rename("n").reset_index()
        tab.to_csv(SC / f"image_sizes_{name}.csv", index=False)
        if n_sizes > 1:
            print("    most common sizes per split and class:")
            for (sp, cl), g in tab.groupby(["split", "class"]):
                top = g.sort_values("n", ascending=False).head(4)
                desc = ", ".join(f"{s} ({n})" for s, n in zip(top["size"], top["n"]))
                print(f"      {sp:<5} {cl:<9} {desc}")
    pd.DataFrame(rows).to_csv(SC / "size_rule_results.csv", index=False)


def main():
    rng = np.random.default_rng(SEED)
    patient_analysis(rng)
    size_rule()
    print(f"\nAll outputs in {SC}")


if __name__ == "__main__":
    main()
