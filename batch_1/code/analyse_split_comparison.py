"""
Image-wise vs lesion-grouped splitting: paired statistics, confound checks
and the main comparison figure. No retraining; reads saved probabilities.

Inputs (all under $MELANOMA_ROOT):
  manifest_ham10000_384.csv, manifest_isic2020_384.csv
  results_v3/probs/fold1..5.npz            (lesion-grouped run)
  results_split_image/probs/fold1..5.npz   (image-wise run)

What it computes:
  1. Paired cluster bootstrap of delta AUROC (image-wise minus lesion-grouped)
     for every classifier and k, internally (HAM10000 OOF, resampling LESIONS)
     and externally (ISIC 2020 fold ensemble, resampling PATIENTS, taken from
     the official ISIC 2020 ground-truth CSV, because ISIC 2020 has several
     lesions per patient). Both runs are scored on the same images in each
     replicate, so the interval reflects the paired difference.
  2. Split composition check: unique lesions and images-per-lesion in the
     training partitions of both schemes (rebuilt deterministically with the
     same splitters and seed as the two runs). This tests one explanation for
     the small external gain of the image-wise models: their training sets
     may cover more distinct lesions for the same number of images.
  3. Composition of the leaked vs clean test subsets in the image-wise run
     (melanoma share, share of multi-image lesions), which shows how far the
     leaked/clean AUROC split is confounded by case mix.
  4. Figure: internal and external AUROC for both schemes, all 15 configs.

The ISIC 2020 ground-truth CSV (columns image_name, patient_id) is found
automatically in $MELANOMA_ROOT or one folder below; or set ISIC_GT.

Run:
  export MELANOMA_ROOT=/path/to/data
  python3 analyse_split_comparison.py            # B = 2000 bootstrap replicates
  python3 analyse_split_comparison.py --boot 500 # quicker trial
Outputs in $MELANOMA_ROOT/results_split_comparison/
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold,
                                     StratifiedKFold, StratifiedShuffleSplit)

# MELANOMA_ROOT if set, otherwise the data/ folder of this repository
ROOT = Path(os.environ.get("MELANOMA_ROOT", Path(__file__).resolve().parents[1] / "data")).expanduser()
LES_DIR = ROOT / "results_v3" / "probs"
IMG_DIR = ROOT / "results_split_image" / "probs"
OUT = ROOT / "results_split_comparison"

CLASSIFIERS = ["XGBoost", "RandomForest", "SVM-RBF", "KNN", "Stacking"]
KTAGS = ["256", "512", "all"]
N_FOLDS, SEED, VAL_FRACTION = 5, 42, 0.15


# ------------------------------------------------------------------ loading
def load_run(prob_dir):
    files = [prob_dir / f"fold{f}.npz" for f in range(1, N_FOLDS + 1)]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"Missing probability files: {missing}")
    return [np.load(f, allow_pickle=True) for f in files]


def oof_probs(data, n, key):
    p = np.full(n, np.nan)
    for d in data:
        p[d["test_index"]] = d[f"te__{key}"]
    if np.isnan(p).any():
        raise SystemExit(f"OOF coverage incomplete for {key}")
    return p


def ext_probs(data, key):
    return np.mean([d[f"ex__{key}"] for d in data], axis=0)


# ------------------------------------------------------------------ ISIC 2020 patients
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


def isic_patients(isic):
    gt_path = find_ground_truth()
    if gt_path is None:
        raise SystemExit("ISIC 2020 ground-truth CSV (image_name, patient_id) not "
                         "found; set ISIC_GT=/path/to/file.csv")
    gt = pd.read_csv(gt_path)
    lookup = dict(zip(gt["image_name"].astype(str), gt["patient_id"].astype(str)))
    pid = np.array([lookup.get(Path(p).stem) for p in isic["image_path"]], dtype=object)
    found = np.mean([x is not None for x in pid])
    if found < 0.99:
        raise SystemExit(f"Only {100 * found:.1f}% of ISIC 2020 images matched a patient_id.")
    print(f"ISIC 2020 patients: {len(set(pid))} (from {gt_path.name})")
    return pid


# ------------------------------------------------------------------ bootstrap
def cluster_index(groups):
    """Map each cluster to the array of row indices that belong to it."""
    codes, uniq = pd.factorize(groups)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    return [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]


def paired_cluster_bootstrap(y, p_a, p_b, clusters, B, rng):
    """Delta AUROC (a minus b) with a percentile CI; clusters resampled with
    replacement, both models scored on the identical resample."""
    n_c = len(clusters)
    deltas = np.empty(B)
    b = 0
    while b < B:
        pick = rng.integers(0, n_c, n_c)
        idx = np.concatenate([clusters[i] for i in pick])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        deltas[b] = roc_auc_score(yy, p_a[idx]) - roc_auc_score(yy, p_b[idx])
        b += 1
    point = roc_auc_score(y, p_a) - roc_auc_score(y, p_b)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    # two-sided bootstrap p-value for H0: delta = 0
    p_two = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return point, lo, hi, max(p_two, 1.0 / B)


# ------------------------------------------------------------------ split composition
def rebuild_train_sets(scheme, y, groups):
    """Recreates the training partitions exactly as the two runs made them."""
    idx_all = np.arange(len(y))
    if scheme == "lesion":
        outer = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED).split(idx_all, y, groups)
    else:
        outer = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(idx_all, y)
    sets = []
    for tr_idx, te_idx in outer:
        if scheme == "lesion":
            gss = GroupShuffleSplit(1, test_size=VAL_FRACTION, random_state=SEED)
            i_tr, _ = next(gss.split(tr_idx, y[tr_idx], groups[tr_idx]))
        else:
            sss = StratifiedShuffleSplit(1, test_size=VAL_FRACTION, random_state=SEED)
            i_tr, _ = next(sss.split(tr_idx, y[tr_idx]))
        sets.append((tr_idx[i_tr], te_idx))
    return sets


def composition_table(y, groups):
    rows = []
    for scheme in ["lesion", "image"]:
        for fold, (tr, te) in enumerate(rebuild_train_sets(scheme, y, groups), 1):
            mel = tr[y[tr] == 1]
            rows.append({
                "scheme": scheme, "fold": fold, "train_images": len(tr),
                "train_unique_lesions": len(np.unique(groups[tr])),
                "train_melanoma_images": len(mel),
                "train_unique_melanoma_lesions": len(np.unique(groups[mel])),
                "images_per_lesion": len(tr) / len(np.unique(groups[tr])),
                "test_index_first5": ",".join(map(str, np.sort(te)[:5])),
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    ham = pd.read_csv(ROOT / "manifest_ham10000_384.csv")
    isic = pd.read_csv(ROOT / "manifest_isic2020_384.csv")
    y = ham["label"].values.astype(int)
    groups = ham["lesion_group"].values
    y_ext = isic["label"].values.astype(int)
    ext_groups = isic_patients(isic)

    les, img = load_run(LES_DIR), load_run(IMG_DIR)

    # sanity: both runs must have scored ISIC in manifest order
    for d in les + img:
        if not np.array_equal(d["isic_y"], y_ext):
            raise SystemExit("ISIC label order differs from the manifest; stop.")

    # 2. composition (also confirms the rebuilt image-wise folds match the run)
    comp = composition_table(y, groups)
    rebuilt_img_te = [te for _, te in rebuild_train_sets("image", y, groups)]
    for f, d in enumerate(img):
        if not np.array_equal(np.sort(d["test_index"]), np.sort(rebuilt_img_te[f])):
            raise SystemExit(f"Rebuilt image-wise fold {f+1} does not match the saved run.")
    comp.drop(columns="test_index_first5").to_csv(OUT / "split_composition.csv", index=False)
    summ = comp.groupby("scheme")[["train_unique_lesions", "train_unique_melanoma_lesions",
                                   "images_per_lesion"]].mean().round(2)
    print("\n=== Training-partition composition (mean over folds) ===")
    print(summ.to_string())

    # 3. leaked vs clean subset case mix (image-wise run)
    leaked = np.zeros(len(y), dtype=bool)
    for d in img:
        leaked[d["test_index"]] = d["leaked"]
    per_lesion = pd.Series(groups).map(pd.Series(groups).value_counts()).values
    mix = pd.DataFrame([{
        "subset": name, "images": int(m.sum()),
        "melanoma_share": y[m].mean(),
        "share_from_multi_image_lesions": (per_lesion[m] > 1).mean(),
    } for name, m in [("leaked", leaked), ("clean", ~leaked)]])
    mix.to_csv(OUT / "leaked_clean_case_mix.csv", index=False)
    print("\n=== Case mix of leaked vs clean test images (image-wise run) ===")
    print(mix.round(4).to_string(index=False))

    # 1. paired bootstrap
    ham_clusters = cluster_index(groups)
    ext_clusters = cluster_index(ext_groups)
    rows = []
    for ktag in KTAGS:
        for clf in CLASSIFIERS:
            key = f"{ktag}__{clf}"
            pi, pl = oof_probs(img, len(y), key), oof_probs(les, len(y), key)
            ei, el = ext_probs(img, key), ext_probs(les, key)
            d_int = paired_cluster_bootstrap(y, pi, pl, ham_clusters, args.boot, rng)
            d_ext = paired_cluster_bootstrap(y_ext, ei, el, ext_clusters, args.boot, rng)
            rows.append({
                "classifier": clf, "k": ktag,
                "internal_lesion": roc_auc_score(y, pl), "internal_image": roc_auc_score(y, pi),
                "internal_delta": d_int[0], "internal_ci_lo": d_int[1],
                "internal_ci_hi": d_int[2], "internal_p": d_int[3],
                "external_lesion": roc_auc_score(y_ext, el), "external_image": roc_auc_score(y_ext, ei),
                "external_delta": d_ext[0], "external_ci_lo": d_ext[1],
                "external_ci_hi": d_ext[2], "external_p": d_ext[3],
                "gap_lesion": roc_auc_score(y, pl) - roc_auc_score(y_ext, el),
                "gap_image": roc_auc_score(y, pi) - roc_auc_score(y_ext, ei),
            })
            r = rows[-1]
            print(f"k={ktag:>3} {clf:>12}: internal d={r['internal_delta']:+.4f} "
                  f"[{r['internal_ci_lo']:+.4f}, {r['internal_ci_hi']:+.4f}] | "
                  f"external d={r['external_delta']:+.4f} "
                  f"[{r['external_ci_lo']:+.4f}, {r['external_ci_hi']:+.4f}]")
    res = pd.DataFrame(rows)
    res["gap_widening"] = res["gap_image"] - res["gap_lesion"]
    res.to_csv(OUT / "paired_bootstrap_deltas.csv", index=False)
    print("\nMean internal delta {:.4f}, mean external delta {:.4f}, "
          "mean widening of internal-external gap {:.4f}".format(
              res.internal_delta.mean(), res.external_delta.mean(), res.gap_widening.mean()))

    make_figure(res)
    print(f"\nAll outputs in {OUT}")


def make_figure(res):
    labels = [f"{c}\nk={k}" for k, c in zip(res.k, res.classifier)]
    x = np.arange(len(res))
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    panels = [("internal", "HAM10000 out-of-fold AUROC"),
              ("external", "ISIC 2020 external AUROC (fold ensemble)")]
    for ax, (tag, title) in zip(axes, panels):
        ax.scatter(x - 0.12, res[f"{tag}_lesion"], marker="o", s=38,
                   color="#1f4e79", label="Lesion-grouped split", zorder=3)
        ax.scatter(x + 0.12, res[f"{tag}_image"], marker="^", s=42,
                   color="#c0504d", label="Image-wise split", zorder=3)
        for i in x:
            ax.plot([i - 0.12, i + 0.12], [res[f"{tag}_lesion"][i], res[f"{tag}_image"][i]],
                    color="grey", lw=0.8, zorder=2)
        ax.set_ylabel("AUROC")
        ax.set_title(title, fontsize=11, loc="left")
        ax.grid(axis="y", alpha=0.3)
        for b in [4.5, 9.5]:
            ax.axvline(b, color="black", lw=0.5, alpha=0.4)
    axes[0].legend(loc="lower right", frameon=False)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"fig_split_comparison.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)


if __name__ == "__main__":
    main()
