"""
Step 4: provenance audit of public dermoscopy mirrors against HAM10000 and
ISIC 2020, plus leakage inside each mirror's own train/test split.

For every mirror it reports:
  1. Overlap with HAM10000 and with ISIC 2020 (perceptual hash, 256-bit),
     per split, at Hamming thresholds 0/5/10/15. Matching is done both
     "as-is" and "dihedral" (the mirror image is also hashed in its 7
     rotated/flipped versions and the minimum distance is kept), because
     re-uploaded and augmented copies are often rotated or flipped.
  2. Label agreement: mirror class folder vs the HAM10000 diagnosis of the
     matched image.
  3. Internal leakage: how many images in a mirror's test/val split have a
     near-duplicate in its own train split, and how many images have a
     near-duplicate inside the same split.
  4. Mirror-vs-mirror overlap matrix.
  5. A null calibration: HAM10000 vs ISIC 2020, two sets we treat as
     independent, gives the false-match rate of the threshold.

Limits (state these in the paper): phash detects copies that were resized,
re-compressed, rotated by 90-degree steps or flipped. Crops, colour changes,
arbitrary-angle rotations and zooms can push a true copy above the threshold,
so all counts are lower bounds.

Run:
  export MELANOMA_ROOT=/path/to/data
  export KAGGLE_ROOT=/path/to/data/melanoma-skin-cancer
  caffeinate -dimsu python3 mirror_audit.py 2>&1 | tee mirror_audit_log.txt
Resumable: all hashes are cached. First run hashes about 60,000+ images
(8 orientations for mirror images), expect roughly 30-90 minutes.

Outputs in $MELANOMA_ROOT/results_mirror_audit/
  mirror_inventory.csv, overlap_summary.csv, label_crosstab.csv,
  internal_leakage.csv, mirror_vs_mirror.csv, null_calibration.csv,
  matches_<mirror>.csv (every mirror image with its nearest HAM/ISIC image)
  source_composition.csv   images per split x class x source (HAM10000 / ISIC2020 / unmatched)
  reference_coverage.csv   how many distinct HAM10000 / ISIC 2020 images and melanomas each mirror holds
  provenance_rule.csv      test accuracy of a rule that predicts the label from image source alone
"""

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# MELANOMA_ROOT if set, otherwise the data/ folder of this repository
ROOT = Path(os.environ.get("MELANOMA_ROOT", Path(__file__).resolve().parents[1] / "data")).expanduser()
KAGGLE_ROOT = Path(os.environ.get("KAGGLE_ROOT", ROOT / "melanoma-skin-cancer")).expanduser()
OUT = ROOT / "results_mirror_audit"
CACHE = OUT / "cache"

MIRRORS = {
    "Fanconi_3297": KAGGLE_ROOT,
    "Javid_10605": ROOT / "Javed_melanoma_cancer_dataset",
    "SkinCancerISIC_2357": ROOT / "skin_cancer9_calssesisic",
    "ISIC19_20_malig_benign_11400": ROOT / "skin-cancer-isic-2019-2020-malignant-or-benign",
    "HAM_augmented_balanced": ROOT / "skin-cancer-mnist10000-ham-augmented-dataset",
}

HASH_SIZE = 16
THRESHOLDS = [0, 5, 10, 15]
PRIMARY = 10
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
CHUNK = 512
N_WORKERS = max(1, (os.cpu_count() or 2) - 1)

SPLIT_TOKENS = [("train", "train"), ("test", "test"), ("valid", "val"), ("val", "val")]

ORIENTATIONS = [None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180,
                Image.Transpose.ROTATE_270, Image.Transpose.FLIP_LEFT_RIGHT,
                Image.Transpose.FLIP_TOP_BOTTOM, Image.Transpose.TRANSPOSE,
                Image.Transpose.TRANSVERSE]


# ------------------------------------------------------------------ hashing
def _hash_one(args):
    path, dihedral = args
    import imagehash
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    ops = ORIENTATIONS if dihedral else ORIENTATIONS[:1]
    out = []
    for op in ops:
        im = img if op is None else img.transpose(op)
        out.append(imagehash.phash(im, hash_size=HASH_SIZE).hash.flatten())
    return np.array(out, dtype=np.uint8)


def hash_paths(paths, tag, dihedral):
    """Returns (ok_mask, array [n_ok, n_orient, 256] as float32), aligned to
    `paths`. The cache stores the path of every hashed image, so a later run
    with a different file list reuses cached hashes by path and hashes only
    what is new."""
    cache = CACHE / f"{tag}_{'d8' if dihedral else 'd1'}.npz"
    n_or = 8 if dihedral else 1
    known = {}
    cache_has_paths = False
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            files = set(z.files)
            c_ok = z["ok"]                 # each npz entry is read ONCE
            c_H = z["H"]
            c_paths = z["paths"] if "paths" in files else None
        cache_has_paths = c_paths is not None
        if c_paths is None and len(c_ok) == len(paths):
            c_paths = np.array(paths)      # first-version cache, same file order
        if c_paths is not None:
            rows = np.cumsum(c_ok) - 1
            for p, good, r in zip(c_paths, c_ok, rows):
                known[str(p)] = c_H[r] if good else None   # views into one array
    todo = [p for p in paths if p not in known]
    if todo:
        print(f"  hashing {len(todo)} images for {tag} "
              f"({'8 orientations' if dihedral else 'as-is'}, {N_WORKERS} workers)")
        with ProcessPoolExecutor(N_WORKERS) as ex:
            for i, (p, r) in enumerate(zip(todo, ex.map(
                    _hash_one, [(q, dihedral) for q in todo], chunksize=64))):
                known[p] = r
                if (i + 1) % 5000 == 0:
                    print(f"    {i + 1}/{len(todo)}")
    ok = np.array([known[p] is not None for p in paths])
    H = (np.stack([known[p] for p in paths if known[p] is not None]) if ok.any()
         else np.zeros((0, n_or, HASH_SIZE * HASH_SIZE), np.uint8))
    if todo or not cache_has_paths:
        np.savez_compressed(cache, paths=np.array(paths), ok=ok, H=H)
    return ok, H.astype(np.float32)


def nn_search(Q, R, exclude_self=False):
    """Q: [n, n_or, d] query hashes; R: [m, d] reference hashes (orientation 0).
    Minimum Hamming distance over query orientations. Returns nn index,
    nn distance, and per-query count of references within PRIMARY."""
    n = Q.shape[0]
    r_sum = R.sum(axis=1)[None, :]
    best_d = np.full(n, np.inf, np.float32)
    best_i = np.zeros(n, np.int64)
    within = np.zeros(n, np.int64)
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        Dmin = None
        for o in range(Q.shape[1]):
            q = Q[s:e, o]
            D = q.sum(axis=1)[:, None] + r_sum - 2.0 * (q @ R.T)
            Dmin = D if Dmin is None else np.minimum(Dmin, D)
        if exclude_self:
            rows = np.arange(e - s)
            Dmin[rows, s + rows] = np.inf
        best_d[s:e] = Dmin.min(axis=1)
        best_i[s:e] = Dmin.argmin(axis=1)
        within[s:e] = (Dmin <= PRIMARY).sum(axis=1)
    return best_i, best_d, within


# ------------------------------------------------------------------ inventory
def list_images(root):
    rows = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMG_EXT or p.name.startswith("."):
            continue
        rel = p.relative_to(root)
        parts = [x.lower() for x in rel.parts[:-1]]
        split = next((lab for x in parts for tok, lab in SPLIT_TOKENS
                      if (tok in ("train", "test") and x.startswith(tok))
                      or x == tok or x.startswith(tok + "_") or x.startswith(tok + "-")
                      or x.endswith("_" + tok)
                      or (lab == "val" and x in ("validation", "valset", "val_set"))),
                     "all")
        cls = rel.parts[-2] if len(rel.parts) > 1 else "none"
        rows.append({"path": str(p), "rel": str(rel), "split": split, "class": cls})
    df = pd.DataFrame(rows)
    # stray files outside the split folders (e.g. a user's own file in the
    # dataset root) are dropped when the mirror has proper splits
    if not df.empty and (df["split"] != "all").any():
        df = df[df["split"] != "all"].reset_index(drop=True)
    return df


def provenance_tables(name, df):
    """Source composition per split/class, reference coverage, and the
    accuracy of a provenance-only rule (majority class per source, fitted on
    the train split, applied to the test split)."""
    df = df.copy()
    df["source"] = np.select([df["HAM10000_nn_dist"] <= PRIMARY,
                              df["ISIC2020_nn_dist"] <= PRIMARY],
                             ["HAM10000", "ISIC2020"], "unmatched")
    comp = (df.groupby(["split", "class", "source"]).size()
            .unstack(fill_value=0).reset_index())
    comp.insert(0, "mirror", name)

    cov = []
    for ref, n_ref_mel, lab in [("HAM10000", None, "ham_matched_melanoma"),
                                ("ISIC2020", None, "isic_matched_melanoma")]:
        m = df[df[f"{ref}_nn_dist"] <= PRIMARY]
        cov.append({"mirror": name, "reference": ref,
                    "mirror_images_matched": len(m),
                    "unique_reference_images": m[f"{ref}_nn_index"].nunique(),
                    "unique_reference_melanomas":
                        m.loc[m[lab] == 1, f"{ref}_nn_index"].nunique()})

    rule = None
    classes = set(df["class"].str.lower())
    if classes == {"benign", "malignant"} and {"train", "test"} <= set(df["split"]):
        tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
        mapping = tr.groupby("source")["class"].agg(lambda s: s.value_counts().idxmax())
        pred = te["source"].map(mapping)
        rule = {"mirror": name, "n_test": len(te),
                "rule": "; ".join(f"{k}->{v}" for k, v in mapping.items()),
                "test_accuracy_from_provenance_only": float((pred == te["class"]).mean()),
                "test_majority_class_accuracy":
                    float(te["class"].value_counts(normalize=True).max())}
    return comp, cov, rule


def ham_dx_lookup(ham):
    """HAM10000 diagnosis per manifest row, via image_id in the file name."""
    meta_path = ROOT / "HAM10000_metadata.tab"
    if not meta_path.exists():
        meta_path = ROOT / "HAM10000_metadata.csv"
    if not meta_path.exists():
        return None
    sep = "\t" if meta_path.suffix == ".tab" else ","
    meta = pd.read_csv(meta_path, sep=sep)
    dx = dict(zip(meta["image_id"], meta["dx"]))
    ids = [Path(p).stem for p in ham["image_path"]]
    out = [dx.get(i) for i in ids]
    return out if sum(x is not None for x in out) > 0.9 * len(out) else None


# ------------------------------------------------------------------ main
def main():
    OUT.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)

    ham = pd.read_csv(ROOT / "manifest_ham10000_384.csv")
    isic = pd.read_csv(ROOT / "manifest_isic2020_384.csv")
    print("Reference sets:")
    ok_h, Hh = hash_paths(ham["image_path"].tolist(), "ref_ham10000", False)
    ok_i, Hi = hash_paths(isic["image_path"].tolist(), "ref_isic2020", False)
    if not (ok_h.all() and ok_i.all()):
        raise SystemExit("Some reference images failed to open; check manifests.")
    Rh, Ri = Hh[:, 0], Hi[:, 0]
    ham_y = ham["label"].values.astype(int)
    isic_y = isic["label"].values.astype(int)
    ham_dx = ham_dx_lookup(ham)
    if ham_dx is None:
        print("  (HAM10000 metadata not found or ids unmatched; label crosstab "
              "will use melanoma yes/no only)")

    # ---- null calibration: ISIC 2020 vs HAM10000
    _, d_null, _ = nn_search(Hi[:, :1], Rh)
    null = {f"pct_isic2020_within_{t}_of_ham": 100.0 * (d_null <= t).mean()
            for t in THRESHOLDS}
    null.update({"nn_p1": np.percentile(d_null, 1), "nn_median": np.median(d_null)})
    pd.DataFrame([null]).to_csv(OUT / "null_calibration.csv", index=False)
    print(f"\nNull calibration (ISIC 2020 vs HAM10000): "
          f"{null[f'pct_isic2020_within_{PRIMARY}_of_ham']:.3f}% within d<={PRIMARY}, "
          f"median NN distance {null['nn_median']:.0f}")

    inventory, summary, crosstabs, internal = [], [], [], []
    compositions, coverage, rules = [], [], []
    mirror_hashes = {}

    for name, root in MIRRORS.items():
        print(f"\n=== {name} ===")
        if not root.exists():
            print(f"  missing folder {root}; skipped")
            continue
        df = list_images(root)
        if df.empty:
            print("  no images found; skipped")
            continue
        ok, H = hash_paths(df["path"].tolist(), f"mirror_{name}", True)
        df = df[ok].reset_index(drop=True)
        mirror_hashes[name] = (df, H)
        inv = df.groupby(["split", "class"]).size().reset_index(name="n_images")
        inv.insert(0, "mirror", name)
        inventory.append(inv)
        print(f"  {len(df)} images | splits: "
              f"{df['split'].value_counts().to_dict()} | classes: {df['class'].nunique()}")

        # overlap with the two reference sets, as-is and dihedral
        for ref_name, R in [("HAM10000", Rh), ("ISIC2020", Ri)]:
            for mode, Q in [("as_is", H[:, :1]), ("dihedral", H)]:
                idx, d, _ = nn_search(Q, R)
                if mode == "dihedral":
                    df[f"{ref_name}_nn_index"] = idx
                    df[f"{ref_name}_nn_dist"] = d
                for split, g in df.groupby("split"):
                    dd = d[g.index.values]
                    row = {"mirror": name, "split": split, "reference": ref_name,
                           "match_mode": mode, "n_images": len(dd),
                           "nn_median": float(np.median(dd))}
                    for t in THRESHOLDS:
                        row[f"n_d<={t}"] = int((dd <= t).sum())
                        row[f"pct_d<={t}"] = 100.0 * (dd <= t).mean()
                    summary.append(row)
            m = df[f"{ref_name}_nn_dist"] <= PRIMARY
            print(f"  vs {ref_name}: {int(m.sum())}/{len(df)} "
                  f"({100 * m.mean():.1f}%) within d<={PRIMARY} (dihedral)")

        # label agreement with HAM10000 for matched images
        m = df["HAM10000_nn_dist"] <= PRIMARY
        if m.any():
            idx = df.loc[m, "HAM10000_nn_index"].values
            ref_lab = ([ham_dx[i] for i in idx] if ham_dx is not None
                       else np.where(ham_y[idx] == 1, "mel", "non-mel"))
            ct = pd.crosstab(df.loc[m, "class"].values, np.array(ref_lab))
            ct = ct.reset_index().rename(columns={"row_0": "mirror_class"})
            ct.insert(0, "mirror", name)
            crosstabs.append(ct)
        m2 = df["ISIC2020_nn_dist"] <= PRIMARY
        df["ham_matched_melanoma"] = np.where(
            m, ham_y[df["HAM10000_nn_index"].values], -1)
        df["isic_matched_melanoma"] = np.where(
            m2, isic_y[df["ISIC2020_nn_index"].values], -1)

        # internal leakage: eval splits vs this mirror's train split
        train = df["split"] == "train"
        for ev in ["test", "val"]:
            evm = df["split"] == ev
            if train.any() and evm.any():
                _, d, _ = nn_search(H[evm.values], H[train.values, 0])
                internal.append({
                    "mirror": name, "comparison": f"{ev}_vs_train",
                    "n_eval": int(evm.sum()),
                    "n_eval_with_train_duplicate": int((d <= PRIMARY).sum()),
                    "pct": 100.0 * (d <= PRIMARY).mean()})
        # duplicates inside each split (self excluded)
        for split, g in df.groupby("split"):
            ii = g.index.values
            if len(ii) < 2:
                continue
            _, d, _ = nn_search(H[ii], H[ii, 0], exclude_self=True)
            internal.append({
                "mirror": name, "comparison": f"within_{split}",
                "n_eval": len(ii),
                "n_eval_with_train_duplicate": int((d <= PRIMARY).sum()),
                "pct": 100.0 * (d <= PRIMARY).mean()})
        comp, cov, rule = provenance_tables(name, df)
        compositions.append(comp)
        coverage.extend(cov)
        if rule:
            rules.append(rule)
            print(f"  provenance-only rule on test: "
                  f"{100 * rule['test_accuracy_from_provenance_only']:.1f}% accuracy "
                  f"(majority class {100 * rule['test_majority_class_accuracy']:.1f}%)")
        df.drop(columns=["path"]).to_csv(OUT / f"matches_{name}.csv", index=False)

    # mirror-vs-mirror matrix (row mirror images found in column mirror)
    names = list(mirror_hashes)
    mat = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            if a == b:
                continue
            _, d, _ = nn_search(mirror_hashes[a][1], mirror_hashes[b][1][:, 0])
            mat.loc[a, b] = 100.0 * (d <= PRIMARY).mean()
    mat.to_csv(OUT / "mirror_vs_mirror.csv")

    pd.concat(inventory).to_csv(OUT / "mirror_inventory.csv", index=False)
    summ = pd.DataFrame(summary)
    summ.to_csv(OUT / "overlap_summary.csv", index=False)
    if crosstabs:
        pd.concat(crosstabs).fillna(0).to_csv(OUT / "label_crosstab.csv", index=False)
    pd.DataFrame(internal).to_csv(OUT / "internal_leakage.csv", index=False)
    pd.concat(compositions).fillna(0).to_csv(OUT / "source_composition.csv", index=False)
    pd.DataFrame(coverage).to_csv(OUT / "reference_coverage.csv", index=False)
    if rules:
        pd.DataFrame(rules).to_csv(OUT / "provenance_rule.csv", index=False)

    print("\n=== Overlap at d<=10, dihedral matching ===")
    view = summ[summ.match_mode == "dihedral"][
        ["mirror", "split", "reference", "n_images", f"n_d<={PRIMARY}", f"pct_d<={PRIMARY}"]]
    print(view.round(2).to_string(index=False))
    print("\n=== Internal leakage (d<=10) ===")
    print(pd.DataFrame(internal).round(2).to_string(index=False))
    print("\n=== Mirror vs mirror: % of row mirror found in column mirror ===")
    print(mat.round(1).to_string())
    print("\n=== Reference coverage ===")
    print(pd.DataFrame(coverage).to_string(index=False))
    if rules:
        print("\n=== Provenance-only rule (fitted on train, scored on test) ===")
        print(pd.DataFrame(rules).round(4).to_string(index=False))
    print(f"\nAll outputs in {OUT}")


if __name__ == "__main__":
    main()
