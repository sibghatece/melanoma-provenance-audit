"""
Step 0: build the 384-px image cache and the two manifests that every other
script reads.

Inputs (official downloads, placed in $MELANOMA_ROOT; see data/README.md):
  HAM10000_images_part_1/, HAM10000_images_part_2/   the 10,015 JPEGs
  HAM10000_metadata.tab  (or HAM10000_metadata.csv)  columns image_id, lesion_id, dx
  ISIC_Image_Dataset/                                ISIC 2020 training JPEGs
  ISIC_2020_Training_GroundTruth_v2.csv              image_name, patient_id,
                                                     lesion_id, diagnosis
  ISIC_2020_Training_Duplicates.csv                  image_name_1, image_name_2

What it does:
  * HAM10000: label = 1 if dx == "mel", else 0; lesion_group = lesion_id.
  * ISIC 2020: the 425 pixel-wise identical duplicates listed by the
    organisers are removed (the second image of each pair), leaving 32,701
    images; label = 1 if diagnosis == "melanoma"; lesion_group = lesion_id,
    or the image name where lesion_id is missing.
  * Every image is resized once to 384 x 384 (bicubic) and saved as JPEG at
    quality 95. Training scripts read these cached copies.

If the manifests released with the paper are present in
results/manifests/, the ISIC 2020 image list is taken from there (so the
selection is exactly the published one), and both manifests are checked for
identical images, labels and lesion groups.

Run:
  export MELANOMA_ROOT=/path/to/data
  python3 code/prepare_data.py
Outputs in $MELANOMA_ROOT:
  cache_384/ham10000/*.jpg, cache_384/isic2020/*.jpg
  manifest_ham10000_384.csv, manifest_isic2020_384.csv   (image_path, label,
                                                          lesion_group, dx or diagnosis)
  prepare_data_report.txt
Resumable: images already in the cache are skipped.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image

# MELANOMA_ROOT if set, otherwise the data/ folder of this repository
ROOT = Path(os.environ.get("MELANOMA_ROOT", Path(__file__).resolve().parents[1] / "data")).expanduser()
REPO = Path(__file__).resolve().parents[1]

HAM_DIRS = [ROOT / "HAM10000_images_part_1", ROOT / "HAM10000_images_part_2"]
ISIC_DIR = ROOT / "ISIC_Image_Dataset"
ISIC_GT = ROOT / "ISIC_2020_Training_GroundTruth_v2.csv"
ISIC_DUP = ROOT / "ISIC_2020_Training_Duplicates.csv"

CACHE_HAM = ROOT / "cache_384" / "ham10000"
CACHE_ISIC = ROOT / "cache_384" / "isic2020"
SIZE, QUALITY = 384, 95
N_WORKERS = max(1, (os.cpu_count() or 2) - 1)


def find_ham_metadata():
    for name in ("HAM10000_metadata.tab", "HAM10000_metadata.csv", "HAM10000_metadata"):
        p = ROOT / name
        if p.exists():
            return p
    sys.exit(f"HAM10000 metadata not found in {ROOT} (expected HAM10000_metadata.tab or .csv)")


def read_table(path):
    with open(path, encoding="utf-8") as f:
        head = f.readline()
    return pd.read_csv(path, sep="\t" if head.count("\t") > head.count(",") else ",")


def resize_one(job):
    src, dst = job
    if dst.exists():
        return src.name, True, "cached"
    try:
        with Image.open(src) as im:
            im.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC).save(dst, "JPEG", quality=QUALITY)
        return src.name, True, ""
    except Exception as e:  # report and continue
        return src.name, False, str(e)


def cache(sources, out_dir, tag):
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(s, out_dir / s.name) for s in sources]
    ok, failed = 0, []
    print(f"Caching {len(jobs)} {tag} images at {SIZE} px ({N_WORKERS} workers)")
    with ProcessPoolExecutor(N_WORKERS) as ex:
        futures = [ex.submit(resize_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            name, success, note = fut.result()
            ok += success
            if not success:
                failed.append((name, note))
            if i % 2500 == 0:
                print(f"  {i}/{len(jobs)}")
    return ok, failed


def ham_manifest():
    meta = read_table(find_ham_metadata())
    df = pd.DataFrame({
        "image_path": [str(CACHE_HAM / f"{i}.jpg") for i in meta["image_id"]],
        "label": (meta["dx"] == "mel").astype(int),
        "lesion_group": meta["lesion_id"],
        "dx": meta["dx"],
    })
    return df[df["image_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)


RELEASED_ISIC = REPO / "results" / "manifests" / "manifest_isic2020_384.csv"


def isic_manifest():
    """Removes the listed duplicates. If the manifest released with the paper
    is present, its image list is used, so the selection is exactly the one
    behind the published results; otherwise the second image of each listed
    pair is dropped."""
    gt = pd.read_csv(ISIC_GT)
    dup = pd.read_csv(ISIC_DUP)
    if RELEASED_ISIC.exists():
        keep = set(pd.read_csv(RELEASED_ISIC)["file"].str.replace(".jpg", "", regex=False))
        gt = gt[gt["image_name"].isin(keep)]
    else:
        gt = gt[~gt["image_name"].isin(set(dup["image_name_2"]))]
    df = pd.DataFrame({
        "image_path": [str(CACHE_ISIC / f"{n}.jpg") for n in gt["image_name"]],
        "label": (gt["diagnosis"] == "melanoma").astype(int).values,
        "lesion_group": gt["lesion_id"].fillna(gt["image_name"]).values,
        "diagnosis": gt["diagnosis"].values,
    })
    return df[df["image_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)


def compare_with_release(df, released_csv, tag):
    """Compares file names, labels and lesion groups with the released manifest."""
    if not released_csv.exists():
        return f"{tag}: no released manifest found in results/manifests/, comparison skipped"
    rel = pd.read_csv(released_csv)
    mine = df.assign(file=df["image_path"].map(lambda p: Path(p).name))[["file", "label", "lesion_group"]]
    merged = rel.merge(mine, on="file", how="outer", suffixes=("_released", "_here"), indicator=True)
    only_rel = int((merged["_merge"] == "left_only").sum())
    only_here = int((merged["_merge"] == "right_only").sum())
    both = merged[merged["_merge"] == "both"]
    lab = int((both["label_released"] != both["label_here"]).sum())
    grp = int((both["lesion_group_released"].astype(str) != both["lesion_group_here"].astype(str)).sum())
    status = "IDENTICAL" if only_rel == only_here == lab == grp == 0 else "DIFFERENT"
    return (f"{tag}: {status} to the released manifest (only in released {only_rel}, only here "
            f"{only_here}, label differences {lab}, lesion-group differences {grp})")


def main():
    print(f"[Data folder] {ROOT}")
    ham_src = sorted(p for d in HAM_DIRS for p in d.glob("*.jpg"))
    isic_src = sorted(ISIC_DIR.glob("*.jpg"))
    for need, what in [(ham_src, "HAM10000 images"), (isic_src, "ISIC 2020 images")]:
        if not need:
            sys.exit(f"{what} not found. See data/README.md for the expected layout.")
    for f in (ISIC_GT, ISIC_DUP):
        if not f.exists():
            sys.exit(f"{f.name} not found in {ROOT}. See data/README.md.")

    h_ok, h_fail = cache(ham_src, CACHE_HAM, "HAM10000")
    i_ok, i_fail = cache(isic_src, CACHE_ISIC, "ISIC 2020")

    ham, isic = ham_manifest(), isic_manifest()
    ham.to_csv(ROOT / "manifest_ham10000_384.csv", index=False)
    isic.to_csv(ROOT / "manifest_isic2020_384.csv", index=False)

    lines = [
        f"Cache: {SIZE} x {SIZE}, bicubic, JPEG quality {QUALITY}",
        f"HAM10000: {len(ham)} images, {int(ham['label'].sum())} melanomas, "
        f"{ham['lesion_group'].nunique()} lesions (expected 10,015 / 1,113 / 7,470); "
        f"cached {h_ok}, failed {len(h_fail)}",
        f"ISIC 2020: {len(isic)} images, {int(isic['label'].sum())} melanomas "
        f"(expected 32,701 / 581 after removing the 425 listed duplicates); "
        f"cached {i_ok}, failed {len(i_fail)}",
        compare_with_release(ham, REPO / "results" / "manifests" / "manifest_ham10000_384.csv", "HAM10000"),
        compare_with_release(isic, REPO / "results" / "manifests" / "manifest_isic2020_384.csv", "ISIC 2020"),
    ]
    for name, note in (h_fail + i_fail)[:50]:
        lines.append(f"  failed: {name}: {note}")
    (ROOT / "prepare_data_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
