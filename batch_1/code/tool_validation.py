"""
Tool validation: how often does the provenance audit find a copy after each
kind of change that re-uploaders make?

A random sample of HAM10000 images is transformed in known ways (resizing,
recompression, flips, rotations, crops, colour changes, blur). Each
transformed image is audited exactly as a mirror image is audited in
mirror_audit.py: 256-bit perceptual hash in 8 orientations, nearest
neighbour over the full HAM10000 reference set, minimum Hamming distance.
A copy counts as detected when its nearest reference image is its true
source and the distance is at or below the threshold (10 bits in the paper;
0, 5 and 15 are also reported). Together with the null comparison
(0 of 32,701 ISIC 2020 images matched HAM10000) this gives the tool's
sensitivity per transformation and its false-match rate.

It also times the audit (hashes per second, search time per 1,000 queries
against 10,015 references), which the paper can report as implementation
cost.

Reference hashes are read from the mirror_audit.py cache when present,
otherwise computed.

Run:
  export MELANOMA_ROOT=/path/to/data
  python3 tool_validation.py            # 1,000 sampled images, about 10-20 min
  python3 tool_validation.py --n 300    # quicker trial
Outputs in $MELANOMA_ROOT/results_tool_validation/
  per_image_distances.csv, sensitivity_by_transform.csv, timing.csv
"""

import argparse
import io
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

# MELANOMA_ROOT if set, otherwise the data/ folder of this repository
ROOT = Path(os.environ.get("MELANOMA_ROOT", Path(__file__).resolve().parents[1] / "data")).expanduser()
OUT = ROOT / "results_tool_validation"
CACHE = ROOT / "results_mirror_audit" / "cache" / "ref_ham10000_d1.npz"
HASH_SIZE = 16
THRESHOLDS = [0, 5, 10, 15]
PRIMARY = 10
SEED = 42
N_WORKERS = max(1, (os.cpu_count() or 2) - 1)

ORIENTS = [None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180,
           Image.Transpose.ROTATE_270, Image.Transpose.FLIP_LEFT_RIGHT,
           Image.Transpose.FLIP_TOP_BOTTOM, Image.Transpose.TRANSPOSE,
           Image.Transpose.TRANSVERSE]


# ------------------------------------------------------------------ transformations
def jpeg(im, q):
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def center_crop(im, frac):
    w, h = im.size
    cw, ch = int(w * frac), int(h * frac)
    left, top = (w - cw) // 2, (h - ch) // 2
    return im.crop((left, top, left + cw, top + ch))


def rotate_any(im, deg):
    return im.rotate(deg, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=(0, 0, 0))


TRANSFORMS = [
    ("Group", "Transformation", None),
    ("Resizing and recompression", "Re-encoded, JPEG quality 95", lambda im: jpeg(im, 95)),
    ("Resizing and recompression", "Resized to 224 x 224", lambda im: im.resize((224, 224), Image.Resampling.BILINEAR)),
    ("Resizing and recompression", "Resized to 224 x 224, JPEG quality 75", lambda im: jpeg(im.resize((224, 224), Image.Resampling.BILINEAR), 75)),
    ("Resizing and recompression", "JPEG quality 50", lambda im: jpeg(im, 50)),
    ("Resizing and recompression", "Resized to 128 x 128", lambda im: im.resize((128, 128), Image.Resampling.BILINEAR)),
    ("Orientation", "Rotated 90 degrees", lambda im: im.transpose(Image.Transpose.ROTATE_90)),
    ("Orientation", "Rotated 180 degrees", lambda im: im.transpose(Image.Transpose.ROTATE_180)),
    ("Orientation", "Horizontal flip", lambda im: im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)),
    ("Orientation", "Vertical flip", lambda im: im.transpose(Image.Transpose.FLIP_TOP_BOTTOM)),
    ("Arbitrary rotation", "Rotated 10 degrees", lambda im: rotate_any(im, 10)),
    ("Arbitrary rotation", "Rotated 30 degrees", lambda im: rotate_any(im, 30)),
    ("Arbitrary rotation", "Rotated 45 degrees", lambda im: rotate_any(im, 45)),
    ("Cropping", "Centre crop 95%", lambda im: center_crop(im, 0.95)),
    ("Cropping", "Centre crop 90%", lambda im: center_crop(im, 0.90)),
    ("Cropping", "Centre crop 80%", lambda im: center_crop(im, 0.80)),
    ("Cropping", "Centre crop 70%", lambda im: center_crop(im, 0.70)),
    ("Colour and blur", "Brightness x1.2", lambda im: ImageEnhance.Brightness(im).enhance(1.2)),
    ("Colour and blur", "Contrast x1.2", lambda im: ImageEnhance.Contrast(im).enhance(1.2)),
    ("Colour and blur", "Saturation x1.3", lambda im: ImageEnhance.Color(im).enhance(1.3)),
    ("Colour and blur", "Gaussian blur, radius 1", lambda im: im.filter(ImageFilter.GaussianBlur(1))),
    ("Colour and blur", "Gaussian blur, radius 2", lambda im: im.filter(ImageFilter.GaussianBlur(2))),
    ("Combined", "Resize 224 + horizontal flip + JPEG 75", lambda im: jpeg(im.resize((224, 224), Image.Resampling.BILINEAR).transpose(Image.Transpose.FLIP_LEFT_RIGHT), 75)),
    ("Combined", "Crop 90% + brightness x1.1 + resize 224", lambda im: ImageEnhance.Brightness(center_crop(im, 0.9)).enhance(1.1).resize((224, 224), Image.Resampling.BILINEAR)),
]
TRANSFORMS = [t for t in TRANSFORMS if t[2] is not None]


# ------------------------------------------------------------------ hashing (same as mirror_audit.py)
def phash_bits(im):
    import imagehash
    return imagehash.phash(im, hash_size=HASH_SIZE).hash.flatten().astype(np.uint8)


def _ref_hash(path):
    try:
        return phash_bits(Image.open(path).convert("RGB"))
    except Exception:
        return None


def _query_hashes(args):
    """All transformed versions of one image, each hashed in 8 orientations."""
    path = args
    img = Image.open(path).convert("RGB")
    out = []
    for _, _, fn in TRANSFORMS:
        t = fn(img)
        out.append(np.stack([phash_bits(t if o is None else t.transpose(o)) for o in ORIENTS]))
    return np.stack(out)          # [n_transforms, 8, 256]


def reference_hashes(paths):
    if CACHE.exists():
        with np.load(CACHE, allow_pickle=False) as z:
            ok, H = z["ok"], z["H"]
            cpaths = z["paths"] if "paths" in z.files else None
        if ok.all() and (cpaths is None or list(map(str, cpaths)) == list(paths)) and len(H) == len(paths):
            print(f"  reference hashes loaded from {CACHE.name}")
            return H[:, 0].astype(np.float32)
    print(f"  hashing {len(paths)} reference images ({N_WORKERS} workers)")
    with ProcessPoolExecutor(N_WORKERS) as ex:
        res = list(ex.map(_ref_hash, paths, chunksize=64))
    if any(r is None for r in res):
        raise SystemExit("Some reference images could not be opened.")
    return np.stack(res).astype(np.float32)


def nn_search(Q, R, chunk=512):
    """Q [n, 8, 256], R [m, 256] -> nearest index and min distance over orientations."""
    r_sum = R.sum(axis=1)[None, :]
    idx = np.empty(len(Q), np.int64)
    dist = np.empty(len(Q), np.float32)
    for s in range(0, len(Q), chunk):
        best = None
        for o in range(Q.shape[1]):
            q = Q[s:s + chunk, o]
            D = q.sum(axis=1)[:, None] + r_sum - 2.0 * (q @ R.T)
            best = D if best is None else np.minimum(best, D)
        idx[s:s + chunk] = best.argmin(axis=1)
        dist[s:s + chunk] = best.min(axis=1)
    return idx, dist


# ------------------------------------------------------------------ main
def main():
    print(f"[Data folder] {ROOT}")
    if not (ROOT / "manifest_ham10000_384.csv").exists():
        raise SystemExit(f"manifest_ham10000_384.csv not found in {ROOT}. Run code/prepare_data.py "
                         "first, or set MELANOMA_ROOT to your data folder.")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ham = pd.read_csv(ROOT / "manifest_ham10000_384.csv")
    paths = ham["image_path"].tolist()

    t0 = time.time()
    R = reference_hashes(paths)
    t_ref = time.time() - t0

    rng = np.random.default_rng(SEED)
    sample = np.sort(rng.choice(len(paths), size=min(args.n, len(paths)), replace=False))
    print(f"  transforming and hashing {len(sample)} images x {len(TRANSFORMS)} transformations "
          f"x 8 orientations ({N_WORKERS} workers)")
    t0 = time.time()
    with ProcessPoolExecutor(N_WORKERS) as ex:
        Qs = list(ex.map(_query_hashes, [paths[i] for i in sample], chunksize=8))
    t_hash = time.time() - t0
    Q = np.stack(Qs).astype(np.float32)        # [n, n_transforms, 8, 256]
    n_hashes = Q.shape[0] * Q.shape[1] * Q.shape[2]

    rows, per_image = [], []
    t_search = 0.0
    for j, (group, name, _) in enumerate(TRANSFORMS):
        t0 = time.time()
        idx, dist = nn_search(Q[:, j], R)
        t_search += time.time() - t0
        correct = idx == sample
        # distance to the true source image, whether or not it was the nearest
        q, r = Q[:, j], R[sample][:, None, :]
        d_true = (q.sum(axis=2) + r.sum(axis=2) - 2.0 * (q * r).sum(axis=2)).min(axis=1)
        row = {"group": group, "transformation": name, "n": len(sample),
               "median_distance_to_source": float(np.median(d_true)),
               "p90_distance_to_source": float(np.percentile(d_true, 90))}
        for t in THRESHOLDS:
            row[f"sensitivity_d<={t}"] = 100.0 * np.mean(correct & (dist <= t))
        row["wrong_nearest_within_10"] = int(np.sum(~correct & (dist <= PRIMARY)))
        rows.append(row)
        per_image.append(pd.DataFrame({"transformation": name, "ham_index": sample,
                                       "nearest_index": idx, "nearest_distance": dist,
                                       "distance_to_true_source": d_true}))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "sensitivity_by_transform.csv", index=False)
    pd.concat(per_image).to_csv(OUT / "per_image_distances.csv", index=False)

    n_queries = len(sample) * len(TRANSFORMS)
    timing = pd.DataFrame([{
        "reference_images": len(paths), "reference_hash_seconds": t_ref,
        "query_hashes": n_hashes, "query_hash_seconds": t_hash,
        "hashes_per_second": n_hashes / t_hash,
        "search_seconds_per_1000_queries": 1000 * t_search / n_queries,
        "workers": N_WORKERS}])
    timing.to_csv(OUT / "timing.csv", index=False)

    pd.set_option("display.width", 200)
    show = res[["group", "transformation", "median_distance_to_source", "p90_distance_to_source",
                "sensitivity_d<=0", "sensitivity_d<=5", "sensitivity_d<=10",
                "sensitivity_d<=15", "wrong_nearest_within_10"]]
    print("\n=== Sensitivity of the audit by transformation (% of sampled images detected) ===")
    print(show.round(1).to_string(index=False))
    print("\n=== Timing ===")
    print(timing.round(2).to_string(index=False))
    print(f"\nAll outputs in {OUT}")


if __name__ == "__main__":
    main()
