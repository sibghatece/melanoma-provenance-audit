"""
Copies the result files from your data folder into results/ of this
repository, so that the published repository contains the actual numbers
behind every table and figure. Run it once, before the first commit.

What is copied
  * every .csv, .txt and .json file inside the results_* folders and
    figures_jiim/ (figures as .svg/.eps/.tif)
  * saved prediction files (.npz) under results_*/probs/ and
    results_seeds/*/, which the analysis scripts read, if each is < 50 MB
  * the two manifests, reduced to file name, label, lesion group and
    diagnosis (no paths)
  * top-level *_log.txt run logs
What is never copied
  * images of any kind (.jpg, .png ...), the image caches, model weights
    (.pt), cached features (.npy) and hash caches
Every copied text file has your local folder paths replaced by
$MELANOMA_ROOT / $KAGGLE_ROOT, so no personal path ends up on GitHub.

It also writes results/environment.txt with the Python and package
versions used.

Run:
  export MELANOMA_ROOT=/path/to/data
  python3 tools/collect_results.py
"""

import os
import platform
import shutil
import sys
from importlib import metadata
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("MELANOMA_ROOT", REPO / "data")).expanduser().resolve()
KAGGLE_ROOT = Path(os.environ.get("KAGGLE_ROOT", ROOT / "melanoma-skin-cancer")).expanduser().resolve()
OUT = REPO / "results"

TEXT = {".csv", ".txt", ".json"}
FIG = {".svg", ".eps", ".tif"}
NEVER = {".jpg", ".jpeg", ".png", ".bmp", ".pt", ".pth", ".npy"}
MAX_NPZ = 50 * 1024 * 1024
PACKAGES = ["numpy", "pandas", "scikit-learn", "torch", "torchvision", "xgboost",
            "shap", "ImageHash", "Pillow", "matplotlib"]


def scrub(text):
    """Replaces local paths with placeholders."""
    for path, tag in [(str(KAGGLE_ROOT), "$KAGGLE_ROOT"), (str(ROOT), "$MELANOMA_ROOT"),
                      (str(Path.home()), "~")]:
        text = text.replace(path, tag)
    return text


def copy_text(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(scrub(src.read_text(errors="replace")))


def main():
    print(f"[Data folder] {ROOT}")
    if not (ROOT / "manifest_ham10000_384.csv").exists():
        sys.exit("manifest_ham10000_384.csv not found; set MELANOMA_ROOT to your data folder.")
    n_text = n_fig = n_npz = skipped = 0

    for folder in sorted(ROOT.glob("results_*")) + [ROOT / "figures_jiim"]:
        if not folder.is_dir():
            continue
        for src in folder.rglob("*"):
            if not src.is_file() or "cache" in src.parts:
                continue
            ext = src.suffix.lower()
            dst = OUT / src.relative_to(ROOT)
            if ext in NEVER:
                continue
            if ext in TEXT:
                copy_text(src, dst)
                n_text += 1
            elif ext in FIG and folder.name == "figures_jiim":
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                n_fig += 1
            elif ext == ".npz" and ("probs" in src.parts or folder.name == "results_seeds"):
                if src.stat().st_size < MAX_NPZ:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    n_npz += 1
                else:
                    skipped += 1

    for log in ROOT.glob("*_log.txt"):
        copy_text(log, OUT / "logs" / log.name)
        n_text += 1

    (OUT / "manifests").mkdir(parents=True, exist_ok=True)
    for name, extra in [("manifest_ham10000_384.csv", "dx"), ("manifest_isic2020_384.csv", "diagnosis")]:
        df = pd.read_csv(ROOT / name)
        cols = ["file", "label", "lesion_group"] + ([extra] if extra in df.columns else [])
        df.assign(file=df["image_path"].map(lambda p: Path(p).name))[cols].to_csv(
            OUT / "manifests" / name, index=False)

    lines = [f"Python {platform.python_version()} on {platform.system()} {platform.machine()}"]
    for p in PACKAGES:
        try:
            lines.append(f"{p}=={metadata.version(p)}")
        except metadata.PackageNotFoundError:
            lines.append(f"{p}: not installed")
    (OUT / "environment.txt").write_text("\n".join(lines) + "\n")

    leftover = [f for f in OUT.rglob("*") if f.is_file() and f.suffix in TEXT
                and ("/Users/" in f.read_text(errors="ignore") or "\\Users\\" in f.read_text(errors="ignore"))]
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"Copied {n_text} text files, {n_fig} figure files, {n_npz} prediction files "
          f"({skipped} prediction files over 50 MB skipped). results/ is now {size:.1f} MB.")
    print("Files still containing a local path:", [str(f.relative_to(REPO)) for f in leftover] or "none")


if __name__ == "__main__":
    main()
