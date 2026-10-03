"""
Builds ESM_1.zip (Online Resource 1) from this repository for submission.

JIIM uses double-blind review, so this copy is anonymised: author names,
contact details, ORCID and the GitHub address are removed from the README,
the licence names "The authors", and CITATION.cff is left out. Code, data
instructions and results are identical to the repository.

Run after tools/collect_results.py:
  python3 tools/make_esm.py            -> ESM_1.zip (blinded, for review)
  python3 tools/make_esm.py --full     -> ESM_1_full.zip (with author details,
                                          for the final accepted version)
"""

import argparse
import re
import shutil
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
JOURNAL = "Journal of Imaging Informatics in Medicine"
SKIP = {".git", "__pycache__", ".DS_Store", "data"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    full = ap.parse_args().full
    name = "ESM_1_full" if full else "ESM_1"
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / name
        shutil.copytree(REPO, dst, ignore=lambda d, files: [f for f in files if f in SKIP or f.endswith(".zip")])
        (dst / "data").mkdir()
        shutil.copy2(REPO / "data" / "README.md", dst / "data" / "README.md")
        readme = (dst / "README.md").read_text()
        header = f"Online Resource 1. {JOURNAL}.\n\n"
        if not full:
            readme = re.sub(r"\n> Sibghatullah I\. Khan[^\n]*\n", "\n", readme)
            readme = re.sub(r"\n## 7\. Contact\n.*", "\n", readme, flags=re.S)
            readme = readme.replace("git clone https://github.com/sibghatece/melanoma-provenance-audit.git\n"
                                    "cd melanoma-provenance-audit\n", "cd ESM_1\n")
            readme = readme.replace("and this repository (`CITATION.cff`)", "")
            (dst / "LICENSE").write_text((dst / "LICENSE").read_text().replace("Sibghatullah I. Khan", "The authors"))
            (dst / "CITATION.cff").unlink(missing_ok=True)
            for f in [dst / "tools" / "make_esm.py"]:
                f.unlink(missing_ok=True)
        (dst / "README.md").write_text(header + readme)
        leaks = [str(f.relative_to(dst)) for f in dst.rglob("*") if f.is_file() and f.suffix in {".md", ".py", ".txt", ".cff", ".csv"}
                 and not full and re.search(r"Khan|sibghat|0000-0003-1263-8100|Sreenidhi", f.read_text(errors="ignore"))]
        out = REPO / f"{name}.zip"
        shutil.make_archive(str(out.with_suffix("")), "zip", tmp, name)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    if not full:
        print("Identifying text left in the blinded copy:", leaks or "none")


if __name__ == "__main__":
    main()
