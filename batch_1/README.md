# melanoma-provenance-audit

Code, results and user manual for the paper

> **Re-hosted benchmark images in public dermoscopy datasets: a provenance audit and its consequences for melanoma classification**
> Sibghatullah I. Khan, Sreenidhi Institute of Science and Technology, Hyderabad, India

The repository contains two things:

1. **A provenance audit tool** (`code/mirror_audit.py`) that checks any folder of dermoscopy images against reference collections (here HAM10000 and ISIC 2020) and reports which images are copies, where they sit in the folder's own train/test split, how the folder's classes relate to the sources of its images, and how well the source alone predicts the class.
2. **The scripts for every experiment in the paper**, with the result files they produced, so that each table and figure can be checked or regenerated.

No images are distributed here. All datasets must be downloaded from their original providers (see `data/README.md`).

---

## 1. What the tool does

For every image in a mirror, the tool computes a 256-bit perceptual hash (DCT-based, 16 x 16) in eight orientations (four rotations by 90 degrees, each with and without a flip) and finds the nearest reference image by Hamming distance, keeping the smallest distance over the eight orientations. An image counts as a copy when that distance is 10 bits or less.

It then reports, for each mirror:

| Output | File in `results/results_mirror_audit/` |
|---|---|
| Overlap with each reference set, per split, at 0, 5, 10 and 15 bits | `overlap_summary.csv` |
| Class folder against the diagnosis of the matched reference image | `label_crosstab.csv` |
| Test images with a copy in the mirror's own training split | `internal_leakage.csv` |
| Share of each class coming from each source | `source_composition.csv` |
| Test accuracy of a rule that predicts the class from the source alone | `provenance_rule.csv` |
| Distinct reference images and melanomas each mirror holds | `reference_coverage.csv` |
| Overlap between the mirrors themselves | `mirror_vs_mirror.csv` |
| False-match rate of the null comparison | `null_calibration.csv` |
| Nearest-neighbour distance for every image | `matches_<mirror>.csv` |

A null comparison (ISIC 2020 against HAM10000, two collections that share no images) gives the false-match rate, and `code/tool_validation.py` gives the detection rate for 23 known transformations.

**What it detects and what it misses.** In our validation the tool found more than 99% of copies that were resized, recompressed, rotated by 90-degree steps or flipped, and 90.7% to 99.9% of copies with moderate colour changes or blur. It did not detect copies that were cropped (even by 5%) or rotated by other angles. Counts are therefore lower bounds.

### Using the tool on your own mirror

Add an entry to the `MIRRORS` dictionary near the top of `code/mirror_audit.py`:

```python
MIRRORS = {
    "MyMirror": ROOT / "my_mirror_folder",
    ...
}
```

The folder may use any layout. The split (`train`, `test`, `val`, `valid`, `validation`) is read from the folder names in each image path, and the class is the name of the folder that directly contains the image. Run the script as in Step 1 below; reference hashes are cached, so only the new mirror is hashed.

---

## 2. Requirements

* Python 3.10 or later (the results were produced with the versions listed in `results/environment.txt`)
* A GPU is optional. Training ran on an Apple silicon laptop (PyTorch MPS backend); CUDA and CPU also work, more slowly on CPU.
* About 3 GB of disk for the image cache.

```bash
git clone https://github.com/sibghatece/melanoma-provenance-audit.git
cd melanoma-provenance-audit
python3 -m pip install -r requirements.txt
```

All scripts read the data folder from the environment variable `MELANOMA_ROOT`. If it is not set, they use the `data/` folder of this repository.

```bash
export MELANOMA_ROOT=/path/to/data          # folder laid out as in data/README.md
export KAGGLE_ROOT=$MELANOMA_ROOT/melanoma-skin-cancer   # the Fanconi mirror (default)
```

---

## 3. Running the experiments

Run the steps in order. Each step writes its outputs into `$MELANOMA_ROOT` and is resumable: completed folds, cached hashes and cached features are skipped on a rerun. On macOS, prefix long runs with `caffeinate -dimsu` so the laptop does not sleep; `python3 -u ... | tee log.txt` keeps a live log.

| Step | Command | Produces | Paper | Run time on our laptop |
|---|---|---|---|---|
| 0 | `python3 code/prepare_data.py` | 384-px cache, two manifests | Methods | 45 to 90 min |
| 1 | `python3 code/mirror_audit.py` | `results_mirror_audit/` | Tables 1 to 3, Figs. 2 and 4 | 30 to 90 min (first run; hashes are cached) |
| 2 | `python3 code/tool_validation.py` | `results_tool_validation/` | Fig. 3 | about 10 min |
| 3 | `python3 code/run_pipeline_v3.py` | `results_v3/` (lesion-grouped run, five fold models) | Table 5, Fig. 6 | 14 to 20 h |
| 4 | `python3 code/recompute_v3_thresholds.py` | corrected operating points in `results_v3/` | Methods | seconds |
| 5 | `python3 code/run_leakage_magnitude.py --split image` | `results_split_image/` (image-wise run) | Table 5, Fig. 6 | 5 to 7 h |
| 6 | `python3 code/analyse_split_comparison.py` | `results_split_comparison/` | Table 5, Fig. 6 | minutes |
| 7 | `python3 code/run_extra_seeds.py` | `results_seeds/` (seeds 7 and 123) | Table 6 | 26 to 28 h |
| 8 | `python3 code/kaggle_seen_unseen.py` | `results_kaggle_twins/` (Experiment A) | Table 4, Fig. 5 | not timed (one feature-extraction pass with each fold model) |
| 9 | `python3 code/shortcut_experiment.py` | `results_shortcut/` (Experiment C) | Table 7, Fig. 7 | not timed (one feature-extraction pass) |
| 10 | `python3 code/shortcut_followup.py` | patient-clean and image-size checks | Table 7, Fig. 7 | minutes |
| 11 | `python3 code/controlled_confounding.py` | `results_confounding/` (Experiment D) | Table 8, Fig. 8 | 30 to 45 min |
| 12 | `python3 code/make_figures.py` | `figures_jiim/Fig1` to `Fig8` (.svg, .eps, .tif) | Figs. 1 to 8 | about 1 min |

Steps 1 and 2 need no GPU. Step 8 needs the fold models from Step 3. Steps 9 to 11 use ImageNet weights only and can run without Steps 3 to 7.

### Checkpoints

Your numbers should match these. Training steps can differ slightly between hardware and library versions; the audit steps are deterministic.

* **Step 0:** HAM10000 10,015 images, 1,113 melanomas, 7,470 lesions; ISIC 2020 32,701 images, 581 melanomas. If `results/manifests/` is present, `prepare_data_report.txt` should say IDENTICAL for both.
* **Step 1:** null comparison, 0 of 32,701 ISIC 2020 images within 10 bits of HAM10000 (median nearest distance 86 bits). Fanconi mirror 75.5% HAM10000 copies; Javid mirror 57.2% ISIC 2020 copies; provenance-only rule 91.4% (Javid) and 99.4% (ISIC 2019 and 2020).
* **Step 2:** 99.3% to 99.9% detection for resizing, recompression, 90-degree rotations and flips; 0% for crops and free rotations.
* **Step 3:** HAM10000 out-of-fold AUROC 0.918 and ISIC 2020 AUROC 0.720 for XGBoost on all features.
* **Steps 5 to 7:** internal AUROC gain of image-wise over lesion-grouped splitting 0.034 ± 0.011 over three seeds; about 60% of melanoma test images have another image of the same lesion in training.
* **Step 8:** AUROC 0.905 (twin never seen) against 0.987 (twin seen); paired difference 0.082 (95% CI 0.068 to 0.095).
* **Step 9:** Javid mirror, AUROC 0.976 on the whole test folder against 0.860 within the ISIC 2020 source; source classifier 0.998.
* **Step 11:** confounded model 0.995 on the confounded test set, 0.733 within ISIC 2020; control 0.856.

---

## 4. Results included in this repository

`results/` holds the files our runs produced (tables as CSV, logs, saved prediction files that the analysis scripts read, figures, and the manifests reduced to file name, label and lesion group). Steps 4, 6 and 12 can be rerun directly from these files without retraining. Model weights, cached features and images are not included.

`results/environment.txt` lists the Python and package versions used.

`tools/collect_results.py` is the script that filled `results/` from our data folder, and `tools/make_esm.py` builds the anonymised copy of this repository submitted to the journal as Online Resource 1.

---

## 5. Data and licences

The datasets are not redistributed. HAM10000 and ISIC 2020 are released under CC BY-NC 4.0 by their providers; the Kaggle mirrors are subject to the terms set by their uploaders. Download links and the expected folder layout are in `data/README.md`.

The code is released under the MIT licence (`LICENSE`).

---

## 6. Citation

If you use the tool or the code, please cite the paper (details will be added on publication) and this repository (`CITATION.cff`).

## 7. Contact

Sibghatullah I. Khan, sibghatikhan@gmail.com, ORCID 0000-0003-1263-8100
