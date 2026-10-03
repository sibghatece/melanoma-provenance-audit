"""
Generates the eight figures of the JIIM manuscript from the saved result files.

  Fig1  Block diagram of the study methodology (no data)
  Fig2  Nearest-neighbour Hamming distances, null comparison and mirrors
        <- results_mirror_audit/cache/ref_*_d1.npz, results_mirror_audit/matches_*.csv
  Fig3  Detection rate of the audit by transformation
        <- results_tool_validation/sensitivity_by_transform.csv
  Fig4  Source composition of the five mirrors by class
        <- results_mirror_audit/source_composition.csv
  Fig5  ROC curves on identical Kaggle images, twin seen vs never seen
        <- results_kaggle_twins/scores_per_image.csv
  Fig6  Paired AUROC differences, image-wise minus lesion-grouped CV
        <- results_split_comparison/paired_bootstrap_deltas.csv
  Fig7  Source shortcut in the Javid mirror
        <- results_shortcut/shortcut_results.csv, patient_overlap_results.csv
  Fig8  Controlled source confounding
        <- results_confounding/confounding_per_repeat.csv

Each figure is written as .svg (editable), .eps (JIIM's preferred vector
format, fonts embedded) and .tif (600 dpi, RGB). Widths follow JIIM:
174 mm for full-width figures, 129 mm for Fig4. Lettering is Arial 8-9 pt,
there are no titles inside the figures, and colours are always paired with
hatching or marker shapes so the figures stay readable in greyscale.

Run:
  export MELANOMA_ROOT=/path/to/data
  python3 make_figures.py
Output: $MELANOMA_ROOT/figures_jiim/Fig1..Fig8 .svg/.eps/.tif
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

# MELANOMA_ROOT if set, otherwise the data/ folder of this repository
ROOT = Path(os.environ.get("MELANOMA_ROOT", Path(__file__).resolve().parents[1] / "data")).expanduser()
OUT = ROOT / "figures_jiim"
MM = 1 / 25.4
FULL, MID = 174 * MM, 129 * MM
SEED = 42

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2,
    "svg.fonttype": "none", "ps.fonttype": 42, "pdf.fonttype": 42,
})

MIRROR_NAMES = {
    "Fanconi_3297": "Fanconi",
    "Javid_10605": "Javid",
    "SkinCancerISIC_2357": "Skin Cancer ISIC",
    "ISIC19_20_malig_benign_11400": "ISIC 2019 and 2020",
    "HAM_augmented_balanced": "Balanced HAM10000",
}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.eps", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.tif", dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"  wrote {name}.svg / .eps / .tif")


# ------------------------------------------------------------------ Fig 1
def box(ax, x, y, w, h, text, fill="#FFFFFF", bold=False, lw=0.8, fs=7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                linewidth=lw, edgecolor="black", facecolor=fill))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="black",
                                shrinkA=0, shrinkB=0, mutation_scale=8))


def fig1():
    """Block diagram of the whole study methodology."""
    fig, ax = plt.subplots(figsize=(FULL, 6.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    dark, mid, light, white = "#C8C8C8", "#DDDDDD", "#F2F2F2", "#FFFFFF"
    fs = 6.8

    def band(y0, y1, label):
        ax.add_patch(FancyBboxPatch((0.0, y0), 1.0, y1 - y0, boxstyle="round,pad=0.004,rounding_size=0.01",
                                    linewidth=0.6, edgecolor="grey", facecolor="none", linestyle="--"))
        ax.text(0.01, y1 - 0.008, label, fontsize=7.5, fontweight="bold", va="top")

    def line(x0, y0, x1, y1):
        ax.plot([x0, x1], [y0, y1], color="black", lw=0.8)

    # ---- data
    band(0.82, 0.995, "Data")
    box(ax, 0.06, 0.84, 0.25, 0.10, "HAM10000\n10,015 images, 7,470 lesions\n1,113 melanomas", light, fs=fs)
    box(ax, 0.375, 0.84, 0.25, 0.10, "ISIC 2020\n32,701 images, 2,056 patients\n581 melanomas", light, fs=fs)
    box(ax, 0.69, 0.84, 0.25, 0.10, "Five Kaggle mirrors\n67,166 images\nown training and test folders", light, fs=fs)

    # ---- provenance audit tool
    band(0.425, 0.805, "Provenance audit tool")
    box(ax, 0.06, 0.685, 0.565, 0.065, "Perceptual hash of reference images (256 bits)", white, fs=fs)
    box(ax, 0.69, 0.685, 0.25, 0.065, "Perceptual hash of mirror\nimages in 8 orientations", white, fs=fs)
    arrow(ax, 0.185, 0.84, 0.185, 0.75)
    arrow(ax, 0.50, 0.84, 0.50, 0.75)
    arrow(ax, 0.815, 0.84, 0.815, 0.75)
    box(ax, 0.32, 0.565, 0.36, 0.075, "Nearest-neighbour search\nmatch if Hamming distance \u2264 10", mid, fs=fs)
    arrow(ax, 0.40, 0.685, 0.42, 0.64)
    arrow(ax, 0.815, 0.685, 0.62, 0.64)
    box(ax, 0.03, 0.565, 0.25, 0.075, "Validation: null comparison,\nISIC 2020 against HAM10000", white, fs=fs)
    box(ax, 0.72, 0.565, 0.25, 0.075, "Validation: 23 known\ntransformations, 1,000 images", white, fs=fs)
    outs = ["Overlap per split", "Class against source", "Internal leakage and\npatient overlap", "Provenance-only rule"]
    xs = [0.03, 0.27, 0.51, 0.75]
    for x0, o in zip(xs, outs):
        box(ax, x0, 0.445, 0.22, 0.07, o, white, fs=fs)
        arrow(ax, 0.50, 0.565, x0 + 0.11, 0.515)

    # ---- consequences
    band(0.015, 0.395, "Measuring the consequences")
    arrow(ax, 0.50, 0.445, 0.50, 0.355)
    ax.text(0.515, 0.372, "matches, sources and splits", fontsize=6.5, va="center")
    box(ax, 0.03, 0.27, 0.45, 0.075, "EfficientNet-B0 fine-tuned on HAM10000,\nmulti-stage features and shallow classifiers", dark, fs=fs)
    box(ax, 0.52, 0.27, 0.45, 0.075, "EfficientNet-B0 with ImageNet weights only,\nmulti-stage features and XGBoost", dark, fs=fs)
    exps = ["A  Paired twin test\ntwin seen vs\nnever seen",
            "B  Image-wise vs\nlesion-grouped\nsplits, 3 seeds",
            "C  Source shortcut\nin two mirrors",
            "D  Controlled\nconfounding,\n5 repetitions"]
    ex = [0.03, 0.265, 0.52, 0.755]
    for x0, txt in zip(ex, exps):
        box(ax, x0, 0.135, 0.215, 0.10, txt, light, fs=fs)
    for x0 in ex:
        arrow(ax, x0 + 0.1075, 0.27, x0 + 0.1075, 0.235)
    # join experiments into the recommendations
    for x0 in ex:
        line(x0 + 0.1075, 0.135, x0 + 0.1075, 0.115)
    line(ex[0] + 0.1075, 0.115, ex[-1] + 0.1075, 0.115)
    arrow(ax, 0.50, 0.115, 0.50, 0.09)
    box(ax, 0.30, 0.035, 0.40, 0.055, "Recommended provenance checks", white, bold=True, fs=fs)
    save(fig, "Fig1")


# ------------------------------------------------------------------ Fig 4
def fig4():
    comp = pd.read_csv(ROOT / "results_mirror_audit" / "source_composition.csv")
    for col in ["HAM10000", "ISIC2020", "unmatched"]:
        if col not in comp.columns:
            comp[col] = 0
    rows = []
    for mirror, name in MIRROR_NAMES.items():
        sub = comp[comp["mirror"] == mirror]
        if sub.empty:
            print(f"  (Fig4: {mirror} not found in source_composition.csv; skipped)")
            continue
        classes = sorted(sub["class"].str.lower().unique())
        if set(classes) == {"benign", "malignant"}:
            for cl in ["benign", "malignant"]:
                s = sub[sub["class"].str.lower() == cl][["HAM10000", "ISIC2020", "unmatched"]].sum()
                rows.append((f"{name}, {cl}", s))
        else:
            s = sub[["HAM10000", "ISIC2020", "unmatched"]].sum()
            rows.append((f"{name}, all {len(classes)} classes", s))
    labels = [r[0] for r in rows][::-1]
    data = np.array([r[1].values for r in rows], dtype=float)[::-1]
    n = data.sum(axis=1)
    pct = 100 * data / n[:, None]

    fig, ax = plt.subplots(figsize=(FULL, 0.28 * len(rows) + 0.9))
    styles = [("HAM10000 copy", "#4D4D4D", "////"),
              ("ISIC 2020 copy", "#A6A6A6", "...."),
              ("Unmatched", "#FFFFFF", "")]
    left = np.zeros(len(labels))
    for j, (lab, col, hatch) in enumerate(styles):
        ax.barh(labels, pct[:, j], left=left, color=col, hatch=hatch, edgecolor="black",
                linewidth=0.5, height=0.62, label=lab)
        left += pct[:, j]
    for i, total in enumerate(n):
        ax.text(101.5, i, f"n = {int(total):,}", va="center", fontsize=7)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of images (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False,
              handlelength=1.8)
    plt.rcParams["hatch.linewidth"] = 0.5
    save(fig, "Fig4")


# ------------------------------------------------------------------ Fig 5
def fig5():
    d = pd.read_csv(ROOT / "results_kaggle_twins" / "scores_per_image.csv")
    st = d[[f"status_fold{i}" for i in range(1, 6)]].values
    P = d[[f"p_fold{i}" for i in range(1, 6)]].values
    seen = st == "train"
    keep = seen.any(axis=1)
    y = d["ham_label_melanoma"].values[keep]
    p_unseen = d["p_twin_never_seen"].values[keep]
    rng = np.random.default_rng(SEED)
    Pk, Sk = P[keep], seen[keep]
    choices = [np.where(Sk[i])[0] for i in range(len(y))]
    # 200 random single-model draws, as in the paper; plot the draw whose
    # AUROC is closest to the median so the figure matches the reported value
    draws = []
    for _ in range(200):
        ps = np.array([Pk[i, rng.choice(c)] for i, c in enumerate(choices)])
        draws.append((roc_auc_score(y, ps), ps))
    med = np.median([a for a, _ in draws])
    p_seen = min(draws, key=lambda t: abs(t[0] - med))[1]

    fig, ax = plt.subplots(figsize=(MID, MID * 0.82))
    for p, lab, ls, mk in [(p_seen, "Twin seen in training", "-", "o"),
                           (p_unseen, "Twin never seen", "--", "s")]:
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, ls, color="black", label=f"{lab} (AUROC {auc:.3f})",
                marker=mk, markevery=max(len(fpr) // 12, 1), markersize=3.5,
                markerfacecolor="white", markeredgewidth=0.7)
    ax.axvline(0.10, color="grey", lw=0.7, ls=":")
    ax.text(0.115, 0.05, "90% specificity", fontsize=7, color="dimgrey")
    ax.plot([0, 1], [0, 1], color="lightgrey", lw=0.7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("1 \u2212 specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    print(f"  Fig5: n={len(y)} images, {int(y.sum())} melanomas "
          f"(seen curve = the draw closest to the median of 200, median AUROC {med:.3f})")
    save(fig, "Fig5")


# ------------------------------------------------------------------ Fig 6
def fig6():
    r = pd.read_csv(ROOT / "results_split_comparison" / "paired_bootstrap_deltas.csv",
                    dtype={"k": str})
    clf_order = ["XGBoost", "RandomForest", "SVM-RBF", "KNN", "Stacking"]
    clf_label = {"XGBoost": "XGBoost", "RandomForest": "Random forest",
                 "SVM-RBF": "SVM", "KNN": "k-NN", "Stacking": "Stacking"}
    k_order = ["256", "512", "all"]
    r["ci"] = pd.Categorical(r["classifier"], clf_order)
    r["ki"] = pd.Categorical(r["k"], k_order)
    r = r.sort_values(["ki", "ci"]).reset_index(drop=True)
    labels = [f"{clf_label[c]}, k = {'all' if k == 'all' else k}"
              for c, k in zip(r["classifier"], r["k"])]
    ypos = np.arange(len(r))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(FULL, 4.3), sharey=True)
    for ax, tag, letter, mk in [(axes[0], "internal", "a", "o"),
                                (axes[1], "external", "b", "s")]:
        est = r[f"{tag}_delta"].values
        lo, hi = r[f"{tag}_ci_lo"].values, r[f"{tag}_ci_hi"].values
        ax.errorbar(est, ypos, xerr=[est - lo, hi - est], fmt=mk, color="black",
                    markerfacecolor="white", markersize=4, elinewidth=0.8, capsize=2,
                    markeredgewidth=0.8)
        ax.axvline(0, color="grey", lw=0.7, ls="--")
        for b in [4.5, 9.5]:
            ax.axhline(b, color="lightgrey", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="bottom", ha="right")
        ax.text(0.5, 1.02, "HAM10000 (internal)" if tag == "internal" else "ISIC 2020 (external)",
                transform=ax.transAxes, fontsize=8, ha="center", va="bottom")
    lim = max(r["internal_ci_hi"].max(), r["external_ci_hi"].max()) + 0.01
    low = min(r["internal_ci_lo"].min(), r["external_ci_lo"].min()) - 0.01
    for ax in axes:
        ax.set_xlim(low, lim)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(labels)
    fig.supxlabel("AUROC difference, image-wise minus lesion-grouped cross-validation",
                  fontsize=8.5, y=0.01)
    fig.tight_layout()
    save(fig, "Fig6")


# ------------------------------------------------------------------ Fig 2
def _load_hashes(tag):
    z = np.load(ROOT / "results_mirror_audit" / "cache" / f"{tag}_d1.npz")
    return z["H"].astype(np.float32)[:, 0]


def _nn_dist(A, B, chunk=512):
    b_sum = B.sum(axis=1)[None, :]
    out = np.empty(len(A), dtype=np.float32)
    for s in range(0, len(A), chunk):
        a = A[s:s + chunk]
        out[s:s + chunk] = (a.sum(axis=1)[:, None] + b_sum - 2.0 * (a @ B.T)).min(axis=1)
    return out


def fig2():
    null = _nn_dist(_load_hashes("ref_isic2020"), _load_hashes("ref_ham10000"))
    panels = [("Null: ISIC 2020 against HAM10000", null)]
    for mirror, name in MIRROR_NAMES.items():
        f = ROOT / "results_mirror_audit" / f"matches_{mirror}.csv"
        if not f.exists():
            print(f"  (Fig2: {f.name} missing; panel skipped)")
            continue
        m = pd.read_csv(f)
        d = np.minimum(m["HAM10000_nn_dist"].values, m["ISIC2020_nn_dist"].values)
        panels.append((name, d))
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(FULL, 1.75 * nrow + 0.3), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    bins = np.arange(0, 132, 4)
    for ax, (title, d), letter in zip(axes, panels, "abcdef"):
        ax.hist(d, bins=bins, weights=np.full(len(d), 100.0 / len(d)), color="#A6A6A6",
                edgecolor="black", linewidth=0.4)
        ax.axvline(10, color="black", lw=0.8, ls="--")
        share = 100 * (d <= 10).mean()
        ax.text(0.97, 0.93, f"\u2264 10 bits: {share:.1f}%", transform=ax.transAxes,
                ha="right", va="top", fontsize=7)
        ax.text(0.5, 1.03, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5)
        ax.text(-0.02, 1.03, letter, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, fontweight="bold")
        ax.set_xlim(0, 128)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.supxlabel("Hamming distance to nearest reference image (bits)", fontsize=8.5, y=0.01)
    fig.supylabel("Images (%)", fontsize=8.5, x=0.01)
    fig.tight_layout()
    print(f"  Fig2: null median {np.median(null):.0f} bits, "
          f"{100 * (null <= 10).mean():.3f}% within 10 bits")
    save(fig, "Fig2")


# ------------------------------------------------------------------ Fig 7
def fig7():
    sc = pd.read_csv(ROOT / "results_shortcut" / "shortcut_results.csv")
    sc = sc[sc["mirror"] == "Javid_10605"]
    po = pd.read_csv(ROOT / "results_shortcut" / "patient_overlap_results.csv")
    po = po[(po["mirror"] == "Javid_10605") & (po["subset"] == "patient-clean only")]
    rows = []
    spec = [("A_full_test", "A  Melanoma model, whole test folder"),
            ("B_same_model_ISIC2020_source_only", "B  Same model, ISIC 2020 source only"),
            ("D_control_trained_and_tested_within_ISIC2020_source", "D  Control within ISIC 2020 source"),
            ("C_source_classifier_(ISIC2020_vs_not)", "C  Source model, ISIC 2020 or not")]
    for key, lab in spec:
        r = sc[sc["evaluation"] == key].iloc[0]
        rows.append((lab, r["auroc"], r["auroc_ci_lo"], r["auroc_ci_hi"], "o"))
    for key, lab in [("B_full_mirror_model", "B  patient-clean images"),
                     ("D_within_source_control", "D  patient-clean images")]:
        r = po[po["model"] == key].iloc[0]
        rows.append((lab, r["auroc"], r["ci_lo"], r["ci_hi"], "s"))
    order = [0, 3, 1, 2, 4, 5]          # A, C, B, D, B clean, D clean
    rows = [rows[i] for i in order]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(MID, 2.6))
    for yi, (lab, est, lo, hi, mk) in zip(y, rows):
        ax.errorbar(est, yi, xerr=[[est - lo], [hi - est]], fmt=mk, color="black",
                    markerfacecolor="white", markersize=4.5, elinewidth=0.8, capsize=2,
                    markeredgewidth=0.8)
        ax.text(1.02, yi, f"{est:.3f}", va="center", fontsize=7,
                transform=ax.get_yaxis_transform())
    ax.axhline(3.5, color="lightgrey", lw=0.6)
    ax.axhline(1.5, color="lightgrey", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0.7, 1.005)
    ax.set_xlabel("AUROC (95% CI)")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "Fig7")


# ------------------------------------------------------------------ Fig 3
def fig3():
    r = pd.read_csv(ROOT / "results_tool_validation" / "sensitivity_by_transform.csv")
    r = r.iloc[::-1].reset_index(drop=True)
    groups = r["group"].values
    y = np.arange(len(r))
    shades = {"Resizing and recompression": "#4D4D4D", "Orientation": "#7F7F7F",
              "Arbitrary rotation": "#FFFFFF", "Cropping": "#FFFFFF",
              "Colour and blur": "#BFBFBF", "Combined": "#E6E6E6"}
    hatches = {"Arbitrary rotation": "////", "Cropping": "...."}
    fig, ax = plt.subplots(figsize=(FULL, 0.17 * len(r) + 0.8))
    for i, row in r.iterrows():
        ax.barh(i, row["sensitivity_d<=10"], color=shades.get(row["group"], "#BFBFBF"),
                hatch=hatches.get(row["group"], ""), edgecolor="black", linewidth=0.5, height=0.7)
        ax.text(min(row["sensitivity_d<=10"], 100) + 1, i, f"{row['sensitivity_d<=10']:.1f}",
                va="center", fontsize=6.5)
    ax.set_yticks(y)
    ax.set_yticklabels(r["transformation"], fontsize=7)
    for i in range(1, len(groups)):
        if groups[i] != groups[i - 1]:
            ax.axhline(i - 0.5, color="lightgrey", lw=0.6)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Copies detected at Hamming distance \u2264 10 (%)")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "Fig3")


# ------------------------------------------------------------------ Fig 8
def fig8():
    per = pd.read_csv(ROOT / "results_confounding" / "confounding_per_repeat.csv")
    tests = [("T_confounded", "Confounded test"), ("T_ISIC", "Within ISIC 2020"), ("T_HAM", "Within HAM10000")]
    fig, ax = plt.subplots(figsize=(MID, 2.7))
    x = np.arange(len(tests))
    for off, model, mk, lab in [(-0.12, "CONFOUNDED", "o", "Trained on confounded data"),
                                (0.12, "CONTROL", "s", "Trained without the confound")]:
        sub = per[per["model"] == model]
        means, sds = [], []
        for j, (tk, _) in enumerate(tests):
            v = sub[sub["test"] == tk]["auroc"].values
            ax.scatter(np.full(len(v), j + off), v, s=9, color="grey", zorder=2, linewidths=0)
            means.append(v.mean())
            sds.append(v.std(ddof=1))
        ax.errorbar(x + off, means, yerr=sds, fmt=mk, color="black", markerfacecolor="white",
                    markersize=5, capsize=2.5, elinewidth=0.8, markeredgewidth=0.8, label=lab, zorder=3)
    src = per[per["model"] == "SOURCE"]["auroc"]
    ax.set_xticks(x)
    ax.set_xticklabels([t[1] for t in tests])
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.65, 1.01)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left", fontsize=7)
    print(f"  Fig8: source model AUROC {src.mean():.3f} (reported in text, not plotted)")
    save(fig, "Fig8")


def main():
    print(f"Writing figures to {OUT}")
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    fig8()


if __name__ == "__main__":
    main()
