"""
Experiment B, extra seeds: image-wise vs lesion-grouped cross-validation on
HAM10000 repeated with two more random seeds (7 and 123), so the effect can
be reported as mean and SD over three seeds (42 from the original runs,
plus 7 and 123).

Each run uses exactly the v3 backbone training (EfficientNet-B0, 320 px,
same optimiser, schedule, augmentation, early stopping) and the headline
shallow configuration of the paper: XGBoost on all 1,904 multi-stage
features (no SHAP ranking is needed when all features are used). The seed
changes the fold assignment, the inner validation split, the network
initialisation of the head, the augmentation order and the XGBoost
subsampling.

Four runs, each resumable fold by fold:
  lesion, seed 7   image, seed 7   lesion, seed 123   image, seed 123
Roughly 6 to 7 hours per run on the M-series GPU, about 26 to 28 hours total.

After the runs, the script aggregates all three seeds. Seed 42 is read from
the existing results_v3/probs (lesion-grouped) and results_split_image/probs
(image-wise), XGBoost on all features, so nothing is rerun for it.

Run everything (resumable, can be stopped and restarted any time):
  export MELANOMA_ROOT=/path/to/data
  caffeinate -dimsu python3 run_extra_seeds.py 2>&1 | tee extra_seeds_log.txt
Only aggregate (after all runs exist):
  python3 run_extra_seeds.py --aggregate-only
Outputs in $MELANOMA_ROOT/results_seeds/
"""

import argparse
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold,
                                     StratifiedKFold, StratifiedShuffleSplit)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# MELANOMA_ROOT if set, otherwise the data/ folder of this repository
ROOT = Path(os.environ.get("MELANOMA_ROOT", Path(__file__).resolve().parents[1] / "data")).expanduser()
OUT = ROOT / "results_seeds"
SEEDS = [7, 123]
RUNS = [("lesion", 7), ("image", 7), ("lesion", 123), ("image", 123)]

# v3 settings (unchanged)
IMG_SIZE, BATCH_SIZE, N_FOLDS, EPOCHS, PATIENCE = 320, 16, 5, 30, 6
WEIGHT_DECAY, LABEL_SMOOTH, VAL_FRACTION = 1e-4, 0.05, 0.15
LR_BACKBONE_EARLY, LR_BACKBONE_LATE, LR_HEAD = 1e-5, 5e-5, 5e-4
STAGE_INDICES = [5, 6, 7, 8]
N_BOOT = 1000


# ------------------------------------------------------------------ torch parts
def torch_setup():
    import torch
    import torch.nn as nn
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    class DS(Dataset):
        def __init__(self, paths, labels, tf):
            self.p, self.y, self.tf = list(paths), list(labels), tf

        def __len__(self):
            return len(self.p)

        def __getitem__(self, i):
            return self.tf(Image.open(self.p[i]).convert("RGB")), self.y[i]

    return torch, nn, DataLoader, DS, models, train_tf, eval_tf, device


def train_fold(T, tr_p, tr_y, va_p, va_y, fold, seed):
    torch, nn, DataLoader, DS, models, train_tf, eval_tf, device = T
    torch.manual_seed(seed * 100 + fold)
    tr = DataLoader(DS(tr_p, tr_y, train_tf), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    va = DataLoader(DS(va_p, va_y, eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier[1].in_features, 2))
    m = m.to(device)
    blocks = list(m.features)
    half = len(blocks) // 2
    opt = torch.optim.AdamW([
        {"params": [p for b in blocks[:half] for p in b.parameters()], "lr": LR_BACKBONE_EARLY},
        {"params": [p for b in blocks[half:] for p in b.parameters()], "lr": LR_BACKBONE_LATE},
        {"params": list(m.classifier.parameters()), "lr": LR_HEAD}], weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n_pos = int(np.sum(tr_y))
    w = torch.tensor([1.0, (len(tr_y) - n_pos) / max(n_pos, 1)], dtype=torch.float32).to(device)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=LABEL_SMOOTH)
    best, best_state, wait = -1.0, None, 0
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        m.train()
        for x, y in tr:
            opt.zero_grad()
            crit(m(x.to(device)), y.to(device)).backward()
            opt.step()
        sched.step()
        m.eval()
        ps, ys = [], []
        with torch.no_grad():
            for x, y in va:
                ps.append(torch.softmax(m(x.to(device)), 1)[:, 1].cpu().numpy())
                ys.append(y.numpy())
        auc = roc_auc_score(np.concatenate(ys), np.concatenate(ps))
        print(f"    [fold {fold}] epoch {ep}/{EPOCHS} val_auc={auc:.4f} ({time.time() - t0:.0f}s)")
        if auc > best:
            best, wait = auc, 0
            best_state = {k: v.cpu().clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"    [fold {fold}] early stop at {ep} (best {best:.4f})")
                break
    m.load_state_dict(best_state)
    return m


def features(T, model, paths, tag):
    torch, nn, DataLoader, DS, models, train_tf, eval_tf, device = T
    dl = DataLoader(DS(paths, np.zeros(len(paths), int), eval_tf), batch_size=32,
                    shuffle=False, num_workers=0)
    model.eval()
    out = []
    with torch.no_grad():
        for bi, (x, _) in enumerate(dl):
            x = x.to(device)
            views = []
            for v in (x, torch.flip(x, [3])):
                h, parts = v, []
                for idx, block in enumerate(model.features):
                    h = block(h)
                    if idx in STAGE_INDICES:
                        parts.append(nn.functional.adaptive_avg_pool2d(h, 1).flatten(1))
                views.append(torch.cat(parts, 1))
            out.append(torch.stack(views).mean(0).cpu().numpy())
            if bi % 200 == 0:
                print(f"      [{tag}] {bi * 32}/{len(paths)}")
    return np.concatenate(out)


# ------------------------------------------------------------------ splits and classifier
def partitions(split, y, groups, seed):
    idx = np.arange(len(y))
    if split == "lesion":
        outer = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(idx, y, groups)
    else:
        outer = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(idx, y)
    parts = []
    for tr_idx, te_idx in outer:
        if split == "lesion":
            i_tr, i_va = next(GroupShuffleSplit(1, test_size=VAL_FRACTION, random_state=seed)
                              .split(tr_idx, y[tr_idx], groups[tr_idx]))
        else:
            i_tr, i_va = next(StratifiedShuffleSplit(1, test_size=VAL_FRACTION, random_state=seed)
                              .split(tr_idx, y[tr_idx]))
        parts.append((tr_idx[i_tr], tr_idx[i_va], te_idx))
    return parts


def xgb(y, seed):
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
                         random_state=seed, n_jobs=-1)


def youden(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


# ------------------------------------------------------------------ one run
def run(split, seed, ham, isic):
    out = OUT / f"{split}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    paths, y, groups = ham["image_path"].values, ham["label"].values.astype(int), ham["lesion_group"].values
    parts = partitions(split, y, groups, seed)
    if all((out / f"fold{f}.npz").exists() for f in range(1, N_FOLDS + 1)):
        print(f"\n=== {split}, seed {seed}: already complete ===")
        return
    print(f"\n=== {split}, seed {seed} ===")
    T = torch_setup()
    random.seed(seed)
    np.random.seed(seed)
    for f, (tr, va, te) in enumerate(parts, 1):
        if (out / f"fold{f}.npz").exists():
            print(f"  fold {f}: done, skipped")
            continue
        t0 = time.time()
        leaked = np.isin(groups[te], groups[tr])
        print(f"  fold {f}: train {len(tr)}, val {len(va)}, test {len(te)}, "
              f"test images whose lesion is in train {leaked.sum()} ({100 * leaked.mean():.1f}%)")
        model = train_fold(T, paths[tr], y[tr], paths[va], y[va], f, seed)
        Xtr = features(T, model, paths[tr], "train")
        Xva = features(T, model, paths[va], "val")
        Xte = features(T, model, paths[te], "test")
        Xex = features(T, model, isic["image_path"].values, "isic")
        sc = StandardScaler().fit(Xtr)
        clf = xgb(y[tr], seed).fit(sc.transform(Xtr), y[tr])
        thr = youden(y[va], clf.predict_proba(sc.transform(Xva))[:, 1])
        p_te = clf.predict_proba(sc.transform(Xte))[:, 1]
        p_ex = clf.predict_proba(sc.transform(Xex))[:, 1]
        np.savez_compressed(out / f"fold{f}.npz", test_index=te, te_p=p_te, ex_p=p_ex,
                            thr=np.array(thr), leaked=leaked)
        print(f"  fold {f}: HAM AUROC {roc_auc_score(y[te], p_te):.4f} | ISIC AUROC "
              f"{roc_auc_score(isic['label'].values, p_ex):.4f} | {(time.time() - t0) / 60:.0f} min")
        del model
        if T[-1].type == "mps":
            T[0].mps.empty_cache()


# ------------------------------------------------------------------ aggregation
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
    raise SystemExit("ISIC 2020 ground-truth CSV (image_name, patient_id) not found; set ISIC_GT")


def load_run(split, seed, n):
    """Returns out-of-fold probabilities, external ensemble probabilities and leak flags."""
    if seed == 42:
        d = ROOT / ("results_v3" if split == "lesion" else "results_split_image") / "probs"
        te_key, ex_key = "te__all__XGBoost", "ex__all__XGBoost"
    else:
        d = OUT / f"{split}_seed{seed}"
        te_key, ex_key = "te_p", "ex_p"
    files = [d / f"fold{f}.npz" for f in range(1, N_FOLDS + 1)]
    if not all(f.exists() for f in files):
        return None
    oof, leak, ex = np.full(n, np.nan), np.zeros(n, bool), []
    for f in files:
        z = np.load(f, allow_pickle=True)
        oof[z["test_index"]] = z[te_key]
        if "leaked" in z.files:
            leak[z["test_index"]] = z["leaked"]
        ex.append(z[ex_key])
    return oof, np.mean(ex, axis=0), leak


def clusters(g):
    codes, uniq = pd.factorize(g)
    order = np.argsort(codes, kind="stable")
    b = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    return [order[b[i]:b[i + 1]] for i in range(len(uniq))]


def paired_boot(y, pa, pb, cl, rng):
    d = []
    while len(d) < N_BOOT:
        idx = np.concatenate([cl[i] for i in rng.integers(0, len(cl), len(cl))])
        if y[idx].min() == y[idx].max():
            continue
        d.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
    return np.percentile(d, [2.5, 97.5])


def sens90(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(tpr[(1 - fpr) >= 0.90].max())


def aggregate(ham, isic):
    y = ham["label"].values.astype(int)
    groups = ham["lesion_group"].values
    ye = isic["label"].values.astype(int)
    gt = pd.read_csv(find_ground_truth())
    pid = dict(zip(gt["image_name"].astype(str), gt["patient_id"].astype(str)))
    patients = np.array([pid.get(Path(p).stem, Path(p).stem) for p in isic["image_path"]])
    cl_h, cl_e = clusters(groups), clusters(patients)
    rng = np.random.default_rng(0)
    rows = []
    for seed in [42] + SEEDS:
        les, img = load_run("lesion", seed, len(y)), load_run("image", seed, len(y))
        if les is None or img is None:
            print(f"  seed {seed}: runs incomplete, skipped")
            continue
        (pl, el, _), (pi, ei, leak) = les, img
        ci_int = paired_boot(y, pi, pl, cl_h, rng)
        ci_ext = paired_boot(ye, ei, el, cl_e, rng)
        r = {"seed": seed,
             "internal_lesion": roc_auc_score(y, pl), "internal_image": roc_auc_score(y, pi),
             "external_lesion": roc_auc_score(ye, el), "external_image": roc_auc_score(ye, ei),
             "sens90_internal_lesion": sens90(y, pl), "sens90_internal_image": sens90(y, pi),
             "pct_test_leaked": 100 * leak.mean(),
             "pct_melanoma_test_leaked": 100 * leak[y == 1].mean()}
        r["internal_delta"] = r["internal_image"] - r["internal_lesion"]
        r["internal_ci_lo"], r["internal_ci_hi"] = ci_int
        r["external_delta"] = r["external_image"] - r["external_lesion"]
        r["external_ci_lo"], r["external_ci_hi"] = ci_ext
        r["sens90_delta"] = r["sens90_internal_image"] - r["sens90_internal_lesion"]
        r["gap_widening"] = r["internal_delta"] - r["external_delta"]
        rows.append(r)
        print(f"  seed {seed}: internal {r['internal_lesion']:.4f} -> {r['internal_image']:.4f} "
              f"(d {r['internal_delta']:+.4f} [{ci_int[0]:+.4f}, {ci_int[1]:+.4f}]) | external "
              f"{r['external_lesion']:.4f} -> {r['external_image']:.4f} "
              f"(d {r['external_delta']:+.4f} [{ci_ext[0]:+.4f}, {ci_ext[1]:+.4f}]) | "
              f"melanoma leaked {r['pct_melanoma_test_leaked']:.1f}%")
    if not rows:
        return
    df = pd.DataFrame(rows)
    num = df.drop(columns="seed")
    summary = pd.concat([df, pd.DataFrame([{"seed": "mean", **num.mean().to_dict()},
                                           {"seed": "sd", **num.std(ddof=1).to_dict()}])])
    summary.to_csv(OUT / "seed_summary.csv", index=False)
    if len(rows) > 1:
        m, s = num.mean(), num.std(ddof=1)
        print(f"\n  Over {len(rows)} seeds: internal delta {m['internal_delta']:.4f} +/- "
              f"{s['internal_delta']:.4f}; external delta {m['external_delta']:.4f} +/- "
              f"{s['external_delta']:.4f}; gap widening {m['gap_widening']:.4f} +/- "
              f"{s['gap_widening']:.4f}; sensitivity at 90% specificity delta "
              f"{m['sens90_delta']:.3f} +/- {s['sens90_delta']:.3f}")
    print(f"\nSaved {OUT / 'seed_summary.csv'}")


def main():
    print(f"[Data folder] {ROOT}")
    if not (ROOT / "manifest_ham10000_384.csv").exists():
        raise SystemExit(f"manifest_ham10000_384.csv not found in {ROOT}. Run code/prepare_data.py "
                         "first, or set MELANOMA_ROOT to your data folder.")
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ham = pd.read_csv(ROOT / "manifest_ham10000_384.csv")
    isic = pd.read_csv(ROOT / "manifest_isic2020_384.csv")
    if not args.aggregate_only:
        for split, seed in RUNS:
            run(split, seed, ham, isic)
    print("\n=== AGGREGATION (XGBoost, all features) ===")
    aggregate(ham, isic)


if __name__ == "__main__":
    main()
