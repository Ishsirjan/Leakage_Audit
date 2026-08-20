"""Build Schaefer-100 connectomes from UCLA CNP rest fMRIPrep images.

Mirrors the COBRE pipeline enough for a within-cohort SAFE vs LEAKED contrast:
parcel time series, Pearson r, Fisher z, optional age/sex covariates.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.maskers import NiftiLabelsMasker
import nibabel as nib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leakage_audit.protocol import (  # noqa: E402
    LAPConfig,
    make_full_connectome_builder,
    nested_evaluate,
    run_safe_vs_leaked_topk,
    sanitize,
)

DATA = Path("D:/Leakage_Audit_data")
FUNC_ROOT = DATA / "ds000030/derivatives/fmriprep"
PHENO = DATA / "participants.tsv"
OUT_NPZ = DATA / "ucla_cnp_schaefer100_fc.npz"
RESULTS = ROOT / "results"


def load_confounds(path: Path, n_time: int) -> np.ndarray | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t")
    num = df.select_dtypes(include=[np.number]).copy()
    # Drop columns that are almost empty; fill the rest.
    keep = [c for c in num.columns if num[c].notna().mean() >= 0.8]
    if not keep:
        return None
    arr = num[keep].to_numpy(dtype=float)
    if arr.shape[0] != n_time:
        return None
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def main():
    ph = pd.read_csv(PHENO, sep="\t")
    ph = ph[(ph["diagnosis"].isin(["CONTROL", "SCHZ"])) & (ph["rest"] == 1.0)].copy()
    atlas = fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2,
                                      data_dir=str(DATA / "nilearn_data"))
    masker = NiftiLabelsMasker(
        labels_img=atlas.maps, standardize=True, memory=None, verbose=0
    )

    edges, y, sids, ages, sexes, fds = [], [], [], [], [], []
    t0 = time.time()
    for i, row in enumerate(ph.itertuples(index=False), 1):
        sid = str(row.participant_id)
        bold = FUNC_ROOT / sid / "func" / f"{sid}_task-rest_bold_space-MNI152NLin2009cAsym_preproc.nii.gz"
        conf_p = FUNC_ROOT / sid / "func" / f"{sid}_task-rest_bold_confounds.tsv"
        if not bold.exists():
            continue
        import nibabel as nib
        import nibabel as nib
        n_time = nib.load(str(bold)).shape[-1]
        cmat = load_confounds(conf_p, n_time)
        ts = masker.fit_transform(str(bold), confounds=cmat)
        if ts.ndim != 2 or ts.shape[0] < 80 or ts.shape[1] != 100:
            continue
        if (ts.std(axis=0) == 0).sum() > 5:
            continue
        fc = np.corrcoef(ts.T)
        fc = np.clip(np.nan_to_num(fc, nan=0.0), -0.999, 0.999)
        iu = np.triu_indices(100, k=1)
        edges.append(np.arctanh(fc[iu]))
        y.append(1 if row.diagnosis == "SCHZ" else 0)
        sids.append(sid)
        ages.append(float(row.age) if pd.notna(row.age) else np.nan)
        sexes.append(1.0 if str(row.gender).upper().startswith("M") else 0.0)
        if cmat is not None and "FramewiseDisplacement" in pd.read_csv(conf_p, sep="\t").columns:
            fd = pd.read_csv(conf_p, sep="\t")["FramewiseDisplacement"].to_numpy(dtype=float)
            fds.append(float(np.nanmean(fd)))
        else:
            fds.append(np.nan)
        if i % 10 == 0:
            print(f"{i}/{len(ph)} kept {len(edges)}  ({time.time()-t0:.0f}s)", flush=True)

    X = sanitize(np.array(edges)).astype(np.float32)
    y = np.asarray(y, dtype=int)
    demo = np.column_stack([
        np.nan_to_num(np.array(ages, dtype=float), nan=np.nanmedian(np.array(ages, dtype=float))),
        np.array(sexes, dtype=float),
        np.nan_to_num(np.array(fds, dtype=float), nan=np.nanmedian(np.array(fds, dtype=float))),
    ]).astype(np.float32)
    np.savez_compressed(OUT_NPZ, X=X, y=y, demo=demo, sids=np.array(sids))
    print(f"wrote {OUT_NPZ}  n={len(y)} cases={int(y.sum())} controls={int((1-y).sum())} p={X.shape[1]}")

    cfg = LAPConfig(n_splits=5, n_repeats=3, random_state=42, n_jobs=2)
    contrast = run_safe_vs_leaked_topk(X, y, np.array(sids), demo=demo, k=20, cfg=cfg)
    contrast["cohort"] = "UCLA_CNP"
    RESULTS.mkdir(exist_ok=True)
    contrast.to_csv(RESULTS / "external_cnp_contrast.csv", index=False)
    print(contrast[["name", "auc_mean", "auc_std", "delta_vs_safe"]].to_string(index=False))

    full = nested_evaluate(
        X, y, np.array(sids), make_full_connectome_builder(demo=demo),
        cfg, name="cnp_full_connectome_logistic",
    )
    pd.DataFrame([{k: v for k, v in full.items() if k not in ("y_oof", "p_oof")}]).to_csv(
        RESULTS / "external_cnp_full.csv", index=False
    )
    print(f"full connectome AUC {full['auc_mean']:.3f}")


if __name__ == "__main__":
    main()
