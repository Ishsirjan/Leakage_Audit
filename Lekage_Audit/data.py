"""COBRE loading + connectome cache."""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from .config import COBRE_DIR, NPZ_PATH, PHENO_PATH, SUBJECTS_FILE, WORKDIR, ensure_workdir


def normalize_subject_id(s) -> str:
    s = str(s).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s.zfill(7) if s.isdigit() else s


def load_covariates(subjects, labels):
    if not PHENO_PATH.is_file():
        raise FileNotFoundError(f"Missing phenotypic data: {PHENO_PATH}")
    pheno = pd.read_csv(PHENO_PATH, sep="\t")
    pheno["StrID"] = pheno["ID"].astype(str).str.zfill(7)
    pheno = pheno.set_index("StrID")
    subjects = [normalize_subject_id(s) for s in subjects]
    missing = [s for s in subjects if s not in pheno.index]
    if missing:
        raise KeyError(f"{len(missing)} subjects not in phenotype file (e.g. {missing[:3]})")
    cov_df = pheno.loc[subjects, ["Current Age", "Gender", "FD"]].copy()
    cov_df["Gender"] = (cov_df["Gender"] == "Male").astype(int)
    cov_df["label"] = labels.astype(int)
    cov_df["subject_id"] = subjects
    return cov_df.reset_index(drop=True)


def build_fc_from_nifti():
    import nibabel as nib
    from nilearn import datasets, image, input_data

    with open(SUBJECTS_FILE) as f:
        subject_ids = [ln.strip() for ln in f if ln.strip()]
    pheno = pd.read_csv(PHENO_PATH, sep="\t")
    pheno["StrID"] = pheno["ID"].astype(str).str.zfill(7)
    pheno = pheno[pheno["StrID"].isin(subject_ids)]
    sz = set(pheno[pheno["Subject Type"] == "Patient"]["StrID"])

    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
    atlas_labels = [x.decode("utf-8") if isinstance(x, bytes) else x for x in atlas["labels"]]
    atlas_img = atlas["maps"]

    mats, subs = [], []
    for i, sub in enumerate(subject_ids, 1):
        nii = glob.glob(str(COBRE_DIR / f"fmri_{sub}.nii*"))
        tsv = glob.glob(str(COBRE_DIR / f"fmri_{sub}.tsv*"))
        if not nii or not tsv:
            continue
        conf = pd.read_csv(tsv[0], sep="\t").select_dtypes(include=[np.number]).fillna(0)
        img = nib.load(nii[0])
        atlas_r = image.resample_to_img(atlas_img, img, interpolation="nearest")
        masker = input_data.NiftiLabelsMasker(
            labels_img=atlas_r, standardize=True, memory=str(WORKDIR / "nilearn_cache")
        )
        ts = masker.fit_transform(img, confounds=conf)
        mats.append(np.corrcoef(ts.T))
        subs.append(sub)
        if i % 20 == 0:
            print(f"  built {i}/{len(subject_ids)}")

    fc = np.stack(mats)
    labels = np.array([1 if s in sz else 0 for s in subs], dtype=int)
    ensure_workdir()
    np.savez_compressed(
        NPZ_PATH, fc=fc, subjects=np.array(subs), labels=labels, atlas_labels=np.array(atlas_labels)
    )
    print(f"✓ Saved cache {NPZ_PATH}")
    return fc, labels, subs, atlas_labels


def load_data():
    """Return fc, labels, cov_df, atlas_labels."""
    ensure_workdir()
    if NPZ_PATH.is_file():
        d = np.load(NPZ_PATH, allow_pickle=True)
        fc = d["fc"]
        subjects = [normalize_subject_id(s) for s in d["subjects"]]
        labels = d["labels"].astype(int)
        atlas_labels = list(d["atlas_labels"]) if "atlas_labels" in d.files else None
        print(f"✓ Loaded cache: {NPZ_PATH}  shape={fc.shape}")
    else:
        print("No NPZ cache — building from NIfTI (one-time, slow)...")
        fc, labels, subjects, atlas_labels = build_fc_from_nifti()

    cov_df = load_covariates(subjects, labels)
    print(f"✓ Subjects: {len(labels)} | SZ={(labels == 1).sum()} HC={(labels == 0).sum()}")
    if atlas_labels is None:
        from nilearn import datasets

        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
        atlas_labels = [x.decode("utf-8") if isinstance(x, bytes) else x for x in atlas["labels"]]
    return fc, labels, cov_df, atlas_labels


def build_X(fc, pairs, demo=None):
    """Extract edge features for a list of (i,j) pairs; optional demo concat."""
    cols = [fc[:, i, j] for i, j in pairs]
    X = np.column_stack(cols) if cols else np.zeros((fc.shape[0], 0))
    if demo is not None:
        X = np.hstack([X, demo])
    return X
