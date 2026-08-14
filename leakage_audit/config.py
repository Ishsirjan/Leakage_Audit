"""Path configuration via environment variables."""
from __future__ import annotations

import os
from pathlib import Path

_HOME = Path.home()
_DEFAULT_DL = _HOME / "Downloads"


def _env(*names: str, default: str) -> str:
    for name in names:
        val = os.environ.get(name)
        if val:
            return val
    return default


# COBRE Figshare extract (NIfTI + phenotypic_data.tsv)
COBRE_DIR = Path(
    _env("LEAKAGE_AUDIT_COBRE", "SCHIZOAPATHY_COBRE", default=str(_DEFAULT_DL / "4197885"))
)

# Outputs: NPZ cache, CSVs, figures
WORKDIR = Path(
    _env("LEAKAGE_AUDIT_WORKDIR", "SCHIZOAPATHY_WORKDIR", default=str(_DEFAULT_DL))
)

# Optional UCLA CNP BIDS root
CNP_DIR = Path(
    _env("LEAKAGE_AUDIT_CNP", "SCHIZOAPATHY_CNP", default=str(WORKDIR / "ds000030"))
)

NPZ_PATH = WORKDIR / "subject_fc_matrices.npz"
PHENO_PATH = COBRE_DIR / "phenotypic_data.tsv"
SUBJECTS_FILE = COBRE_DIR / "downloaded_subjects.txt"

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_workdir() -> Path:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    return WORKDIR
