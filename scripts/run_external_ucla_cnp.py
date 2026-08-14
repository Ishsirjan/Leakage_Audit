"""Optional external validation: COBRE → UCLA CNP (OpenNeuro ds000030)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leakage_audit.config import CNP_DIR, RESULTS_DIR, WORKDIR, ensure_workdir
from leakage_audit.data import load_data
from leakage_audit.protocol import (
    LAPConfig,
    run_safe_vs_leaked_topk,
    train_cobre_test_external,
    vec_upper,
)

CNP_NPZ = WORKDIR / "ucla_cnp_schaefer100_fc.npz"


def main():
    ensure_workdir()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cobre_fc, cobre_y, _, _ = load_data()
    cobre_X, _ = vec_upper(cobre_fc)

    if not CNP_NPZ.is_file():
        print("CNP NPZ not found:", CNP_NPZ)
        print("Download rest-only later when disk allows:")
        print("  openneuro-py download --dataset=ds000030 --target-dir=<path> --include=participants.tsv --include=**/func/*rest*")
        print("BIDS root currently:", CNP_DIR)
        pd.DataFrame([{"status": "CNP_NOT_AVAILABLE"}]).to_csv(RESULTS_DIR / "external_cnp_status.csv", index=False)
        return

    d = np.load(CNP_NPZ, allow_pickle=True)
    cnp_X, _ = vec_upper(d["fc"])
    cnp_y = d["labels"].astype(int)
    cnp_sids = np.array([str(s) for s in d["subjects"]])

    rows = []
    for use_en in (True, False):
        out = train_cobre_test_external(cobre_X, cobre_y, cnp_X, cnp_y, k=20, use_elasticnet=use_en)
        rows.append(out)
        print(out)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "external_cobre_to_cnp.csv", index=False)

    leak = run_safe_vs_leaked_topk(cnp_X, cnp_y, cnp_sids, demo=None, k=20, cfg=LAPConfig(n_repeats=3))
    leak.to_csv(RESULTS_DIR / "external_cnp_lap_audit.csv", index=False)
    print(leak[["name", "auc_mean", "delta_vs_safe"]])
    pd.DataFrame([{"status": "OK", "cnp_n": int(len(cnp_y))}]).to_csv(RESULTS_DIR / "external_cnp_status.csv", index=False)


if __name__ == "__main__":
    main()
