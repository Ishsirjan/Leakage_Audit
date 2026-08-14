"""LAP residualisation + controlled leakage + locked COBRE split."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leakage_audit.config import RESULTS_DIR, WORKDIR, ensure_workdir
from leakage_audit.data import load_data
from leakage_audit.protocol import (
    LAPConfig,
    default_elasticnet_pipe,
    make_full_connectome_builder,
    make_topk_builder,
    nested_evaluate,
    run_safe_vs_leaked_topk,
    train_cobre_test_external,
    vec_upper,
)

FULL = False  # True -> 5x10
CFG = LAPConfig(n_splits=5, n_repeats=10 if FULL else 3, random_state=42)


def main():
    ensure_workdir()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"CV 5x{CFG.n_repeats} | WORKDIR={WORKDIR}")

    fc, labels, cov_df, _ = load_data()
    sids = cov_df["subject_id"].astype(str).values
    demo = cov_df[["Current Age", "Gender", "FD"]].values.astype(float)
    fd = cov_df[["FD"]].values.astype(float)
    X, _ = vec_upper(fc)

    print("\n[1] SAFE vs LEAKED top-20...")
    leak_df = run_safe_vs_leaked_topk(X, labels, sids, demo=demo, k=20, cfg=CFG)
    print(leak_df[["name", "auc_mean", "auc_std", "delta_vs_safe"]].to_string(index=False))
    leak_df.to_csv(RESULTS_DIR / "lap_controlled_leakage.csv", index=False)
    leak_df.to_csv(WORKDIR / "lap_controlled_leakage.csv", index=False)

    print("\n[2] Elastic-net + FD residualisation...")
    pipe, grid = default_elasticnet_pipe()
    rows = []
    for name, builder in [
        ("en_demo", make_full_connectome_builder(demo=demo)),
        ("en_edges", make_full_connectome_builder(demo=None)),
        ("en_edges_FDresid", make_full_connectome_builder(demo=None, residualize_on=fd)),
        ("top20_SAFE_raw", make_topk_builder(k=20, demo=None)),
        ("top20_SAFE_FDresid", make_topk_builder(k=20, demo=None, residualize_on=fd)),
    ]:
        use_en = name.startswith("en_")
        r = nested_evaluate(
            X, labels, sids, builder, CFG,
            pipe=pipe if use_en else None,
            grid=grid if use_en else None,
            name=name,
        )
        print(f"  {name}: AUC={r['auc_mean']:.3f}±{r['auc_std']:.3f}")
        rows.append({k: v for k, v in r.items() if k not in ("y_oof", "p_oof")})
    sens = pd.DataFrame(rows)
    sens.to_csv(RESULTS_DIR / "lap_residualisation_sensitivity.csv", index=False)

    print("\n[3] Locked 70/30 splits...")
    lock_rows = []
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.30, random_state=42)
    for i, (tr, te) in enumerate(sss.split(X, labels), 1):
        for use_en in (True, False):
            out = train_cobre_test_external(X[tr], labels[tr], X[te], labels[te], k=20, use_elasticnet=use_en)
            out["split"] = i
            lock_rows.append(out)
    lock_df = pd.DataFrame(lock_rows)
    lock_df.to_csv(RESULTS_DIR / "lap_locked_split_cobre.csv", index=False)
    print(lock_df.groupby("mode")[["auc", "accuracy"]].agg(["mean", "std"]))

    summary = pd.DataFrame([
        {"metric": "controlled_safe_auc", "value": float(leak_df.loc[leak_df.name.str.contains("SAFE"), "auc_mean"].iloc[0])},
        {"metric": "controlled_leaked_auc", "value": float(leak_df.loc[leak_df.name.str.contains("LEAKED"), "auc_mean"].iloc[0])},
        {"metric": "controlled_delta", "value": float(leak_df.loc[leak_df.name.str.contains("LEAKED"), "delta_vs_safe"].iloc[0])},
        {"metric": "en_demo_auc", "value": float(sens.loc[sens.name == "en_demo", "auc_mean"].iloc[0])},
        {"metric": "en_edges_fdresid_auc", "value": float(sens.loc[sens.name == "en_edges_FDresid", "auc_mean"].iloc[0])},
        {"metric": "locked_en_auc_mean", "value": float(lock_df[lock_df.mode == "elasticnet_full"]["auc"].mean())},
    ])
    summary.to_csv(RESULTS_DIR / "lap_summary_metrics.csv", index=False)
    print("✓ wrote results/lap_*.csv")


if __name__ == "__main__":
    main()
