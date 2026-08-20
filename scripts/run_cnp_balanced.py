"""1:1 age/sex/FD-matched UCLA CNP subsample (50 SZ, 50 HC).

Greedy nearest-neighbour matching on standardised covariates, then the same
SAFE vs LEAKED twenty-edge contrast as COBRE, plus a label-permutation null
on the observed (matched) connectomes.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

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
NPZ = DATA / "ucla_cnp_schaefer100_fc.npz"
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))
from run_null_cnp import edge_dependency, null_connectomes  # noqa: E402

N_SEEDS = 10
CFG = LAPConfig(n_splits=5, n_repeats=3, random_state=42, n_jobs=2)


def match_1to1(y, demo, rng):
    """Greedy 1:1 matching of controls to cases on standardised age, sex, FD."""
    z = demo.copy().astype(float)
    z = (z - z.mean(0)) / (z.std(0) + 1e-8)
    cases = np.where(y == 1)[0]
    ctrls = np.where(y == 0)[0]
    rng.shuffle(cases)
    used = set()
    keep = list(cases)
    for i in cases:
        d = np.linalg.norm(z[ctrls] - z[i], axis=1)
        order = np.argsort(d)
        for j in order:
            c = int(ctrls[j])
            if c not in used:
                used.add(c)
                keep.append(c)
                break
    keep = np.array(keep)
    return np.sort(keep)


def main():
    z = np.load(NPZ)
    X = sanitize(z["X"]).astype(np.float32)
    y = np.asarray(z["y"], dtype=int)
    demo = np.asarray(z["demo"], dtype=np.float32)
    sids = np.asarray(z["sids"])
    print(f"full n={len(y)} cases={int(y.sum())} controls={int((1 - y).sum())}", flush=True)

    rng = np.random.default_rng(42)
    idx = match_1to1(y, demo, rng)
    Xb, yb, demob, sidsb = X[idx], y[idx], demo[idx], sids[idx]
    print(f"matched n={len(yb)} cases={int(yb.sum())} controls={int((1 - yb).sum())}", flush=True)
    print(
        "FD mean SZ/HC:",
        float(demob[yb == 1, 2].mean()),
        float(demob[yb == 0, 2].mean()),
        "age mean SZ/HC:",
        float(demob[yb == 1, 0].mean()),
        float(demob[yb == 0, 0].mean()),
        flush=True,
    )
    np.savez_compressed(DATA / "ucla_cnp_matched50.npz", X=Xb, y=yb, demo=demob, sids=sidsb)

    t0 = time.time()
    contrast = run_safe_vs_leaked_topk(Xb, yb, sidsb, demo=demob, k=20, cfg=CFG)
    contrast["cohort"] = "UCLA_CNP_matched50"
    contrast.to_csv(RESULTS / "external_cnp_matched50_contrast.csv", index=False)
    print(contrast[["name", "auc_mean", "auc_std", "accuracy_mean", "delta_vs_safe"]].to_string(index=False), flush=True)

    full = nested_evaluate(
        Xb, yb, sidsb, make_full_connectome_builder(demo=demob),
        CFG, name="cnp_matched50_full_connectome",
    )
    pd.DataFrame([{k: v for k, v in full.items() if k not in ("y_oof", "p_oof")}]).to_csv(
        RESULTS / "external_cnp_matched50_full.csv", index=False
    )
    print(f"full connectome AUC {full['auc_mean']:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    n, p = Xb.shape
    print("independent Gaussian null, n=100 50/50", flush=True)
    safes, leakeds = [], []
    for s in range(N_SEEDS):
        rng = np.random.default_rng(8200 + s)
        Xi = rng.standard_normal((n, p)).astype(np.float32)
        yi = np.zeros(n, dtype=int)
        yi[rng.choice(n, n // 2, False)] = 1
        d = run_safe_vs_leaked_topk(
            Xi, yi, np.arange(n), k=20,
            cfg=LAPConfig(n_splits=5, n_repeats=3, random_state=8200 + s, n_jobs=2),
        )
        safes.append(float(d.loc[d.name.str.contains("SAFE"), "auc_mean"].iloc[0]))
        leakeds.append(float(d.loc[d.name.str.contains("LEAKED"), "auc_mean"].iloc[0]))
        print(f"  ind seed {s} safe {safes[-1]:.3f} leaked {leakeds[-1]:.3f}", flush=True)
    ind = dict(
        cell="CNP-matched50", n=n, p=p, k=20, n_pos=50,
        auc_safe_mean=float(np.mean(safes)), auc_safe_std=float(np.std(safes)),
        auc_leaked_mean=float(np.mean(leakeds)), auc_leaked_std=float(np.std(leakeds)),
        optimism_mean=float(np.mean(np.array(leakeds) - np.array(safes))), n_seeds=N_SEEDS,
    )
    print(ind, flush=True)
    matched = pd.read_csv(RESULTS / "null_control_matched.csv")
    matched = matched[~matched.cell.str.startswith("CNP")]
    matched = pd.concat([matched, pd.DataFrame([ind])], ignore_index=True)
    matched.to_csv(RESULTS / "null_control_matched.csv", index=False)

    print("correlated synthetic null, n=100", flush=True)
    safes, leakeds, deps = [], [], []
    for s in range(N_SEEDS):
        rng = np.random.default_rng(9200 + s)
        Xc = null_connectomes(rng, n, 100, 152)
        yi = np.zeros(n, dtype=int)
        yi[rng.choice(n, n // 2, False)] = 1
        d = run_safe_vs_leaked_topk(
            Xc, yi, np.arange(n), k=20,
            cfg=LAPConfig(n_splits=5, n_repeats=3, random_state=9200 + s, n_jobs=2),
        )
        safes.append(float(d.loc[d.name.str.contains("SAFE"), "auc_mean"].iloc[0]))
        leakeds.append(float(d.loc[d.name.str.contains("LEAKED"), "auc_mean"].iloc[0]))
        deps.append(edge_dependency(Xc, rng))
        print(f"  corr seed {s} safe {safes[-1]:.3f} leaked {leakeds[-1]:.3f}", flush=True)
    corr_row = dict(
        cell="CNP-matched50, correlated edges",
        n=n, p=p, k=20, n_rois=100, n_timepoints=152, n_pos=50,
        mean_abs_edge_corr=float(np.mean(deps)),
        auc_safe_mean=float(np.mean(safes)), auc_safe_std=float(np.std(safes)),
        auc_leaked_mean=float(np.mean(leakeds)), auc_leaked_std=float(np.std(leakeds)),
        optimism_mean=float(np.mean(np.array(leakeds) - np.array(safes))), n_seeds=N_SEEDS,
    )
    print(corr_row, flush=True)
    conn = pd.read_csv(RESULTS / "null_control_connectome.csv")
    conn = conn[~conn.cell.str.startswith("CNP")]
    conn = pd.concat([conn, pd.DataFrame([corr_row])], ignore_index=True)
    conn.to_csv(RESULTS / "null_control_connectome.csv", index=False)

    print("label permutation of observed matched connectomes", flush=True)
    safes, leakeds = [], []
    for s in range(N_SEEDS):
        rng = np.random.default_rng(10200 + s)
        yp = yb.copy()
        rng.shuffle(yp)
        d = run_safe_vs_leaked_topk(
            Xb, yp, sidsb, demo=demob, k=20,
            cfg=LAPConfig(n_splits=5, n_repeats=3, random_state=10200 + s, n_jobs=2),
        )
        safes.append(float(d.loc[d.name.str.contains("SAFE"), "auc_mean"].iloc[0]))
        leakeds.append(float(d.loc[d.name.str.contains("LEAKED"), "auc_mean"].iloc[0]))
        print(f"  perm seed {s} safe {safes[-1]:.3f} leaked {leakeds[-1]:.3f}", flush=True)
    perm = dict(
        cell="CNP-matched50, label permutation",
        n=n, p=p, k=20, n_pos=50,
        auc_safe_mean=float(np.mean(safes)), auc_safe_std=float(np.std(safes)),
        auc_leaked_mean=float(np.mean(leakeds)), auc_leaked_std=float(np.std(leakeds)),
        optimism_mean=float(np.mean(np.array(leakeds) - np.array(safes))), n_seeds=N_SEEDS,
    )
    print(perm, flush=True)
    pd.DataFrame([perm]).to_csv(RESULTS / "null_control_cnp_permutation.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
