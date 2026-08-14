"""Multi-method leakage-free benchmark (paper Table 3)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_ind
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leakage_audit.config import RESULTS_DIR, WORKDIR, ensure_workdir
from leakage_audit.data import build_X, load_data
from leakage_audit.protocol import sanitize, vec_upper

FAST = False  # False = 5×10 paper setting
N_SPLITS, N_REPEATS, RANDOM_STATE = 5, (3 if FAST else 10), 42

# Five covariate-adjusted FDR edges (Table 2) — Schaefer-100 labels
FIXED_EDGE_NAMES = [
    ("7Networks_LH_Vis_3", "7Networks_RH_Default_Temp_2"),
    ("7Networks_LH_SalVentAttn_FrOperIns_1", "7Networks_RH_Default_PFCv_1"),
    ("7Networks_LH_Default_Par_2", "7Networks_RH_Cont_PFCl_1"),
    ("7Networks_RH_Vis_8", "7Networks_RH_Default_Temp_2"),
    ("7Networks_RH_SalVentAttn_TempOccPar_2", "7Networks_RH_Default_PFCv_1"),
]


def resolve_fixed_pairs(atlas_labels):
    lab = {str(a): i for i, a in enumerate(atlas_labels)}
    pairs = []
    for a, b in FIXED_EDGE_NAMES:
        if a not in lab or b not in lab:
            raise KeyError(f"Missing atlas labels for {a} / {b}")
        pairs.append((lab[a], lab[b]))
    return pairs


def parcel_network(label):
    parts = str(label).split("_")
    return parts[2] if len(parts) >= 4 and parts[1] in ("LH", "RH") else "Unknown"


def network_features(fc, atlas_labels):
    nets = np.array([parcel_network(lab) for lab in atlas_labels])
    uniq = sorted(n for n in set(nets) if n not in ("Background", "Unknown"))
    feats, names = [], []
    for a, na in enumerate(uniq):
        for b in range(a, len(uniq)):
            nb = uniq[b]
            ia, ib = np.where(nets == na)[0], np.where(nets == nb)[0]
            block = fc[:, ia][:, :, ib]
            if na == nb:
                if block.shape[1] < 2:
                    continue
                tri = np.triu_indices(block.shape[1], k=1)
                vals = block[:, tri[0], tri[1]]
            else:
                vals = block.reshape(fc.shape[0], -1)
            col = np.nan_to_num(np.nanmean(vals, axis=1), nan=0.0)
            feats.append(col)
            names.append(f"{na}-{nb}")
    return sanitize(np.column_stack(feats)), names


def cpm_two_feature(fc_tr, y_tr, fc_te, pct=10):
    X_tr, _ = vec_upper(fc_tr)
    X_te, _ = vec_upper(fc_te)
    corrs = np.zeros(X_tr.shape[1])
    for k in range(X_tr.shape[1]):
        if X_tr[:, k].std() > 1e-8:
            corrs[k], _ = pearsonr(X_tr[:, k], y_tr)
    n_top = max(1, int(X_tr.shape[1] * pct / 100))
    pos, neg = np.argsort(corrs)[-n_top:], np.argsort(corrs)[:n_top]
    return (
        sanitize(np.column_stack([X_tr[:, pos].sum(1), X_tr[:, neg].sum(1)])),
        sanitize(np.column_stack([X_te[:, pos].sum(1), X_te[:, neg].sum(1)])),
    )


def topk_t_edges(fc_tr, y_tr, fc_te, k=20):
    t_stats, _ = ttest_ind(fc_tr[y_tr == 1], fc_tr[y_tr == 0], axis=0)
    np.fill_diagonal(t_stats, 0)
    iu = np.triu_indices(fc_tr.shape[1], k=1)
    tvec = np.array([t_stats[i, j] for i, j in zip(*iu)])
    idx = np.argsort(np.abs(tvec))[-k:]
    pairs = [(iu[0][i], iu[1][i]) for i in idx]
    return build_X(fc_tr, pairs, None), build_X(fc_te, pairs, None)


def eval_method(name, fc, labels, cov_df, feature_fn, models):
    outer = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for mname, (pipe, grid) in models.items():
        fold_aucs, fold_accs, yt_all, yp_all = [], [], [], []
        for tr, te in outer.split(fc, labels):
            X_tr, X_te = feature_fn(fc[tr], labels[tr], fc[te])
            demo_tr = cov_df.iloc[tr][["Current Age", "Gender", "FD"]].values.astype(float)
            demo_te = cov_df.iloc[te][["Current Age", "Gender", "FD"]].values.astype(float)
            if name.endswith("_demo"):
                X_tr = np.hstack([X_tr, demo_tr])
                X_te = np.hstack([X_te, demo_te])
            X_tr, X_te = sanitize(X_tr), sanitize(X_te)
            gs = GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc", n_jobs=-1)
            gs.fit(X_tr, labels[tr])
            prob = gs.predict_proba(X_te)[:, 1]
            if len(np.unique(labels[te])) > 1:
                fold_aucs.append(roc_auc_score(labels[te], prob))
            fold_accs.append(accuracy_score(labels[te], (prob >= 0.5).astype(int)))
            yt_all.extend(labels[te].tolist())
            yp_all.extend(prob.tolist())
        rows.append({
            "method": name,
            "model": mname,
            "auc_mean": float(np.mean(fold_aucs)),
            "auc_std": float(np.std(fold_aucs)),
            "accuracy_mean": float(np.mean(fold_accs)),
            "accuracy_std": float(np.std(fold_accs)),
            "auc_pooled_oof": float(roc_auc_score(yt_all, yp_all)),
            "n_folds": len(fold_aucs),
            "leakage_safe": True,
        })
    return rows


def main():
    ensure_workdir()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fc, labels, cov_df, atlas_labels = load_data()
    fixed_pairs = resolve_fixed_pairs(atlas_labels)

    models_lr = {
        "LogisticRegression": (
            Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=3000))]),
            {"c__C": [0.01, 0.1, 1, 10]},
        )
    }
    models_en = {
        "ElasticNet": (
            Pipeline([
                ("s", StandardScaler()),
                ("c", LogisticRegression(penalty="elasticnet", solver="saga", max_iter=5000)),
            ]),
            {"c__C": [0.01, 0.1, 1], "c__l1_ratio": [0.1, 0.5, 0.9]},
        )
    }

    def full_edges(tr, y, te):
        return sanitize(vec_upper(tr)[0]), sanitize(vec_upper(te)[0])

    def pca30(tr, y, te):
        Xtr, Xte = sanitize(vec_upper(tr)[0]), sanitize(vec_upper(te)[0])
        pca = PCA(n_components=min(30, Xtr.shape[0] - 2), random_state=RANDOM_STATE)
        return pca.fit_transform(Xtr), pca.transform(Xte)

    def net28(tr, y, te):
        return network_features(tr, atlas_labels)[0], network_features(te, atlas_labels)[0]

    def fixed5(tr, y, te):
        return build_X(tr, fixed_pairs), build_X(te, fixed_pairs)

    methods = [
        ("fixed5_edges_demo", fixed5, models_lr),
        ("cpm10pct_demo", lambda tr, y, te: cpm_two_feature(tr, y, te, 10), models_lr),
        ("top20_train_t_demo", lambda tr, y, te: topk_t_edges(tr, y, te, 20), models_lr),
        ("network28_demo", net28, models_lr),
        ("elasticnet_4950edges", full_edges, models_en),
        ("elasticnet_4950edges_demo", full_edges, models_en),
        ("pca30_edges_demo", pca30, models_lr),
    ]

    all_rows, t0 = [], time.time()
    for i, (name, fn, models) in enumerate(methods, 1):
        print(f"[{i}/{len(methods)}] {name}")
        all_rows.extend(eval_method(name, fc, labels, cov_df, fn, models))
    df = pd.DataFrame(all_rows).sort_values("auc_mean", ascending=False)
    out = RESULTS_DIR / "methods_benchmark_comparison.csv"
    df.to_csv(out, index=False)
    df.to_csv(WORKDIR / "methods_benchmark_comparison.csv", index=False)
    print(df.head(10).to_string(index=False))
    print(f"✓ {out} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
