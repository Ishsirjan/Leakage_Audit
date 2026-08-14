"""
Leakage Audit Protocol (LAP)
============================
Feature builders fit on training indices only; SAFE vs LEAKED audit helpers.
MLMI 2026 companion code — Ishsirjan Kaur Chandok.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class LAPConfig:
    n_splits: int = 5
    n_repeats: int = 10
    random_state: int = 42
    inner_splits: int = 5
    n_jobs: int = -1


def subject_level_auc(subject_ids, y, probs) -> float:
    df = pd.DataFrame({"sid": subject_ids, "y": y, "p": probs})
    g = df.groupby("sid", as_index=False).agg(y=("y", "first"), p=("p", "mean"))
    if g["y"].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(g["y"], g["p"]))


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece)


def sanitize(X):
    X = np.asarray(X, dtype=float)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def vec_upper(fc: np.ndarray):
    r = fc.shape[1]
    iu = np.triu_indices(r, k=1)
    X = np.stack([fc[:, i, j] for i, j in zip(iu[0], iu[1])], axis=1)
    return sanitize(X), iu


def topk_t_indices(X: np.ndarray, y: np.ndarray, k: int = 20) -> np.ndarray:
    t, _ = stats.ttest_ind(X[y == 1], X[y == 0], axis=0, equal_var=False)
    t = np.nan_to_num(t, nan=0.0)
    return np.argsort(np.abs(t))[-k:]


def residualize_edges(X, confounds, train_idx, test_idx):
    X = sanitize(X)
    C = np.nan_to_num(np.asarray(confounds, dtype=float), nan=0.0)
    C_tr = np.column_stack([np.ones(len(train_idx)), C[train_idx]])
    C_te = np.column_stack([np.ones(len(test_idx)), C[test_idx]])
    X_tr, X_te = X[train_idx].copy(), X[test_idx].copy()
    beta, _, _, _ = np.linalg.lstsq(C_tr, X_tr, rcond=None)
    return sanitize(X_tr - C_tr @ beta), sanitize(X_te - C_te @ beta)


FeatureBuilder = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple]


def make_topk_builder(k=20, demo=None, residualize_on=None) -> FeatureBuilder:
    def builder(X_all, y, tr, te):
        if residualize_on is not None:
            X_tr, X_te = residualize_edges(X_all, residualize_on, tr, te)
        else:
            X_tr, X_te = X_all[tr], X_all[te]
        idx = topk_t_indices(X_tr, y[tr], k=k)
        X_tr, X_te = X_tr[:, idx], X_te[:, idx]
        if demo is not None:
            X_tr = np.hstack([X_tr, demo[tr]])
            X_te = np.hstack([X_te, demo[te]])
        return sanitize(X_tr), sanitize(X_te)

    return builder


def make_leaked_topk_builder(k=20, demo=None) -> FeatureBuilder:
    def builder(X_all, y, tr, te):
        idx = topk_t_indices(X_all, y, k=k)
        X_tr, X_te = X_all[tr][:, idx], X_all[te][:, idx]
        if demo is not None:
            X_tr = np.hstack([X_tr, demo[tr]])
            X_te = np.hstack([X_te, demo[te]])
        return sanitize(X_tr), sanitize(X_te)

    return builder


def make_full_connectome_builder(demo=None, residualize_on=None) -> FeatureBuilder:
    def builder(X_all, y, tr, te):
        if residualize_on is not None:
            X_tr, X_te = residualize_edges(X_all, residualize_on, tr, te)
        else:
            X_tr, X_te = X_all[tr], X_all[te]
        if demo is not None:
            X_tr = np.hstack([X_tr, demo[tr]])
            X_te = np.hstack([X_te, demo[te]])
        return sanitize(X_tr), sanitize(X_te)

    return builder


def default_logistic_pipe():
    return Pipeline([
        ("s", StandardScaler()),
        ("c", LogisticRegression(max_iter=5000, random_state=42)),
    ]), {"c__C": [0.01, 0.1, 1.0, 10.0]}


def default_elasticnet_pipe():
    return Pipeline([
        ("s", StandardScaler()),
        ("c", LogisticRegression(
            penalty="elasticnet", solver="saga", max_iter=8000, random_state=42
        )),
    ]), {"c__C": [0.01, 0.1, 1.0], "c__l1_ratio": [0.1, 0.5, 0.9]}


def nested_evaluate(X_all, y, subject_ids, feature_builder, cfg, pipe=None, grid=None, name="model"):
    if pipe is None or grid is None:
        pipe, grid = default_logistic_pipe()
    outer = RepeatedStratifiedKFold(
        n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, random_state=cfg.random_state
    )
    inner = StratifiedKFold(n_splits=cfg.inner_splits, shuffle=True, random_state=cfg.random_state)
    fold_aucs, fold_accs, sid_all, y_all, p_all = [], [], [], [], []
    for tr, te in outer.split(X_all, y):
        X_tr, X_te = feature_builder(X_all, y, tr, te)
        y_tr, y_te = y[tr], y[te]
        gs = GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc", n_jobs=cfg.n_jobs)
        gs.fit(X_tr, y_tr)
        prob = gs.predict_proba(X_te)[:, 1]
        pred = (prob >= 0.5).astype(int)
        if len(np.unique(y_te)) > 1:
            fold_aucs.append(roc_auc_score(y_te, prob))
        fold_accs.append(accuracy_score(y_te, pred))
        sid_all.extend(np.asarray(subject_ids)[te].tolist())
        y_all.extend(y_te.tolist())
        p_all.extend(prob.tolist())
    y_all, p_all, sid_all = map(np.asarray, (y_all, p_all, sid_all))
    return {
        "name": name,
        "auc_mean": float(np.mean(fold_aucs)),
        "auc_std": float(np.std(fold_aucs)),
        "accuracy_mean": float(np.mean(fold_accs)),
        "accuracy_std": float(np.std(fold_accs)),
        "auc_pooled_oof_naive": float(roc_auc_score(y_all, p_all)),
        "auc_subject_level_oof": subject_level_auc(sid_all, y_all, p_all),
        "ece": expected_calibration_error(y_all, p_all),
        "brier": float(brier_score_loss(y_all, p_all)),
        "n_folds": len(fold_aucs),
        "y_oof": y_all,
        "p_oof": p_all,
    }


def run_safe_vs_leaked_topk(X_edges, y, subject_ids, demo=None, k=20, cfg=None):
    cfg = cfg or LAPConfig()
    safe = nested_evaluate(
        X_edges, y, subject_ids, make_topk_builder(k=k, demo=demo), cfg, name=f"top{k}_SAFE"
    )
    leaked = nested_evaluate(
        X_edges, y, subject_ids, make_leaked_topk_builder(k=k, demo=demo), cfg, name=f"top{k}_LEAKED"
    )
    rows = [{kk: vv for kk, vv in r.items() if kk not in ("y_oof", "p_oof")} for r in (safe, leaked)]
    df = pd.DataFrame(rows)
    df["delta_vs_safe"] = df["auc_mean"] - df.loc[df["name"].str.contains("SAFE"), "auc_mean"].values[0]
    return df


def train_cobre_test_external(X_train, y_train, X_test, y_test, k=20, use_elasticnet=True):
    if use_elasticnet:
        pipe, grid = default_elasticnet_pipe()
        Xtr, Xte = sanitize(X_train), sanitize(X_test)
        mode = "elasticnet_full"
    else:
        pipe, grid = default_logistic_pipe()
        idx = topk_t_indices(X_train, y_train, k=k)
        Xtr, Xte = sanitize(X_train[:, idx]), sanitize(X_test[:, idx])
        mode = f"top{k}_locked"
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc", n_jobs=-1)
    gs.fit(Xtr, y_train)
    prob = gs.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "auc": float(roc_auc_score(y_test, prob)) if len(np.unique(y_test)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(y_test, pred)),
        "mode": mode,
    }
