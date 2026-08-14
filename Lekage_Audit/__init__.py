"""Leakage Audit Protocol (LAP) — SAFE vs LEAKED connectome evaluation."""

from .protocol import (
    LAPConfig,
    make_full_connectome_builder,
    make_leaked_topk_builder,
    make_topk_builder,
    nested_evaluate,
    residualize_edges,
    run_safe_vs_leaked_topk,
    train_cobre_test_external,
    vec_upper,
)

__all__ = [
    "LAPConfig",
    "vec_upper",
    "residualize_edges",
    "make_topk_builder",
    "make_leaked_topk_builder",
    "make_full_connectome_builder",
    "nested_evaluate",
    "run_safe_vs_leaked_topk",
    "train_cobre_test_external",
]
