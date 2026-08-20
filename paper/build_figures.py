"""Figures for the MLMI 2026 camera-ready paper.

All values are read from results/*.csv or from the reported tables of the
accepted manuscript. Nothing here is simulated.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

INK = "#14181d"
BLUE = "#2f5d8c"
RED = "#9d3a32"
AMBER = "#b07d2b"
GREEN = "#2f6f5e"
GRAY = "#7c8894"
LIGHT = "#d9dee4"

available = {f.name for f in font_manager.fontManager.ttflist}
SERIF = next((f for f in ("Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif")
              if f in available), "serif")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [SERIF],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.8,
    "axes.linewidth": 0.7,
    "axes.edgecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.1,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 400,
    "figure.dpi": 200,
})


def save(fig, stem: str):
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote", stem)


def panel(ax, letter, dx=-0.085, dy=1.045):
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontweight="bold",
            fontsize=9, va="bottom", ha="left")


# --------------------------------------------------------------------------
# Fig. 1 — validation design
# --------------------------------------------------------------------------
def fig_design():
    fig, ax = plt.subplots(figsize=(6.9, 2.35))
    ax.set_xlim(0, 100)
    ax.set_ylim(-1, 35)
    ax.axis("off")

    def box(x, y, w, h, title, body, edge=INK, lw=0.8, fill="white"):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge,
                               linewidth=lw, zorder=2))
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top",
                fontsize=8.0, fontweight="bold", color=INK, zorder=3)
        ax.text(x + w / 2, y + h / 2 - 2.2, body, ha="center", va="center",
                fontsize=6.9, color="#33404d", linespacing=1.30, zorder=3)

    def arrow(x0, y0, x1, y1, style="-|>", color=INK, ls="-"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                     mutation_scale=8, lw=0.8, color=color,
                                     linestyle=ls, zorder=1))

    box(0.5, 18, 21, 16, "COBRE cohort",
        "n = 146\nSchaefer-100\n4,950 edges")
    box(27, 18, 21, 16, "Exploratory inference",
        "covariate-adjusted OLS\nFDR q < 0.05\n29 → 5 edges")
    box(27, 0, 21, 16, "Leakage-free prediction",
        "nested 5 × 10 CV\nconstruction inside\ntraining folds only")
    box(54.5, 18, 21, 16, "Protocol family",
        "elastic net → frozen\nedges → dataset-level\nselection")
    box(54.5, 0, 21, 16, "Matched contrast",
        "top-20 |t| + logistic\nselection inside vs.\noutside the fold")
    box(81, 9, 18.5, 16, "Audit output",
        "honest AUC,\noptimism Δ,\ncalibration")

    arrow(21.5, 26, 27, 26)
    arrow(21.5, 24.5, 27, 8)
    arrow(48, 26, 54.5, 26)
    arrow(48, 8, 54.5, 8)
    arrow(75.5, 26, 81, 20)
    arrow(75.5, 8, 81, 14)
    save(fig, "fig1_design")


# --------------------------------------------------------------------------
# Fig. 2 — exploratory connectome
# --------------------------------------------------------------------------
EDGES = [
    ("7Networks_LH_Vis_3", "7Networks_RH_Default_Temp_2", 4.38),
    ("7Networks_LH_SalVentAttn_FrOperIns_1", "7Networks_RH_Default_PFCv_1", -4.93),
    ("7Networks_LH_Default_Par_2", "7Networks_RH_Cont_PFCl_1", -4.30),
    ("7Networks_RH_Vis_8", "7Networks_RH_Default_Temp_2", -4.33),
    ("7Networks_RH_SalVentAttn_TempOccPar_2", "7Networks_RH_Default_PFCv_1", -4.65),
]
SHORT = {
    "7Networks_LH_Vis_3": "LH Vis 3",
    "7Networks_RH_Default_Temp_2": "RH DMN Temp 2",
    "7Networks_LH_SalVentAttn_FrOperIns_1": "LH SAL FrOperIns 1",
    "7Networks_RH_Default_PFCv_1": "RH DMN PFCv 1",
    "7Networks_LH_Default_Par_2": "LH DMN Par 2",
    "7Networks_RH_Cont_PFCl_1": "RH FPC PFCl 1",
    "7Networks_RH_Vis_8": "RH Vis 8",
    "7Networks_RH_SalVentAttn_TempOccPar_2": "RH SAL TempOccPar 2",
}
NET_COL = {"Vis": "#6e4b9e", "SalVentAttn": "#2f6f5e", "Default": "#9d3a32", "Cont": "#b07d2b"}


def net_of(name):
    for key in ("Vis", "SalVentAttn", "Default", "Cont"):
        if f"_{key}_" in name:
            return key
    return "Cont"


def fig_connectome():
    from nilearn import plotting

    cent = pd.read_csv(HERE / "Schaefer100_centroids.csv")
    cent["ROI Name"] = cent["ROI Name"].str.strip()
    pos = {r["ROI Name"]: (float(r["R"]), float(r["A"]), float(r["S"]))
           for _, r in cent.iterrows()}

    nodes = []
    for a, b, _ in EDGES:
        for n in (a, b):
            if n not in nodes:
                nodes.append(n)
    idx = {n: i for i, n in enumerate(nodes)}
    coords = np.array([pos[n] for n in nodes])
    adj = np.zeros((len(nodes), len(nodes)))
    for a, b, t in EDGES:
        adj[idx[a], idx[b]] = t
        adj[idx[b], idx[a]] = t

    fig = plt.figure(figsize=(6.9, 2.05))
    ax_b = fig.add_axes([0.005, 0.10, 0.615, 0.86])
    ax_c = fig.add_axes([0.655, 0.10, 0.33, 0.86])

    plotting.plot_connectome(
        adj, coords, display_mode="lzr", axes=ax_b,
        node_color=[NET_COL[net_of(n)] for n in nodes],
        node_size=32, edge_cmap=plt.get_cmap("coolwarm"),
        edge_vmin=-5.0, edge_vmax=5.0, edge_kwargs={"linewidth": 1.6},
        colorbar=False, annotate=False, black_bg=False,
    )

    ang = np.linspace(np.pi / 2, 2.5 * np.pi, len(nodes), endpoint=False)
    xy = {n: (np.cos(t), np.sin(t)) for n, t in zip(nodes, ang)}
    ax_c.add_patch(plt.Circle((0, 0), 1.0, fill=False, ec=LIGHT, lw=0.8))
    for a, b, t in EDGES:
        x0, y0 = xy[a]
        x1, y1 = xy[b]
        ax_c.annotate("", xy=(x1, y1), xytext=(x0, y0),
                      arrowprops=dict(arrowstyle="-", color=RED if t > 0 else BLUE,
                                      lw=0.7 + 0.30 * abs(t),
                                      connectionstyle="arc3,rad=0.2",
                                      shrinkA=6, shrinkB=6, alpha=0.9))
    for n, (x, y) in xy.items():
        ax_c.scatter([x], [y], s=34, c=NET_COL[net_of(n)], edgecolors="white",
                     linewidths=0.6, zorder=5)
        ha = "left" if x > 0.08 else ("right" if x < -0.08 else "center")
        ax_c.text(x * 1.16, y * 1.16, SHORT[n], ha=ha, va="center", fontsize=6.1,
                  color=INK)
    ax_c.set_xlim(-2.05, 2.05)
    ax_c.set_ylim(-1.55, 1.55)
    ax_c.set_aspect("equal")
    ax_c.axis("off")

    handles = [
        Line2D([0], [0], color=RED, lw=1.8, label="SZ > HC"),
        Line2D([0], [0], color=BLUE, lw=1.8, label="SZ < HC"),
        Line2D([0], [0], marker="o", ls="", mfc=NET_COL["Vis"], mec="white", ms=5, label="visual"),
        Line2D([0], [0], marker="o", ls="", mfc=NET_COL["SalVentAttn"], mec="white", ms=5, label="salience/VA"),
        Line2D([0], [0], marker="o", ls="", mfc=NET_COL["Default"], mec="white", ms=5, label="default"),
        Line2D([0], [0], marker="o", ls="", mfc=NET_COL["Cont"], mec="white", ms=5, label="control"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               fontsize=6.8, bbox_to_anchor=(0.5, -0.02), handletextpad=0.4,
               columnspacing=1.3)
    save(fig, "fig2_connectome")


# --------------------------------------------------------------------------
# Fig. 3 — leakage-free method comparison
# --------------------------------------------------------------------------
METHOD_ROWS = [
    ("fixed5_edges_demo", "Fixed 5 FDR edges + demo", True),
    ("pca30_edges_demo", "PCA-30 + demo", False),
    ("elasticnet_4950edges_demo", "Elastic net + demo", False),
    ("elasticnet_4950edges", "Elastic net (edges only)", False),
    ("fixed5_edges", "Fixed 5 FDR edges", True),
    ("cpm10pct_demo", "CPM 10% + demo", False),
    ("cpm10pct", "CPM 10%", False),
    ("top20_train_t_demo", r"Top-20 $|t|$ + demo", False),
    ("top20_train_t", r"Top-20 $|t|$", False),
    ("network28_demo", "Network-28 + demo", False),
    ("network28", "Network-28", False),
]


def fig_methods():
    df = pd.read_csv(RESULTS / "methods_benchmark_comparison.csv")
    rows = []
    for key, lab, circ in METHOD_ROWS:
        sub = df[df.method == key]
        if sub.empty:
            continue
        best = sub.sort_values("auc_mean", ascending=False).iloc[0]
        rows.append((lab, float(best.auc_mean), float(best.auc_std), circ))
    rows = list(reversed(rows))

    fig, ax = plt.subplots(figsize=(5.0, 3.45))
    y = np.arange(len(rows))
    for yi, (lab, m, s, circ) in zip(y, rows):
        col = RED if circ else BLUE
        ax.errorbar(m, yi, xerr=s, fmt="none", ecolor=col, elinewidth=0.9,
                    capsize=2.2, capthick=0.9, alpha=0.85, zorder=2)
        ax.plot([m], [yi], marker="D" if circ else "o", ms=4.4,
                mfc="white" if circ else col, mec=col, mew=1.1, zorder=3)
        ax.text(m + s + 0.012, yi, f"{m:.3f}", va="center", fontsize=7, color=INK)
    ax.axvline(0.5, ls=(0, (4, 3)), lw=0.7, color=GRAY, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(0.46, 0.97)
    ax.set_xlabel("Nested cross-validated AUC (fold mean $\\pm$ SD)")
    ax.grid(axis="x", color=LIGHT, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    handles = [
        Line2D([0], [0], marker="o", color=BLUE, ls="", ms=4.4, label="training-fold construction"),
        Line2D([0], [0], marker="D", color=RED, ls="", ms=4.4, mfc="white",
               label="full-cohort edge identities"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    save(fig, "fig3_methods")


# --------------------------------------------------------------------------
# Fig. 4 — leakage regimes
# --------------------------------------------------------------------------
def fig_leakage():
    leak = pd.read_csv(RESULTS / "lap_controlled_leakage.csv")
    safe = leak[leak.name.str.contains("SAFE")].iloc[0]
    leaked = leak[leak.name.str.contains("LEAKED")].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.75))

    ax = axes[0]
    labs = ["elastic net\n(leakage-free)", "frozen FDR\nedges"]
    vals = [0.768, 0.855]
    errs = [0.076, 0.066]
    cols = [BLUE, AMBER]
    hatch = ["", "///"]
    for i in range(2):
        ax.bar(i, vals[i], yerr=errs[i], width=0.55, color=cols[i], alpha=0.85,
               edgecolor=INK, lw=0.7, capsize=2.6, hatch=hatch[i],
               error_kw={"elinewidth": 0.8}, zorder=2)
        ax.text(i, vals[i] + errs[i] + 0.022, f"{vals[i]:.3f}", ha="center", fontsize=7.4)
    ax.annotate("", xy=(1, 1.012), xytext=(0, 1.012),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color=INK))
    ax.text(0.5, 1.028, "$\\Delta$ = +0.087", ha="center", fontsize=7.4)
    ax.axhline(0.5, ls=(0, (4, 3)), lw=0.7, color=GRAY, zorder=1)
    ax.set_xticks(range(2))
    ax.set_xticklabels(labs)
    ax.set_ylim(0.45, 1.09)
    ax.set_ylabel("Nested CV AUC")
    panel(ax, "a")

    ax = axes[1]
    vals2 = [float(safe.auc_mean), float(leaked.auc_mean)]
    errs2 = [float(safe.auc_std), float(leaked.auc_std)]
    for i, (c, h) in enumerate(zip([BLUE, RED], ["", "xxx"])):
        ax.bar(i, vals2[i], yerr=errs2[i], width=0.46, color=c, alpha=0.85,
               edgecolor=INK, lw=0.7, capsize=2.6, hatch=h,
               error_kw={"elinewidth": 0.8}, zorder=2)
        ax.text(i, vals2[i] + errs2[i] + 0.022, f"{vals2[i]:.3f}", ha="center", fontsize=7.4)
    ax.annotate("", xy=(1, 1.012), xytext=(0, 1.012),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color=INK))
    ax.text(0.5, 1.028, f"$\\Delta$ = +{vals2[1] - vals2[0]:.3f}", ha="center", fontsize=7.4)
    ax.axhline(0.5, ls=(0, (4, 3)), lw=0.7, color=GRAY, zorder=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["selection inside\nthe training fold",
                        "selection on the\nfull cohort"])
    ax.set_ylim(0.45, 1.09)
    panel(ax, "b")
    fig.tight_layout(w_pad=2.4)
    save(fig, "fig4_leakage")


# --------------------------------------------------------------------------
# Fig. 5 — motion sensitivity and locked splits
# --------------------------------------------------------------------------
def fig_motion():
    sens = pd.read_csv(RESULTS / "lap_residualisation_sensitivity.csv")
    lock = pd.read_csv(RESULTS / "lap_locked_split_cobre.csv")

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.75))

    ax = axes[0]
    want = [("en_demo", "EN\n+ demo"), ("en_edges", "EN\nedges"),
            ("en_edges_FDresid", "EN edges\nFD-resid."),
            ("top20_SAFE_raw", "Top-20\nedges"),
            ("top20_SAFE_FDresid", "Top-20\nFD-resid.")]
    cols = [BLUE, BLUE, GREEN, GRAY, GREEN]
    for i, ((k, lab), c) in enumerate(zip(want, cols)):
        r = sens[sens.name == k].iloc[0]
        ax.bar(i, float(r.auc_mean), yerr=float(r.auc_std), width=0.6, color=c,
               alpha=0.85, edgecolor=INK, lw=0.7, capsize=2.4,
               error_kw={"elinewidth": 0.8}, zorder=2)
        ax.text(i, float(r.auc_mean) + float(r.auc_std) + 0.02,
                f"{float(r.auc_mean):.3f}", ha="center", fontsize=7)
    ax.axhline(0.5, ls=(0, (4, 3)), lw=0.7, color=GRAY, zorder=1)
    ax.set_xticks(range(len(want)))
    ax.set_xticklabels([w[1] for w in want])
    ax.set_ylim(0.45, 0.95)
    ax.set_ylabel("Nested CV AUC (5 $\\times$ 3)")
    panel(ax, "a")

    ax = axes[1]
    for mode, lab, c, mk in [("elasticnet_full", "elastic net", BLUE, "o"),
                             ("top20_locked", r"top-20 $|t|$", AMBER, "s")]:
        sub = lock[lock["mode"] == mode].sort_values("split")
        ax.plot(sub["split"], sub["auc"], marker=mk, ms=4, color=c, label=lab, lw=1.0)
        m = float(sub["auc"].mean())
        ax.axhline(m, color=c, ls=(0, (4, 3)), lw=0.7, alpha=0.8)
        ax.text(5.30, m, f"mean {m:.3f}", color=c, fontsize=6.8, va="center",
                ha="left", bbox=dict(fc="white", ec="none", pad=0.8))
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlim(0.7, 6.6)
    ax.set_ylim(0.60, 0.85)
    ax.set_xlabel("Locked 70/30 split")
    ax.set_ylabel("Test AUC")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color=LIGHT, lw=0.5)
    ax.set_axisbelow(True)
    panel(ax, "b")
    fig.tight_layout(w_pad=2.4)
    save(fig, "fig5_motion")


# --------------------------------------------------------------------------
# Fig. 6 — leaked reference coefficients
# --------------------------------------------------------------------------
COEFS = [
    ("Mean FD", 0.174, 0.125),
    ("LH Vis 3 – RH DMN Temp 2", 0.169, 0.103),
    ("RH SAL TempOccPar 1 – RH Limbic", -0.148, 0.102),
    ("LH SAL PFCl 1 – RH Limbic TempPole", -0.142, 0.091),
    ("LH SomMot 3 – LH SAL PFCl 1", 0.134, 0.064),
    ("LH Vis 8 – RH FPC PFCl 4", 0.131, 0.061),
    ("LH Limbic TempPole 1 – LH DMN PFC", -0.131, 0.059),
    ("LH DMN Par 2 – RH FPC PFCl 1", -0.127, 0.080),
]


def fig_coefficients():
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    rows = list(reversed(COEFS))
    y = np.arange(len(rows))
    for yi, (lab, m, s) in zip(y, rows):
        motion = lab == "Mean FD"
        col = RED if motion else BLUE
        ax.errorbar(m, yi, xerr=s, fmt="none", ecolor=col, elinewidth=0.9,
                    capsize=2.2, capthick=0.9, alpha=0.85)
        ax.plot([m], [yi], marker="s" if motion else "o", ms=4.4, color=col,
                mec=col, mfc=col if not motion else "white", mew=1.1)
    ax.axvline(0, color=INK, lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_xlabel("Logistic coefficient (mean $\\pm$ SD across outer folds)")
    ax.set_xlim(-0.30, 0.34)
    ax.grid(axis="x", color=LIGHT, lw=0.5)
    ax.set_axisbelow(True)
    handles = [Line2D([0], [0], marker="s", ls="", mfc="white", mec=RED, ms=4.4,
                      label="head-motion covariate"),
               Line2D([0], [0], marker="o", ls="", color=BLUE, ms=4.4, label="connectivity edge")]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    save(fig, "fig6_coefficients")


# --------------------------------------------------------------------------
# Fig. 7 — calibration
# --------------------------------------------------------------------------
def fig_calibration():
    sens = pd.read_csv(RESULTS / "lap_residualisation_sensitivity.csv").set_index("name")
    ctl = pd.read_csv(RESULTS / "lap_controlled_leakage.csv").set_index("name")

    models = [
        ("Elastic net\n+ demo", sens.loc["en_demo"], BLUE),
        ("Elastic net\nedges only", sens.loc["en_edges"], BLUE),
        (r"Top-20 $|t|$" + "\nleakage-free", ctl.loc["top20_SAFE"], GRAY),
        (r"Top-20 $|t|$" + "\nleaked", ctl.loc["top20_LEAKED"], RED),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.6))

    ax = axes[0]
    x = np.arange(len(models))
    w = 0.38
    ece = [float(r.ece) for _, r, _ in models]
    brier = [float(r.brier) for _, r, _ in models]
    ax.bar(x - w / 2, ece, w, color=BLUE, alpha=0.85, edgecolor=INK, lw=0.7, label="ECE")
    ax.bar(x + w / 2, brier, w, color=GRAY, alpha=0.85, edgecolor=INK, lw=0.7,
           hatch="///", label="Brier score")
    for xi, (a, b) in enumerate(zip(ece, brier)):
        ax.text(xi - w / 2, a + 0.006, f"{a:.3f}", ha="center", fontsize=6.5)
        ax.text(xi + w / 2, b + 0.006, f"{b:.3f}", ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in models])
    ax.set_ylim(0, 0.30)
    ax.set_ylabel("Calibration metric")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.grid(axis="y", color=LIGHT, lw=0.5)
    ax.set_axisbelow(True)
    panel(ax, "a")

    ax = axes[1]
    naive = [float(r.auc_pooled_oof_naive) for _, r, _ in models]
    subj = [float(r.auc_subject_level_oof) for _, r, _ in models]
    ax.bar(x - w / 2, naive, w, color=BLUE, alpha=0.85, edgecolor=INK, lw=0.7,
           label="pooled over folds")
    ax.bar(x + w / 2, subj, w, color=GREEN, alpha=0.85, edgecolor=INK, lw=0.7,
           hatch="\\\\\\", label="subject-averaged")
    for xi, (a, b) in enumerate(zip(naive, subj)):
        ax.text(xi - w / 2, a + 0.012, f"{a:.3f}", ha="center", fontsize=6.5)
        ax.text(xi + w / 2, b + 0.012, f"{b:.3f}", ha="center", fontsize=6.5)
    ax.axhline(0.5, ls=(0, (4, 3)), lw=0.7, color=GRAY)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in models])
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("Out-of-fold AUC")
    ax.legend(frameon=False, loc="upper left", ncol=1)
    ax.grid(axis="y", color=LIGHT, lw=0.5)
    ax.set_axisbelow(True)
    panel(ax, "b")

    fig.tight_layout(w_pad=2.2)
    save(fig, "fig7_calibration")


# --------------------------------------------------------------------------
# Fig. 8 — protocol check and null control
# --------------------------------------------------------------------------
COBRE_LEAKED = 0.903
COBRE_SAFE = 0.709


def fig_external():
    matched = pd.read_csv(RESULTS / "null_control_matched.csv").set_index("cell")
    corr = pd.read_csv(RESULTS / "null_control_connectome.csv").set_index("cell")
    by_k = pd.read_csv(RESULTS / "null_control_by_k.csv")
    by_n = pd.read_csv(RESULTS / "null_control_by_n.csv")
    cnp_path = RESULTS / "external_cnp_matched50_contrast.csv"
    if not cnp_path.exists():
        cnp_path = RESULTS / "external_cnp_contrast.csv"
    cnp = pd.read_csv(cnp_path)
    cnp_safe = float(cnp[cnp.name.str.contains("SAFE")].auc_mean.iloc[0])
    cnp_leaked = float(cnp[cnp.name.str.contains("LEAKED")].auc_mean.iloc[0])

    perm_path = RESULTS / "null_control_cnp_permutation.csv"
    cnp_ind_key = "CNP-matched50" if "CNP-matched50" in matched.index else (
        "CNP-matched" if "CNP-matched" in matched.index else "COBRE-matched"
    )
    cnp_corr_key = "CNP-matched50, correlated edges" if "CNP-matched50, correlated edges" in corr.index else (
        "CNP-matched, correlated edges" if "CNP-matched, correlated edges" in corr.index
        else "COBRE-matched, correlated edges"
    )

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.5))

    ax = axes[0]
    x = np.arange(len(by_k))
    w = 0.38
    ax.bar(x - w / 2, by_k.auc_safe_mean, w, yerr=by_k.auc_safe_std, color=BLUE,
           alpha=0.85, edgecolor=INK, lw=0.7, capsize=2.2,
           error_kw={"elinewidth": 0.7}, label="leakage-free")
    ax.bar(x + w / 2, by_k.auc_leaked_mean, w, yerr=by_k.auc_leaked_std, color=RED,
           alpha=0.85, edgecolor=INK, lw=0.7, capsize=2.2, hatch="xxx",
           error_kw={"elinewidth": 0.7}, label="leaked")
    for xi, v in zip(x, by_k.auc_leaked_mean):
        ax.text(xi + w / 2, v + 0.028, f"{v:.3f}", ha="center", fontsize=6.4)
    ax.axhline(0.5, ls=(0, (4, 3)), lw=0.7, color=GRAY)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$k$ = {int(v)}" for v in by_k.k])
    ax.set_ylim(0.40, 1.20)
    ax.set_ylabel("AUC, no signal present")
    ax.legend(frameon=False, loc="upper left", ncol=2, columnspacing=0.9,
              handlelength=1.2, fontsize=6.8, borderpad=0.1)
    panel(ax, "a")

    ax = axes[1]
    groups = [
        ("COBRE\n$n$ = 146", COBRE_SAFE, COBRE_LEAKED, "COBRE-matched",
         "COBRE-matched, correlated edges", None),
        ("UCLA CNP\n50 / 50", cnp_safe, cnp_leaked, cnp_ind_key, cnp_corr_key,
         perm_path if perm_path.exists() else None),
    ]
    for i, (lab, s, lk, key_ind, key_corr, perm_p) in enumerate(groups):
        lo = float(corr.loc[key_corr, "auc_leaked_mean"])
        if perm_p is not None:
            pr = pd.read_csv(perm_p).iloc[0]
            lo = min(lo, float(pr.auc_leaked_mean))
        hi = float(matched.loc[key_ind, "auc_leaked_mean"])
        lo, hi = (lo, hi) if lo <= hi else (hi, lo)
        ax.add_patch(Rectangle((i - 0.28, lo), 0.56, hi - lo, facecolor=RED,
                               alpha=0.16, edgecolor=RED, lw=0.7, ls=(0, (2, 1.5)),
                               zorder=1,
                               label="leaked, no signal present" if i == 0 else None))
        ax.plot([i], [lk], marker="s", ms=5.2, color=RED, ls="", zorder=3,
                label="leaked, observed" if i == 0 else None)
        ax.plot([i], [s], marker="o", ms=5.2, color=BLUE, ls="", zorder=3,
                label="leakage-free, observed" if i == 0 else None)
        ax.text(i + 0.33, lk, f"{lk:.3f}", va="center", fontsize=6.4, color=RED)
        ax.text(i + 0.33, s, f"{s:.3f}", va="center", fontsize=6.4, color=BLUE)
        ax.text(i - 0.33, (lo + hi) / 2, f"{lo:.2f}–{hi:.2f}", va="center",
                ha="right", fontsize=6.2, color=RED)
    ax.axhline(0.5, ls=(0, (4, 3)), lw=0.7, color=GRAY)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_xlim(-0.75, 1.75)
    ax.set_ylim(0.45, 1.10)
    ax.set_ylabel("AUC")
    ax.legend(frameon=False, loc="upper center", fontsize=6.2, handlelength=1.1,
              borderpad=0.1, labelspacing=0.22, handletextpad=0.5)
    panel(ax, "b")

    ax = axes[2]
    ax.plot(by_n.n, by_n.optimism_mean, marker="o", ms=3.6, color=RED,
            label="null data")
    ax.fill_between(by_n.n, by_n.optimism_mean - by_n.optimism_std,
                    by_n.optimism_mean + by_n.optimism_std, color=RED, alpha=0.15,
                    lw=0)
    ax.plot([146], [COBRE_LEAKED - COBRE_SAFE], marker="D", ms=4.4, mfc="white",
            mec=BLUE, mew=1.1, ls="", label="COBRE")
    ax.plot([100], [cnp_leaked - cnp_safe], marker="D", ms=4.4, mfc="white",
            mec=RED, mew=1.1, ls="", label="UCLA CNP 50/50")
    ax.set_xscale("log")
    ax.set_xticks([50, 100, 200, 400, 800])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Sample size")
    ax.set_ylabel("Optimism (AUC)")
    ax.set_ylim(0, 0.62)
    ax.grid(color=LIGHT, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=6.8)
    panel(ax, "c")

    fig.tight_layout(w_pad=1.9)
    save(fig, "fig8_external")


if __name__ == "__main__":
    print(f"serif font: {SERIF}")
    fig_design()
    fig_connectome()
    fig_methods()
    fig_leakage()
    fig_motion()
    fig_coefficients()
    fig_calibration()
    if (RESULTS / "external_cnp_matched50_contrast.csv").exists() or (
            RESULTS / "external_cnp_contrast.csv").exists():
        fig_external()
    print("done ->", FIG)
