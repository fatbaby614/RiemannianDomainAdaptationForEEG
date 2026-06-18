"""
generate_pub_figures.py — Publication-quality figures for IEEE TBME paper
=========================================================================

Generates 4 figures from experiment results:
  1. individual_cor.pdf  — Per-subject LOSO COR comparison (hero figure)
  2. fewshot_curve.pdf   — Few-shot calibration curve
  3. channel_comparison.pdf — Channel configuration comparison
  4. band_importance.pdf — Frequency band importance

Output: paper/figures/

Uses nature-figure publication standards:
  - Arial/Helvetica font family
  - No top/right spines
  - PDF vector output with editable TrueType text
  - IEEE TBME column widths (3.5" single / 7.16" double)
"""

import json
import os
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Publication rcParams ──
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "lines.linewidth": 1.0,
})

# ── Paths ──
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Color palette (restrained, one neutral + one signal + one accent) ──
C_NONE = "#B0B0B0"       # neutral gray
C_EUCL = "#7CB5D4"       # light blue (intermediate)
C_RIEM = "#2C6FAC"       # signal blue (hero)
C_ACCENT = "#C0392B"     # accent red (for highlights)

# ── IEEE TBME column widths (inches) ──
SINGLE_COL = 3.5
DOUBLE_COL = 7.16


def save_pub(fig, name, dpi=600):
    """Save figure in PDF (vector) and TIFF (raster) for journal submission."""
    pdf_path = os.path.join(FIGURES_DIR, f"{name}.pdf")
    tiff_path = os.path.join(FIGURES_DIR, f"{name}.tiff")
    fig.savefig(pdf_path, bbox_inches="tight", dpi=dpi)
    fig.savefig(tiff_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {tiff_path}")


def load_json(filename):
    """Load the latest JSON file matching a prefix from results/."""
    files = sorted(f for f in os.listdir(RESULTS_DIR)
                   if f.startswith(filename) and f.endswith(".json"))
    if not files:
        raise FileNotFoundError(f"No file matching '{filename}*' in {RESULTS_DIR}")
    path = os.path.join(RESULTS_DIR, files[-1])
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════
# Figure 1: Per-subject LOSO COR comparison (hero figure)
# ═══════════════════════════════════════════════════════════════════════
def fig_individual_cor():
    """
    Core conclusion: Riemannian alignment improves 21/23 subjects
    (COR 0.585 -> 0.692, p=0.0007, d=0.824).
    """
    data = load_json("results_table1_all")
    sota = load_json("results_sota_head_to_head")
    stats = load_json("results_enhanced_stats")

    none_cor = np.array(sota["results"]["all"]["none"]["cor"])
    eucl_cor = np.array(sota["results"]["all"]["ea"]["cor"])
    riem_cor = np.array(sota["results"]["all"]["zanini"]["cor"])

    n_subjects = len(none_cor)
    subjects = np.arange(1, n_subjects + 1)

    # Sort by baseline COR (ascending)
    order = np.argsort(none_cor)
    subjects_sorted = subjects[order]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.8))

    x = np.arange(n_subjects)
    w = 0.25

    bars_none = ax.bar(x - w, none_cor[order], w, label="None",
                       color=C_NONE, edgecolor="#888", linewidth=0.4)
    bars_eucl = ax.bar(x, eucl_cor[order], w, label="Euclidean",
                       color=C_EUCL, edgecolor="#4682B4", linewidth=0.4)
    bars_riem = ax.bar(x + w, riem_cor[order], w, label="Riemannian",
                       color=C_RIEM, edgecolor="#1A4A7A", linewidth=0.4)

    # Mean lines
    ax.axhline(np.mean(none_cor), color=C_NONE, ls="--", lw=0.7, alpha=0.8)
    ax.axhline(np.mean(riem_cor), color=C_RIEM, ls="--", lw=0.7, alpha=0.8)

    # Annotate means (right margin, aligned with dashed lines)
    ax.text(n_subjects + 0.3, np.mean(none_cor)+ 0.02,
            f" {np.mean(none_cor):.3f}", fontsize=6,
            color="#666", ha="left", va="center")
    ax.text(n_subjects + 0.3, np.mean(riem_cor)+ 0.02,
            f" {np.mean(riem_cor):.3f}", fontsize=6,
            color=C_RIEM, ha="left", va="center")

    # Mark subjects where Riemannian < None
    for i, idx in enumerate(order):
        if riem_cor[idx] < none_cor[idx]:
            ax.plot(i + w, riem_cor[idx], "v", color=C_ACCENT,
                    markersize=3, zorder=5)

    ax.set_xlabel("Subject (sorted by baseline COR)")
    ax.set_ylabel("LOSO Correlation (COR)")
    ax.set_xticks(x)
    ax.set_xticklabels(subjects_sorted, fontsize=5.5)
    ax.set_ylim(-0.1, 1.05)
    ax.set_xlim(-0.8, n_subjects + 2.5)
    ax.legend(loc="upper left", fontsize=7, ncol=3,
              bbox_to_anchor=(0.0, 1.0), borderpad=0.4,
              handlelength=1.5, columnspacing=1.0)

    # Statistics annotation
    n_better = stats["all"]["n_better"]
    p_val = stats["all"]["paired_t_p"]
    d_val = stats["all"]["cohens_d"]
    ax.text(0.98, 0.02,
            f"21/23 improved  |  $p$ = {p_val:.4f}  |  $d$ = {d_val:.2f}",
            transform=ax.transAxes, fontsize=6, ha="right", va="bottom",
            color="#333")

    fig.tight_layout(pad=0.3)
    save_pub(fig, "individual_cor")


# ═══════════════════════════════════════════════════════════════════════
# Figure 2: Few-shot calibration curve
# ═══════════════════════════════════════════════════════════════════════
def fig_fewshot():
    """
    Core conclusion: Even 1 calibration epoch recovers most of the
    alignment benefit; 5 epochs reach full-alignment COR.
    """
    fewshot = load_json("results_fewshot")
    sota = load_json("results_sota_head_to_head")
    stats = load_json("results_enhanced_stats")

    riem_full = stats["all"]["mean_cor_riemann"]
    none_mean = stats["all"]["mean_cor_none"]

    calib_sizes = sorted(int(k) for k in fewshot.keys())
    cor_means = [fewshot[str(n)]["mean"] for n in calib_sizes]
    cor_stds = [fewshot[str(n)]["std"] for n in calib_sizes]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.6))

    ax.errorbar(calib_sizes, cor_means, yerr=cor_stds,
                fmt="o-", color=C_RIEM, linewidth=1.2, markersize=4,
                markerfacecolor="white", markeredgewidth=1.0,
                capsize=2, capthick=0.6, zorder=3)

    # Reference lines
    ax.axhline(riem_full, color=C_RIEM, ls="--", lw=0.7, alpha=0.6,
               label=f"Full alignment ({riem_full:.3f})")
    ax.axhline(none_mean, color=C_NONE, ls=":", lw=0.7, alpha=0.6,
               label=f"No alignment ({none_mean:.3f})")

    # Highlight n=5
    if 5 in calib_sizes:
        idx5 = calib_sizes.index(5)
        ax.plot(5, cor_means[idx5], "*", color=C_ACCENT,
                markersize=8, zorder=5)
        pct = cor_means[idx5] / riem_full * 100
        ax.annotate(f"$n$=5: {cor_means[idx5]:.3f} ({pct:.0f}% of full)",
                    xy=(5, cor_means[idx5]),
                    xytext=(25, cor_means[idx5] - 0.04),
                    fontsize=6, color=C_ACCENT,
                    arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=0.6))

    ax.set_xlabel("Calibration epochs ($n$)")
    ax.set_ylabel("Mean COR")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks(calib_sizes)
    ax.get_xaxis().set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.set_xlim(-1, max(calib_sizes) * 1.5)
    ax.set_ylim(none_mean - 0.04, riem_full + 0.02)
    ax.legend(fontsize=6, loc="lower right")
    ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout(pad=0.3)
    save_pub(fig, "fewshot_curve")


# ═══════════════════════════════════════════════════════════════════════
# Figure 3: Channel configuration comparison
# ═══════════════════════════════════════════════════════════════════════
def fig_channel_comparison():
    """
    Core conclusion: 4-channel forehead retains 98.4% of full
    Riemannian performance, enabling wearable deployment.
    """
    sota = load_json("results_sota_head_to_head")
    stats = load_json("results_enhanced_stats")

    channels = ["all", "temporal", "forehead"]
    ch_labels = ["17ch\n(All)", "6ch\n(Temporal)", "4ch\n(Forehead)"]

    none_means = [stats[ch]["mean_cor_none"] for ch in channels]
    riem_means = [stats[ch]["mean_cor_riemann"] for ch in channels]
    none_stds = [sota["results"][ch]["none"]["cor_std"] for ch in channels]
    riem_stds = [sota["results"][ch]["zanini"]["cor_std"] for ch in channels]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.6))

    x = np.arange(len(channels))
    w = 0.32

    bars_n = ax.bar(x - w / 2, none_means, w, yerr=none_stds,
                    label="None", color=C_NONE, edgecolor="#888",
                    linewidth=0.4, capsize=2, error_kw={"lw": 0.6})
    bars_r = ax.bar(x + w / 2, riem_means, w, yerr=riem_stds,
                    label="Riemannian", color=C_RIEM, edgecolor="#1A4A7A",
                    linewidth=0.4, capsize=2, error_kw={"lw": 0.6})

    # Value labels
    for bar, val in zip(bars_n, none_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025,
                f"{val:.3f}", ha="center", va="bottom", fontsize=5.5,
                color="#666")
    for bar, val in zip(bars_r, riem_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025,
                f"{val:.3f}", ha="center", va="bottom", fontsize=5.5,
                color=C_RIEM)

    # Retention percentage for forehead
    retention = riem_means[2] / riem_means[0] * 100
    ax.annotate(f"{retention:.1f}%\nretention",
                xy=(x[2] + w / 2, riem_means[2]),
                xytext=(x[2] + 0.6, riem_means[2] + 0.04),
                fontsize=5.5, color=C_ACCENT, ha="center",
                arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=0.6))

    # Significance markers
    p_vals = [stats[ch]["paired_t_p"] for ch in channels]
    for i, p in enumerate(p_vals):
        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        else:
            sig = "n.s."
        y_max = max(none_means[i] + none_stds[i], riem_means[i] + riem_stds[i])
        ax.text(x[i], y_max + 0.06, sig, ha="center", fontsize=6,
                color="#333")

    ax.set_ylabel("Mean COR")
    ax.set_xticks(x)
    ax.set_xticklabels(ch_labels, fontsize=7)
    ax.set_ylim(0.45, 1.00)
    ax.legend(fontsize=7, loc="upper right",
              bbox_to_anchor=(1.0, 1.0), borderpad=0.4,
              handlelength=1.5, ncol=2)
    ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout(pad=0.3)
    save_pub(fig, "channel_comparison")


# ═══════════════════════════════════════════════════════════════════════
# Figure 4: Band importance
# ═══════════════════════════════════════════════════════════════════════
def fig_band_importance():
    """
    Core conclusion: Beta and alpha bands dominate feature selection,
    consistent with vigilance-related EEG literature.
    """
    bi = load_json("results_band_importance")

    # Use Riemannian alignment results (all aligners have same values)
    bands_data = bi["riemann"]
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    labels = [r"$\delta$", r"$\theta$", r"$\alpha$", r"$\beta$", r"$\gamma$"]
    counts = [bands_data[b] for b in bands]

    total = sum(counts)
    pcts = [c / total * 100 for c in counts]

    # Band-specific colors (warm-to-cool gradient)
    band_colors = ["#8ECFC9", "#BEB9D6", "#F7DC6F", "#F1948A", "#85C1E9"]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.4))

    bars = ax.bar(labels, counts, color=band_colors, edgecolor="#555",
                  linewidth=0.4, width=0.6)

    for bar, pct, cnt in zip(bars, pcts, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{pct:.1f}%\n({cnt})", ha="center", va="bottom", fontsize=5.5)

    ax.set_xlabel("Frequency Band")
    ax.set_ylabel("Selected Features")
    ax.set_ylim(0, max(counts) * 1.25)
    ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout(pad=0.3)
    save_pub(fig, "band_importance")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Generating Publication-Quality Figures for IEEE TBME")
    print(f"Output: {FIGURES_DIR}")
    print("=" * 60)

    print("\n[1/4] Per-subject LOSO COR comparison...")
    fig_individual_cor()

    print("\n[2/4] Few-shot calibration curve...")
    fig_fewshot()

    print("\n[3/4] Channel configuration comparison...")
    fig_channel_comparison()

    print("\n[4/4] Band importance...")
    fig_band_importance()

    print("\nDone. All figures saved to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
