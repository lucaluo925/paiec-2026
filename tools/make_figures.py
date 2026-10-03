"""Learning curves and reliability diagrams from pooled cross-validation records.

    python tools/make_figures.py --records results/cv_records.npz \
        --curves irt empirical_mean irt_pairscope --calibrate irt \
        --ab "irt:random" "irt+fisher:Fisher information" --outdir figures

Colours: validated categorical slots 1-3 (blue, orange, aqua) on the light
surface; the constant-0.5 reference is a thin grey line. Every series carries
a direct end label and a legend entry; the numbers are in results/by_budget.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BUDGETS = (0, 1, 3, 7, 15, 31)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, TEXT, TEXT2, GRID, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#9a9994"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "text.color": TEXT, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "legend.frameon": False, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def load(path):
    data = np.load(path)
    out = {}
    for key in data.files:
        name, b, kind = key.rsplit("|", 2)
        out.setdefault(name, {}).setdefault(int(b), {})[kind] = data[key]
    return out


def brier(entry) -> float:
    """Official pair-macro Brier when pair ids were recorded, else the plain mean."""
    sq = (entry["p"] - entry["y"]) ** 2
    pid = entry.get("pid")
    if pid is None or len(pid) == 0:
        return float(np.mean(sq)) if len(sq) else float("nan")
    n = int(pid.max()) + 1
    counts = np.bincount(pid, minlength=n)
    totals = np.bincount(pid, weights=sq, minlength=n)
    return float((totals[counts > 0] / counts[counts > 0]).mean())


def brier_curve(rec):
    return [brier(rec[b]) for b in BUDGETS]


def _spread(values, gap):
    """Nudge label positions apart by at least ``gap``, preserving order."""
    order = np.argsort(values)
    pos = np.array(values, dtype=float)[order]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    shift = (np.array(values)[order] - pos).mean()  # re-centre the block on the data
    out = np.empty_like(pos)
    out[order] = pos + shift
    return out


def curves_axes(ax, rec, names, labels, title):
    x = np.arange(len(BUDGETS))
    ax.axhline(0.25, color=REF, linewidth=1.0, zorder=1)
    ends = []
    for i, name in enumerate(names):
        y = brier_curve(rec[name])
        ax.plot(x, y, color=SERIES[i], linewidth=2.0, zorder=3, label=labels[i])
        ax.plot(x, y, "o", color=SERIES[i], markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
        ends.append(y[-1])
    lo, hi = ax.get_ylim()
    ax.text(x[-1], 0.25 + 0.012 * (hi - lo), "constant 0.5", color=TEXT2, ha="right", va="bottom", fontsize=8)
    for i, yl in enumerate(_spread(ends, 0.045 * (hi - lo))):
        ax.text(x[-1] + 0.18, yl, f"{labels[i]} {ends[i]:.3f}", color=TEXT, va="center", fontsize=8)
    ax.set_xticks(x, [str(b) for b in BUDGETS])
    ax.set_xlim(-0.3, len(BUDGETS) + 1.6)
    ax.set_xlabel("labels acquired for the subject-benchmark pair (budget)")
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_title(title, loc="left", fontsize=10, color=TEXT)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=len(names), fontsize=8)


def reliability(ax, p, y, budget, color, bscore):
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    ax.plot([0, 1], [0, 1], color=REF, linewidth=1.0, zorder=1)
    ece = 0.0
    for k in range(10):
        m = idx == k
        if not m.any():
            continue
        mp, my, n = p[m].mean(), y[m].mean(), m.sum()
        ece += n / len(p) * abs(mp - my)
        ax.plot(mp, my, "o", color=color, markersize=3 + 7 * np.sqrt(n / len(p)),
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(f"budget {budget}   ECE {ece:.3f}   Brier {bscore:.3f}", loc="left", fontsize=8.5)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--curves", nargs="+", required=True, help="variant names (<= 3)")
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--calibrate", required=True)
    ap.add_argument("--ab", nargs="*", default=[], help="name:label pairs for the acquisition A/B panel")
    ap.add_argument("--ab-records", help="records file for the A/B (defaults to --records)")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    rec = load(args.records)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    labels = args.labels or args.curves
    ncols = 2 if args.ab else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.4 * ncols, 4.4), squeeze=False)
    curves_axes(axes[0, 0], rec, args.curves, labels, "Local hold-out learning curves")
    if args.ab:
        ab_rec = load(args.ab_records) if args.ab_records else rec
        names = [s.split(":", 1)[0] for s in args.ab]
        ab_labels = [s.split(":", 1)[1] for s in args.ab]
        curves_axes(axes[0, 1], ab_rec, names, ab_labels, "Acquisition: random vs information-based")
    fig.tight_layout()
    fig.savefig(out / "learning_curves.pdf")
    fig.savefig(out / "learning_curves.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.6))
    for ax, b in zip(axes.flat, BUDGETS):
        entry = rec[args.calibrate][b]
        reliability(ax, entry["p"], entry["y"], b, SERIES[0], brier(entry))
    for ax in axes[1]:
        ax.set_xlabel("mean predicted probability")
    for ax in axes[:, 0]:
        ax.set_ylabel("observed frequency")
    fig.suptitle(f"Reliability of '{args.calibrate}' by budget (marker area ~ bin count)", x=0.01, ha="left",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "calibration.pdf")
    fig.savefig(out / "calibration.png", dpi=160)
    plt.close(fig)
    print(f"wrote {out}/learning_curves.pdf, {out}/calibration.pdf")


if __name__ == "__main__":
    main()
