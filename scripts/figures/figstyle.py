"""One place for the look of every data figure in the paper.

Keeping the palette and the marker sizes here means a reader can tell two lines
apart the same way in every figure, and that a change of taste is one edit
rather than five.  Conceptual diagrams are drawn elsewhere and do not use this.
"""
import matplotlib.pyplot as plt

#: series colours, in the order a figure should consume them; the method itself is last
PALETTE = ["#8b5fbf", "#2e9e4f", "#e08a1e", "#d1495b"]
MARKERS = ["^", "s", "v", "o"]
OURS = "#d1495b"          #: reserved for GUARD, so it reads the same everywhere
GREY = "#8c8c8c"          #: reference lines, budgets, diagonals -- never a series
FAIL = "#333333"          #: a regime the method does not cover, e.g. broken exchangeability

LW, LW_OURS = 1.0, 1.2
MS, MS_OURS = 2.4, 2.6

def apply():
    """Set the shared rcParams. Call once, before any axes are created."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })

def series(i, ours=False):
    """Keyword arguments for the i-th series of a plot."""
    if ours:
        return dict(color=OURS, marker="o", ms=MS_OURS, lw=LW_OURS)
    return dict(color=PALETTE[i % len(PALETTE)], marker=MARKERS[i % len(MARKERS)],
                ms=MS, lw=LW)

def legend_below(fig, ax, ncol, y=-0.04, fontsize=6.8):
    """One shared legend under the whole figure rather than one per panel."""
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=ncol, frameon=False, fontsize=fontsize,
               bbox_to_anchor=(0.5, y), handletextpad=0.5, columnspacing=1.3)
