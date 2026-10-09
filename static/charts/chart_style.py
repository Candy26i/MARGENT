"""Shared matplotlib style for the project-page figures.

Every figure is a static, light-surface image saved twice (SVG for the page,
PNG for previews) into ``static/images/``.  Text is converted to paths in the
SVG so the figures render identically everywhere.

Colour roles are fixed here so the same thing has the same colour in every
figure.  The hues are the validated categorical palette of the dataviz skill
(adjacent-pair order is colour-blind safe).

``wilson`` and ``mcnemar_exact`` are the interval and the test the MedQA
figures report; they are the functions of ``scripts/analyze_results.py`` on the
MARGENT ``legacy`` branch (tag ``v0.1-full``), copied here so the figures stay
regenerable from this directory alone.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "images"

# surfaces and ink (match static/css/style.css)
SURFACE = "#ffffff"
INK = "#2e3238"
INK2 = "#4a4f57"
MUTED = "#737a84"
RULE = "#e6e8ec"
GRID = "#eef0f3"

# categorical slots, fixed order
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

# role assignments shared by all figures
STAGE = {"base": BLUE, "cold": ORANGE, "grpo": AQUA}          # pipeline stages
FAMILY = {"Generic": BLUE, "Specific": ORANGE}                 # advisor family
BENCH = {"MedQA": BLUE, "LegalBench": ORANGE, "GPQA": AQUA, "MMLU-Pro": VIOLET}
SIZE_MARKER = {"4B": "o", "8B": "s", "9B": "D"}                # manager size
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
NEUTRAL = "#c9ced6"          # reference / no-change marks
NOISE_BAND = (0.75, 0.29, 0.37, 0.10)  # translucent band for "indistinguishable from 0"

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.labelcolor": INK2,
    "axes.edgecolor": RULE,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.major.pad": 6,
    "ytick.major.pad": 6,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "legend.fontsize": 9.5,
    "legend.handlelength": 1.2,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "svg.fonttype": "path",
    "text.color": INK,
}


def apply_style() -> None:
    plt.rcParams.update(RC)


def ygrid(ax, zero=True) -> None:
    """Recessive horizontal grid behind the marks, optional emphasised zero line."""
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    if zero:
        ax.axhline(0, color=INK2, linewidth=0.9, zorder=1)


def xgrid(ax, zero=True) -> None:
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    if zero:
        ax.axvline(0, color=INK2, linewidth=0.9, zorder=1)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    svg_path = OUT / f"{name}.svg"
    fig.savefig(svg_path)
    # drop matplotlib's DOCTYPE line: some hosts refuse SVG files with DTD declarations
    svg = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(re.sub(r"<!DOCTYPE[^>]*>\s*", "", svg, flags=re.S), encoding="utf-8")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"saved {name}.svg / .png")


# statistics (stdlib only)
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes out of n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)
