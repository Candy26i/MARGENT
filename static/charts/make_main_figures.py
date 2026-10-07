"""Figures 1-5 for the "Learning When to Commit" project page.

Reads ``static/data/main_results.json`` (aggregate accuracies for the four
benchmarks), recomputes every derived statistic in code (binomial SEs, means
over the 28 GRPO cells, the cold-start / GRPO decomposition, paired
Specific - Generic differences, Pearson r), compares each one with the
"Verified numbers" of the content spec, and saves

  fig_main_results            4 small multiples: base / cold start / best GRPO cell
  fig_decomposition           cold-start share vs GRPO share of the gain, per benchmark
  fig_grpo_vs_cold            dot plot of GRPO - cold for all 28 cells
  fig_specific_minus_generic  8 paired Specific - Generic differences
  fig_base_vs_final           base accuracy vs GRPO accuracy, 28 cells

Run:
  /opt/anaconda3/bin/python3 static/charts/make_main_figures.py

A check that disagrees with the spec is printed as MISMATCH; the computed
value is always the one drawn.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chart_style as cs  # noqa: E402

DATA = HERE.parent / "data" / "main_results.json"

BENCHES = ["MedQA", "LegalBench", "GPQA", "MMLU-Pro"]
SIZES = ["4B", "8B", "9B"]
SIZE_NAME = {"4B": "Qwen3-4B", "8B": "Qwen3-8B", "9B": "Qwen3.5-9B"}
FAMILIES = ["Generic", "Specific"]
# fixed method order (family, size, GRPO group size); 7 methods x 4 benchmarks = 28 cells
METHODS = ["G-4B-6gen", "G-4B-8gen", "G-8B-4gen", "G-8B-6gen",
           "S-4B-6gen", "S-8B-6gen", "S-9B-6gen"]

# ----------------------------------------------------------------------------
# spec "Verified numbers" (the values the computed statistics are checked against)
# ----------------------------------------------------------------------------
SPEC = {
    "mean_d_base": 13.9,
    "mean_d_cold": 4.8,
    # bench: (total, cold part, GRPO part)
    "decomp": {"MedQA": (13.6, 12.4, 1.3), "LegalBench": (18.9, 7.6, 11.2),
               "GPQA": (5.1, 1.1, 4.0), "MMLU-Pro": (18.1, 15.4, 2.7)},
    # size: (mean d_base, mean d_cold)
    "by_size": {"4B": (16.0, 2.5), "8B": (14.2, 7.7), "9B": (7.0, 3.0)},
    # method: (avg score, d_base, d_cold)
    "per_method": {"G-4B-6gen": (60.6, 14.6, 1.1), "G-4B-8gen": (60.9, 14.9, 1.4),
                   "G-8B-4gen": (67.5, 14.6, 8.1), "G-8B-6gen": (64.0, 11.1, 4.6),
                   "S-4B-6gen": (64.6, 18.6, 5.1), "S-8B-6gen": (69.6, 16.7, 10.2),
                   "S-9B-6gen": (68.6, 7.0, 3.0)},
    # (bench, size): (best cell, best, d_base, d_cold, z vs base)
    "best": {("MedQA", "4B"): ("G-8gen", 70.5, 16.5, 4.5, 2.4),
             ("MedQA", "8B"): ("S-6gen", 82.0, 20.0, 3.0, 3.2),
             ("MedQA", "9B"): ("S-6gen", 82.5, 4.0, 4.5, 0.7),
             ("LegalBench", "4B"): ("G-8gen", 81.0, 24.5, 7.0, 5.5),
             ("LegalBench", "8B"): ("G-4gen", 83.5, 20.5, 24.5, 4.8),
             ("LegalBench", "9B"): ("S-6gen", 80.0, 13.5, 0.5, 3.1),
             ("GPQA", "4B"): ("S-6gen", 47.0, 13.0, 13.0, 1.9),
             ("GPQA", "8B"): ("S-6gen", 44.0, 6.0, 3.0, 0.9),
             ("GPQA", "9B"): ("S-6gen", 46.0, 0.0, 1.0, 0.0),
             ("MMLU-Pro", "4B"): ("S-6gen", 70.0, 30.5, 6.0, 6.4),
             ("MMLU-Pro", "8B"): ("S-6gen", 71.0, 22.5, 12.5, 4.7),
             ("MMLU-Pro", "9B"): ("S-6gen", 66.0, 10.5, 6.0, 2.2)},
    # (bench, size): (Specific - Generic, SE of the difference), both 6gen
    "spec_minus_gen": {("MedQA", "4B"): (-3.0, 6.6), ("MedQA", "8B"): (5.0, 5.7),
                       ("LegalBench", "4B"): (1.0, 4.3), ("LegalBench", "8B"): (0.5, 3.9),
                       ("GPQA", "4B"): (7.0, 7.0), ("GPQA", "8B"): (8.0, 6.9),
                       ("MMLU-Pro", "4B"): (11.0, 4.8), ("MMLU-Pro", "8B"): (9.0, 4.7)},
    "cold_below_base": {("MedQA", "9B"), ("LegalBench", "8B"), ("GPQA", "9B")},
    "grpo_below_cold": {("MMLU-Pro", "G-4B-8gen"): -15.0, ("GPQA", "G-8B-6gen"): -5.0,
                        ("MMLU-Pro", "G-4B-6gen"): -5.0, ("MedQA", "G-8B-4gen"): -4.0,
                        ("MedQA", "G-8B-6gen"): -2.0},
    "pearson_r": 0.88,
    "r2": 0.77,
    "scaling_spec_4to8": {"MedQA": 16.0, "LegalBench": 6.0, "GPQA": -3.0, "MMLU-Pro": 1.0},
    "scaling_spec_8to9": {"MedQA": 0.5, "LegalBench": -1.5, "GPQA": 2.0, "MMLU-Pro": -5.0},
    "scaling_base_8to9": {"MedQA": 16.5, "LegalBench": 3.5, "GPQA": 8.0, "MMLU-Pro": 7.0},
    # 100*sqrt(2*0.25/n): exactly 7.07 and 5.00 (the spec's 4.9 rounded SE to 3.5 first)
    "band_pp": {100: 7.1, 200: 5.0},
}

# neutral grey for the "indistinguishable from zero" band (a red tint would read as a warning)
BAND_GREY = (0.45, 0.48, 0.52, 0.12)

MISMATCHES: list[str] = []


def check(label: str, computed: float, expected: float, tol: float = 0.051) -> None:
    ok = abs(computed - expected) <= tol
    tag = "ok      " if ok else "MISMATCH"
    print(f"[{tag}] {label:<42s} computed {computed:8.3f}   spec {expected:6.1f}")
    if not ok:
        MISMATCHES.append(f"{label}: computed {computed:.3f} vs spec {expected}")


def check_str(label: str, computed: str, expected: str) -> None:
    ok = computed == expected
    tag = "ok      " if ok else "MISMATCH"
    print(f"[{tag}] {label:<42s} computed {computed:>8s}   spec {expected:>6s}")
    if not ok:
        MISMATCHES.append(f"{label}: computed {computed} vs spec {expected}")


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def se_pp(acc_pp: float, n: int) -> float:
    """Binomial SE in percentage points from the observed accuracy and n."""
    p = acc_pp / 100.0
    return 100.0 * math.sqrt(p * (1.0 - p) / n)


def load():
    raw = json.loads(DATA.read_text())
    bench: dict[str, dict] = {}
    cells: list[dict] = []
    for b in BENCHES:
        d = raw[b]
        n = int(d["n"])
        bench[b] = {"n": n,
                    "base": {s: 100.0 * v for s, v in d["base"].items()},
                    "cold": {s: 100.0 * v for s, v in d["cold"].items()}}
        for fam in FAMILIES:
            for size, gens in d[fam].items():
                for ngen, acc in gens.items():
                    c = {"bench": b, "family": fam, "size": size, "ngen": ngen, "n": n,
                         "method": f"{fam[0]}-{size}-{ngen}", "cell": f"{fam[0]}-{ngen}",
                         "base": 100.0 * d["base"][size], "cold": 100.0 * d["cold"][size],
                         "grpo": 100.0 * acc}
                    c["d_base"] = c["grpo"] - c["base"]
                    c["d_cold"] = c["grpo"] - c["cold"]
                    c["cold_part"] = c["cold"] - c["base"]
                    cells.append(c)
    assert len(cells) == 28, len(cells)
    assert {c["method"] for c in cells} == set(METHODS)
    return bench, cells


# ----------------------------------------------------------------------------
# derived statistics + checks
# ----------------------------------------------------------------------------
def derive(bench, cells):
    out: dict = {}
    print("\n== means over the 28 GRPO cells ==")
    out["mean_d_base"] = float(np.mean([c["d_base"] for c in cells]))
    out["mean_d_cold"] = float(np.mean([c["d_cold"] for c in cells]))
    check("mean delta vs base", out["mean_d_base"], SPEC["mean_d_base"])
    check("mean delta vs cold", out["mean_d_cold"], SPEC["mean_d_cold"])

    print("\n== decomposition per benchmark (mean over its 7 cells) ==")
    out["decomp"] = {}
    for b in BENCHES:
        cb = [c for c in cells if c["bench"] == b]
        total = float(np.mean([c["d_base"] for c in cb]))
        cold_part = float(np.mean([c["cold_part"] for c in cb]))
        grpo_part = float(np.mean([c["d_cold"] for c in cb]))
        out["decomp"][b] = (total, cold_part, grpo_part)
        et, ec, eg = SPEC["decomp"][b]
        check(f"{b} total", total, et)
        check(f"{b} cold part", cold_part, ec)
        check(f"{b} GRPO part", grpo_part, eg)
        assert abs(total - cold_part - grpo_part) < 1e-9

    print("\n== by manager size ==")
    out["by_size"] = {}
    for s in SIZES:
        cs_ = [c for c in cells if c["size"] == s]
        db = float(np.mean([c["d_base"] for c in cs_]))
        dc = float(np.mean([c["d_cold"] for c in cs_]))
        out["by_size"][s] = (db, dc)
        check(f"{s} mean delta vs base", db, SPEC["by_size"][s][0])
        check(f"{s} mean delta vs cold", dc, SPEC["by_size"][s][1])

    print("\n== per method (mean over 4 benchmarks) ==")
    out["per_method"] = {}
    for m in METHODS:
        cm = [c for c in cells if c["method"] == m]
        assert len(cm) == 4
        avg = float(np.mean([c["grpo"] for c in cm]))
        db = float(np.mean([c["d_base"] for c in cm]))
        dc = float(np.mean([c["d_cold"] for c in cm]))
        out["per_method"][m] = (avg, db, dc)
        check(f"{m} avg score", avg, SPEC["per_method"][m][0])
        check(f"{m} delta vs base", db, SPEC["per_method"][m][1])
        check(f"{m} delta vs cold", dc, SPEC["per_method"][m][2])
    best_method = max(METHODS, key=lambda m: out["per_method"][m][0])
    check_str("best average method", best_method, "S-8B-6gen")

    print("\n== best cell per (benchmark, size) ==")
    out["best"] = {}
    for b in BENCHES:
        n = bench[b]["n"]
        for s in SIZES:
            group = [c for c in cells if c["bench"] == b and c["size"] == s]
            top = max(group, key=lambda c: c["grpo"])
            ties = [c for c in group if c["grpo"] == top["grpo"]]
            assert len(ties) == 1, (b, s, [c["cell"] for c in ties])
            z = top["d_base"] / math.sqrt(se_pp(top["grpo"], n) ** 2 + se_pp(top["base"], n) ** 2)
            out["best"][(b, s)] = {"cell": top["cell"], "method": top["method"], "grpo": top["grpo"],
                                   "base": top["base"], "cold": top["cold"],
                                   "d_base": top["d_base"], "d_cold": top["d_cold"], "z": z}
            e_cell, e_best, e_db, e_dc, e_z = SPEC["best"][(b, s)]
            check_str(f"{b} {s} best cell", top["cell"], e_cell)
            check(f"{b} {s} best score", top["grpo"], e_best)
            check(f"{b} {s} delta base", top["d_base"], e_db)
            check(f"{b} {s} delta cold", top["d_cold"], e_dc)
            check(f"{b} {s} z vs base", z, e_z)

    print("\n== cold start below base ==")
    cold_below = {(b, s) for b in BENCHES for s in SIZES if bench[b]["cold"][s] < bench[b]["base"][s]}
    out["cold_below_base"] = cold_below
    check_str("cold < base cells", ", ".join(sorted(f"{b}-{s}" for b, s in cold_below)),
              ", ".join(sorted(f"{b}-{s}" for b, s in SPEC["cold_below_base"])))

    print("\n== GRPO below its own cold start ==")
    neg = {(c["bench"], c["method"]): c["d_cold"] for c in cells if c["d_cold"] < 0}
    out["grpo_below_cold"] = neg
    check("number of negative GRPO-cold cells", len(neg), len(SPEC["grpo_below_cold"]), tol=0)
    for k, v in SPEC["grpo_below_cold"].items():
        check(f"{k[0]} {k[1]} GRPO-cold", neg.get(k, float("nan")), v)
    assert all(c["family"] == "Generic" for c in cells if c["d_cold"] < 0)

    print("\n== Specific - Generic (same size, both 6gen) ==")
    out["spec_minus_gen"] = {}
    for b in BENCHES:
        n = bench[b]["n"]
        for s in ("4B", "8B"):
            sp = next(c for c in cells if c["bench"] == b and c["size"] == s and c["method"].startswith("S") and c["ngen"] == "6gen")
            ge = next(c for c in cells if c["bench"] == b and c["size"] == s and c["method"].startswith("G") and c["ngen"] == "6gen")
            diff = sp["grpo"] - ge["grpo"]
            se = math.sqrt(se_pp(sp["grpo"], n) ** 2 + se_pp(ge["grpo"], n) ** 2)
            out["spec_minus_gen"][(b, s)] = (diff, se)
            check(f"{b} {s} Specific-Generic", diff, SPEC["spec_minus_gen"][(b, s)][0])
            check(f"{b} {s} SE(diff)", se, SPEC["spec_minus_gen"][(b, s)][1])
    # how many SEs each difference is from zero; the spec says "only MMLU-Pro exceeds ~2 SE"
    out["z_pairs"] = {k: d / se for k, (d, se) in out["spec_minus_gen"].items()}
    for k, z in out["z_pairs"].items():
        print(f"           {k[0]} {k[1]}: difference = {z:.2f} SE")
    strict = [f"{b}-{s}" for (b, s), z in out["z_pairs"].items() if abs(z) > 2]
    print(f"           strictly > 2 SE: {', '.join(strict)}")
    out["emph_bench"] = "MMLU-Pro"
    z_emph = [abs(z) for (b, _), z in out["z_pairs"].items() if b == out["emph_bench"]]
    z_rest = [abs(z) for (b, _), z in out["z_pairs"].items() if b != out["emph_bench"]]
    out["z_emph_min"], out["z_rest_max"] = min(z_emph), max(z_rest)
    print(f"           {out['emph_bench']} min |z| = {out['z_emph_min']:.2f} SE; "
          f"all other pairs max |z| = {out['z_rest_max']:.2f} SE")
    assert out["z_emph_min"] > out["z_rest_max"], "emphasis on MMLU-Pro is not justified by the data"
    n_pos = sum(1 for (d, _) in out["spec_minus_gen"].values() if d >= 0)
    check("pairs with Specific >= Generic", n_pos, 7, tol=0)

    print("\n== scaling (Specific 6gen) ==")
    out["scaling"] = {}
    for b in BENCHES:
        g = {c["size"]: c["grpo"] for c in cells if c["bench"] == b and c["method"].startswith("S")}
        out["scaling"][b] = (g["8B"] - g["4B"], g["9B"] - g["8B"], bench[b]["base"]["9B"] - bench[b]["base"]["8B"])
        check(f"{b} Specific 4B->8B", out["scaling"][b][0], SPEC["scaling_spec_4to8"][b])
        check(f"{b} Specific 8B->9B", out["scaling"][b][1], SPEC["scaling_spec_8to9"][b])
        check(f"{b} base 8B->9B", out["scaling"][b][2], SPEC["scaling_base_8to9"][b])

    print("\n== base vs final (28 cells) ==")
    x = np.array([c["base"] for c in cells])
    y = np.array([c["grpo"] for c in cells])
    r = float(np.corrcoef(x, y)[0, 1])
    slope, intercept = np.polyfit(x, y, 1)
    out["r"], out["slope"], out["intercept"] = r, float(slope), float(intercept)
    check("Pearson r", r, SPEC["pearson_r"], tol=0.005)
    check("R squared", r * r, SPEC["r2"], tol=0.005)
    print(f"           least-squares fit: y = {intercept:.2f} + {slope:.3f} x")

    print("\n== noise band for a paired difference at p = 0.5 ==")
    out["band"] = {}
    for n in (100, 200):
        band = 100.0 * math.sqrt(2.0 * 0.25 / n)
        out["band"][n] = band
        check(f"sqrt(2)*SE, n={n}", band, SPEC["band_pp"][n])
    return out


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
def fig_main_results(bench, best):
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.9), sharey=True)
    w, gap = 0.26, 0.03
    stages = [("base", "Base (direct answer)"), ("cold", "Cold-start SFT"), ("grpo", "Best GRPO cell")]
    for ax, b in zip(axes.flat, BENCHES):
        n = bench[b]["n"]
        cs.ygrid(ax, zero=False)
        for i, s in enumerate(SIZES):
            bc = best[(b, s)]
            vals = {"base": bc["base"], "cold": bc["cold"], "grpo": bc["grpo"]}
            for j, (key, _) in enumerate(stages):
                x = i + (j - 1) * (w + gap)
                v = vals[key]
                se = se_pp(v, n)
                ax.bar(x, v, width=w, color=cs.STAGE[key], linewidth=0, zorder=2)
                ax.errorbar(x, v, yerr=se, fmt="none", ecolor=cs.INK2, elinewidth=0.9,
                            capsize=2.5, capthick=0.9, zorder=3)
                ax.annotate(f"{v:.1f}", (x, v + se), xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9, color=cs.INK)
                if key == "grpo":
                    ax.annotate(bc["cell"], (x, v + se), xytext=(0, 14), textcoords="offset points",
                                ha="center", va="bottom", fontsize=9, color=cs.MUTED)
        ax.set_title(f"{b}  (n = {n})")
        ax.set_xticks(range(len(SIZES)))
        ax.set_xticklabels([SIZE_NAME[s] for s in SIZES])
        ax.set_xlim(-0.6, len(SIZES) - 0.4)
        ax.set_ylim(0, 114)
        ax.set_yticks(range(0, 101, 20))
        ax.spines["left"].set_bounds(0, 100)
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy (%)")
    handles = [Patch(facecolor=cs.STAGE[k], label=lab) for k, lab in stages]
    fig.tight_layout(rect=(0, 0.0, 1, 0.95), h_pad=1.6)
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0),
               frameon=False, columnspacing=2.0)
    fig.text(0.0, -0.012,
             "Error bars: ±1 binomial SE from the observed accuracy and n.  "
             "The label above each GRPO bar names the best cell: G = generic advisors, "
             "S = domain-specific advisors, kgen = GRPO group size.",
             fontsize=9, color=cs.MUTED, ha="left", va="top", wrap=True)
    cs.save(fig, "fig_main_results")


def fig_decomposition(bench, decomp):
    fig, ax = plt.subplots(figsize=(7, 3.1))
    cs.xgrid(ax, zero=True)
    ys = np.arange(len(BENCHES))[::-1]
    h = 0.52
    for y, b in zip(ys, BENCHES):
        total, cold_part, grpo_part = decomp[b]
        assert cold_part >= 0 and grpo_part >= 0, (b, cold_part, grpo_part)
        ax.barh(y, cold_part, height=h, color=cs.STAGE["cold"], linewidth=0, zorder=2)
        ax.barh(y, grpo_part, left=cold_part, height=h, color=cs.STAGE["grpo"], linewidth=0, zorder=2)
        # 2 px surface gap between the two segments
        ax.plot([cold_part, cold_part], [y - h / 2, y + h / 2], color=cs.SURFACE, lw=1.6, zorder=3)
        ax.annotate(f"+{total:.1f} pp", (total, y), xytext=(7, 1), textcoords="offset points",
                    ha="left", va="bottom", fontsize=10, fontweight="bold", color=cs.INK)
        ax.annotate(f"cold start +{cold_part:.1f}  ·  GRPO +{grpo_part:.1f}", (total, y),
                    xytext=(7, -2), textcoords="offset points",
                    ha="left", va="top", fontsize=9, color=cs.INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{b}  (n = {bench[b]['n']})" for b in BENCHES])
    ax.set_ylim(-0.6, len(BENCHES) - 0.4)
    ax.set_xlim(0, 30)
    ax.set_xticks(range(0, 21, 5))
    ax.spines["bottom"].set_bounds(0, 20)
    ax.set_xlabel("Mean accuracy gain over base (pp), averaged over the benchmark's 7 GRPO cells")
    handles = [Patch(facecolor=cs.STAGE["cold"], label="Cold-start SFT  (cold − base)"),
               Patch(facecolor=cs.STAGE["grpo"], label="GRPO  (final − cold)")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2,
              frameon=False, columnspacing=2.0)
    cs.save(fig, "fig_decomposition")


def fig_grpo_vs_cold(bench, cells, band):
    fig, ax = plt.subplots(figsize=(7, 5.8))
    row = {b: i for i, b in enumerate(reversed(BENCHES))}       # MedQA on top
    off = dict(zip(METHODS, np.linspace(0.36, -0.36, len(METHODS))))
    for b in BENCHES:
        n = bench[b]["n"]
        y = row[b]
        ax.add_patch(Rectangle((-band[n], y - 0.47), 2 * band[n], 0.94,
                               facecolor=BAND_GREY, edgecolor="none", zorder=0))
        ax.text(band[n] + 0.8, y + 0.44, f"noise band ±{band[n]:.1f} pp",
                ha="left", va="top", fontsize=9, color=cs.MUTED, zorder=1)
    cs.xgrid(ax, zero=True)
    for c in cells:
        y = row[c["bench"]] + off[c["method"]]
        ax.plot(c["d_cold"], y, marker=cs.SIZE_MARKER[c["size"]], color=cs.FAMILY[c["family"]],
                ms=8, mec=cs.SURFACE, mew=1.2, ls="none", zorder=3)
        if c["d_cold"] < 0:
            ax.annotate(f"{c['method']}  {c['d_cold']:+.1f}", (c["d_cold"], y), xytext=(-8, 0),
                        textcoords="offset points", ha="right", va="center", fontsize=9, color=cs.INK)
    ax.set_yticks([row[b] for b in BENCHES])
    ax.set_yticklabels([f"{b}\nn = {bench[b]['n']}" for b in BENCHES])
    ax.set_ylim(-0.6, len(BENCHES) - 0.4)
    ax.set_xlim(-28, 30)
    ax.set_xticks(range(-20, 31, 10))
    ax.spines["bottom"].set_bounds(-20, 30)
    ax.set_xlabel("GRPO accuracy − cold-start accuracy (pp), one dot per GRPO cell")
    handles = [
        Line2D([], [], marker="o", color=cs.FAMILY["Generic"], mec=cs.SURFACE, mew=1.2, ms=8, ls="none", label="Generic advisors"),
        Line2D([], [], marker="o", color=cs.FAMILY["Specific"], mec=cs.SURFACE, mew=1.2, ms=8, ls="none", label="Specific advisors"),
        Line2D([], [], marker=cs.SIZE_MARKER["4B"], color=cs.MUTED, mec=cs.SURFACE, mew=1.2, ms=8, ls="none", label="4B"),
        Line2D([], [], marker=cs.SIZE_MARKER["8B"], color=cs.MUTED, mec=cs.SURFACE, mew=1.2, ms=8, ls="none", label="8B"),
        Line2D([], [], marker=cs.SIZE_MARKER["9B"], color=cs.MUTED, mec=cs.SURFACE, mew=1.2, ms=8, ls="none", label="9B"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=5,
              frameon=False, handletextpad=0.4, columnspacing=1.4)
    fig.text(0.0, -0.01,
             "Band: ±1 SE of a paired difference between two scores at p = 0.5\n"
             f"(√2·SE: {band[100]:.1f} pp for n = 100, {band[200]:.1f} pp for n = 200). "
             "Cells below their own cold start are labelled.",
             fontsize=9, color=cs.MUTED, ha="left", va="top", linespacing=1.4)
    fig.tight_layout()
    cs.save(fig, "fig_grpo_vs_cold")


def fig_specific_minus_generic(bench, pairs, z, emph_bench, z_rest_max):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    cs.ygrid(ax, zero=True)
    dx = {"4B": -0.18, "8B": 0.18}
    for i, b in enumerate(BENCHES):
        for s in ("4B", "8B"):
            d, se = pairs[(b, s)]
            x = i + dx[s]
            # emphasis by ink weight, not by the orange that means "Specific" elsewhere
            col = cs.INK if b == emph_bench else cs.MUTED
            ax.errorbar(x, d, yerr=se, fmt="none", ecolor=col, elinewidth=1.2,
                        capsize=3, capthick=1.2, zorder=2)
            ax.plot(x, d, marker=cs.SIZE_MARKER[s], color=col, ms=8, mec=cs.SURFACE, mew=1.2,
                    ls="none", zorder=3)
            ax.annotate(f"{d:+.1f}", (x, d), xytext=(8, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=9, color=cs.INK)
    # annotate the emphasised benchmark with the actual SE multiples; right-anchored
    # just inside the axes so the text never overhangs the plot area
    x_max = len(BENCHES) - 0.4
    top = max(d + se for (bb, s), (d, se) in pairs.items() if bb == emph_bench)
    z_txt = " and ".join(f"{z[(emph_bench, s)]:.1f} SE" for s in ("4B", "8B"))
    ax.annotate(f"{z_txt} from zero", (x_max - 0.05, top), xytext=(0, 8), textcoords="offset points",
                ha="right", va="bottom", fontsize=9, color=cs.INK)
    ax.set_xticks(range(len(BENCHES)))
    ax.set_xticklabels([f"{b}\n(n = {bench[b]['n']})" for b in BENCHES])
    ax.set_xlim(-0.6, x_max)
    ax.set_ylim(-14, 22)
    ax.set_yticks(range(-10, 21, 5))
    ax.spines["left"].set_bounds(-10, 20)
    ax.set_ylabel("Specific − Generic accuracy (pp)\nsame manager size, both 6gen")
    handles = [
        Line2D([], [], marker=cs.SIZE_MARKER["4B"], color=cs.INK2, mec=cs.SURFACE, mew=1.2, ms=8, ls="none", label="4B manager"),
        Line2D([], [], marker=cs.SIZE_MARKER["8B"], color=cs.INK2, mec=cs.SURFACE, mew=1.2, ms=8, ls="none", label="8B manager"),
        Patch(facecolor=cs.INK, label=f"{emph_bench}: ≈ 2 SE from zero"),
        Patch(facecolor=cs.MUTED, label=f"other pairs: ≤ {z_rest_max:.1f} SE"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=4,
              frameon=False, handletextpad=0.5, columnspacing=1.4)
    fig.text(0.0, -0.01, "Error bars: ±1 SE of the difference (binomial SE of each score, combined in quadrature).",
             fontsize=9, color=cs.MUTED, ha="left", va="top")
    fig.tight_layout()
    cs.save(fig, "fig_specific_minus_generic")


def drawn_positions(cells, nudge=0.4, min_gap=1.0):
    """Drawing position per cell.  Cells that share (benchmark, size), and so
    share x, are pushed ±nudge pp apart in x when they lie within min_gap pp
    of each other in y; otherwise one marker would hide the other.  Only the
    drawn position moves; r and the fit use the real values."""
    pos = {id(c): (c["base"], c["grpo"]) for c in cells}
    groups: dict[tuple, list] = {}
    for c in cells:
        groups.setdefault((c["bench"], c["size"]), []).append(c)
    for g in groups.values():
        g.sort(key=lambda c: c["grpo"])
        for a, b in zip(g, g[1:]):
            if abs(a["grpo"] - b["grpo"]) <= min_gap:
                pos[id(a)] = (a["base"] - nudge, a["grpo"])
                pos[id(b)] = (b["base"] + nudge, b["grpo"])
                print(f"           nudged apart in x: {a['bench']} {a['method']} ({a['grpo']:.1f}) "
                      f"and {b['method']} ({b['grpo']:.1f})")
    return pos


# direct benchmark labels next to each cluster: (x, y, ha) in data units
BENCH_LABEL_POS = {
    "GPQA": (40.5, 38.5, "left"),
    "MMLU-Pro": (38.6, 54.0, "right"),
    "MedQA": (63.8, 75.2, "left"),
    "LegalBench": (55.3, 81.0, "right"),
}


def fig_base_vs_final(cells, r, slope, intercept):
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(axis="both", color=cs.GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    lo, hi = 30, 90
    ax.plot([lo, hi], [lo, hi], ls=(0, (5, 4)), color=cs.MUTED, lw=1.2, zorder=1)
    xs = np.array([lo, hi])
    ax.plot(xs, intercept + slope * xs, color=cs.INK2, lw=1.6, zorder=1)
    pos = drawn_positions(cells)
    for c in cells:
        x, y = pos[id(c)]
        ax.plot(x, y, marker=cs.SIZE_MARKER[c["size"]], color=cs.BENCH[c["bench"]],
                ms=8.5, mec=cs.SURFACE, mew=1.2, ls="none", zorder=3)
    # benchmark identity: colour plus a direct label beside each cluster
    for b, (x, y, ha) in BENCH_LABEL_POS.items():
        ax.text(x, y, b, ha=ha, va="center", fontsize=10, color=cs.BENCH[b], zorder=4)
    # direct label on the reference line, placed on its empty middle stretch
    ax.text(60.5, 59.5 + 1.0, "y = x  (no change)", rotation=45, rotation_mode="anchor",
            ha="center", va="bottom", fontsize=9, color=cs.MUTED, zorder=2)
    ax.text(lo + 1.5, hi - 2.0,
            f"Pearson r = {r:.2f}   (R² = {r * r:.2f}),  28 GRPO cells\n"
            f"least-squares fit (solid line):  y = {intercept:.1f} + {slope:.2f}·x",
            ha="left", va="top", fontsize=10, color=cs.INK, linespacing=1.5, zorder=4)
    # typical ±1 SE reference cross (p = 0.5, n = 100)
    cx, cy, se100 = 80, 40, se_pp(50, 100)
    ax.plot([cx - se100, cx + se100], [cy, cy], color=cs.INK2, lw=1, zorder=2)
    ax.plot([cx, cx], [cy - se100, cy + se100], color=cs.INK2, lw=1, zorder=2)
    ax.text(cx, cy - se100 - 1.2, f"±1 SE at p = 0.5\n{se100:.1f} pp (n = 100), {se_pp(50, 200):.1f} pp (n = 200)",
            ha="center", va="top", fontsize=9, color=cs.INK2, linespacing=1.4)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xticks(range(lo, hi + 1, 10))
    ax.set_yticks(range(lo, hi + 1, 10))
    ax.set_xlabel("Base accuracy (%): manager answers directly, no advisors")
    ax.set_ylabel("Accuracy after cold start + GRPO (%)")
    handles = [Line2D([], [], marker="o", color=cs.BENCH[b], mec=cs.SURFACE, mew=1.2, ms=8.5, ls="none", label=b)
               for b in BENCHES]
    handles += [Line2D([], [], marker=cs.SIZE_MARKER[s], color=cs.MUTED, mec=cs.SURFACE, mew=1.2, ms=8.5,
                       ls="none", label=f"{s} manager") for s in SIZES]
    handles += [Line2D([], [], color=cs.MUTED, lw=1.2, ls=(0, (5, 4)), label="y = x"),
                Line2D([], [], color=cs.INK2, lw=1.6, label="least-squares fit")]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.03, 0.5), frameon=False,
              handletextpad=0.5, labelspacing=0.7)
    fig.subplots_adjust(left=0.07, right=0.80, bottom=0.085, top=0.985)
    cs.save(fig, "fig_base_vs_final")


# ----------------------------------------------------------------------------
def main() -> None:
    cs.apply_style()
    bench, cells = load()
    d = derive(bench, cells)

    fig_main_results(bench, d["best"])
    fig_decomposition(bench, d["decomp"])
    fig_grpo_vs_cold(bench, cells, d["band"])
    fig_specific_minus_generic(bench, d["spec_minus_gen"], d["z_pairs"], d["emph_bench"], d["z_rest_max"])
    fig_base_vs_final(cells, d["r"], d["slope"], d["intercept"])

    print()
    if MISMATCHES:
        print(f"{len(MISMATCHES)} check(s) disagree with the spec (computed values were drawn):")
        for m in MISMATCHES:
            print("  -", m)
    else:
        print("all checks agree with the spec")


if __name__ == "__main__":
    main()
