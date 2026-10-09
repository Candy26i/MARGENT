#!/usr/bin/env python3
"""Paper figures for the MARGENT project page (Tables 1-5, 8-10 of the submission).

Reads ``static/data/paper_tables.json`` (the paper's tables, transcribed) and,
for the released 8B traces, ``static/data/medqa_marginal_v1/eval_dev200.jsonl``
and ``eval_test200.jsonl`` (the records of the MARGENT ``legacy`` branch, tag
``v0.1-full``).  Nothing is typed in by hand
except the numbers the paper states in prose, which are used as checks: every
derived quantity (Gain@1, macro means, gains, call reductions, the 20:1 ratio,
the released call-count distribution) is recomputed here and asserted against
the table / the paper text before anything is drawn.

  fig_oracle_depth           Table 1   commit / best one call / best measured, Gain@1
  fig_net_marginal_value     Table 2   net marginal value of one forced call per sub-agent
  fig_disjoint_eval          Table 3   candidate vs MARGENT on collection-disjoint evaluations
  fig_accuracy_vs_calls      Tables 4+5  accuracy vs calls, four small multiples
  fig_correction_corruption  Table 8   correction vs corruption fractions
  fig_rho_sweep              Table 9   commit-to-rescue ratio arrows in the (calls, accuracy) plane
  fig_grpo_collapse          Table 10 + released 8B traces

Run:  /opt/anaconda3/bin/python3 static/charts/make_paper_figures.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import offset_copy

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "paper_tables.json"
sys.path.insert(0, str(HERE))
import chart_style as cs  # noqa: E402

# ----------------------------------------------------------------------------
# colour roles for this family of figures (chart_style slots; validated with the
# dataviz validator, light mode, surface #ffffff, see notes in the run log).
# BLUE is reserved for the learned MARGENT policy in every figure.
# ----------------------------------------------------------------------------
C_CANDIDATE = cs.NEUTRAL      # candidate solution / direct commit (reference grey)
C_MARGENT = cs.BLUE           # learned policy / selective SFT (every MARGENT point, whatever rho)
# measured oracle by depth (Table 1): a neutral ordinal grey ramp, light to dark, so the
# upper bound is never read as a policy result; A0 keeps the candidate/commit grey
C_ORACLE = {"A0": cs.NEUTRAL, "A1": cs.MUTED, "AD": cs.INK2}
C_VERIFIER_FORCED = cs.ORANGE  # always-Verifier baseline
C_FORCE_ALL = cs.VIOLET        # force-all baseline
C_AGENT = {"Extractor": cs.YELLOW, "Reasoner": cs.VIOLET, "Verifier": cs.AQUA}
# correction vs corruption is polarity, drawn with the palette's diverging blue/red pair
# (validator: CVD dE 21.6, all checks pass); the status good/critical pair fails the CVD
# check (dE 4.1 deutan, below the 6 floor), so it is not used
C_CORRECTION = cs.BLUE
C_CORRUPTION = cs.RED
C_GRPO = cs.MUTED             # outcome-only GRPO continuations (panel a reference grey)
C_SPLIT = {"dev": cs.INK2, "test": cs.NEUTRAL}   # released trace splits (panel b), distinct lightness
IN_BAR_INK = {"dev": cs.SURFACE, "test": cs.INK}  # text inside a bar, by the fill's luminance

ARROW = r"$\rightarrow$"
RHO = r"$\rho$"

# ----------------------------------------------------------------------------
# numbers the paper states in prose (checked against the tables, never drawn
# unless the check passes)
# ----------------------------------------------------------------------------
PAPER_TEXT = {
    "gain_at_1_macro": 91.7,          # abstract, Sec. 4.2
    "second_call_adds": (0.5, 4.5),   # Sec. 4.2
    "disjoint_gain_macro": 17.1,      # abstract, Sec. 4.4
    "disjoint_calls_macro": 0.53,     # abstract, Sec. 4.4
    "t4_calls_from_to": (1.00, 0.565),  # Sec. 4.4
    "t4_points_given_up": 1.6,        # Sec. 4.4
    "t5_macro": (73.2, 1.03, 73.6, 3.0, 0.4),  # Sec. 4.4
    "t5_mmlu_match": 71.5,            # abstract
    "t5_aqua": (81.1, 75.2),          # abstract
    "t8_aqua_ratio": 20.0,            # Sec. 4.4 "a twenty-to-one ratio"
    "t8_aqua_pct": (48.0, 2.4),       # Sec. 4.4
    "t9_aqua_removed_pct": 37.4,      # Sec. 4.4
    "t9_aqua_points_lost": 12.6,      # Sec. 4.4
    "t10_calls": {"MedQA, 8B manager": 2.98, "MedQA, 9B manager": 3.00},  # Table 10, Sec. 4.5
    "seeds": (80.5, 1.0, 0.517, 0.043),  # Appendix D
    "released_calls": {"dev": {3: 200}, "test": {1: 1, 2: 1, 3: 198}},  # content spec, released traces
    "released_mean_calls": {"dev": 3.000, "test": 2.985},
}

MISMATCHES: list[str] = []


def check(label: str, computed, expected, tol: float = 0.051) -> None:
    if isinstance(expected, dict):
        ok = computed == expected
    else:
        ok = abs(float(computed) - float(expected)) <= tol
    shown = f"{computed:.4f}" if isinstance(computed, float) else str(computed)
    print(f"  {label:<52} computed {shown:<20} expected {expected!s:<20} {'ok' if ok else 'MISMATCH'}")
    if not ok:
        MISMATCHES.append(f"{label}: computed {computed} != expected {expected}")
    assert ok, MISMATCHES[-1] if not ok else ""


def rows(table: dict, keys: list[str]) -> dict[str, list]:
    return {k: table[k] for k in keys}


# ----------------------------------------------------------------------------
# loading and verification
# ----------------------------------------------------------------------------
def load() -> dict:
    with open(DATA, encoding="utf-8") as f:
        return json.load(f)


def verify(t: dict) -> None:
    print("Table 1: oracle by depth")
    t1 = t["table1_oracle_by_depth"]
    tasks1 = ["MedQA", "AQuA-RAT", "MMLU-Pro", "GPQA"]
    gains, second = [], []
    for task in tasks1:
        n, a0, a1, ad, g = t1[task]
        g_c = 100 * (a1 - a0) / (ad - a0)
        check(f"{task} Gain@1 = (A1-A0)/(AD-A0)", g_c, g)
        gains.append(g)
        second.append(ad - a1)
    for j, col in enumerate(["commit_A0", "best_1_call_A1", "best_measured_AD"], start=1):
        check(f"macro mean {col}", float(np.mean([t1[k][j] for k in tasks1])), t1["Macro mean"][j], tol=0.0051)
    check("macro mean Gain@1 (mean of task values)", float(np.mean(gains)), t1["Macro mean"][4])
    check("macro Gain@1 vs abstract", t1["Macro mean"][4], PAPER_TEXT["gain_at_1_macro"])
    check("second call adds, min", min(second), PAPER_TEXT["second_call_adds"][0])
    check("second call adds, max", max(second), PAPER_TEXT["second_call_adds"][1])

    print("Table 2: net marginal value")
    t2 = t["table2_net_marginal_value_pp"]
    for task in ["MedQA", "MMLU-Pro", "GPQA", "AQuA-RAT"]:
        e, r, v = t2[task]
        assert v == max(e, r, v), f"{task}: Verifier is not the largest"
        assert e <= 2.0, f"{task}: Extractor not close to zero"
    print("  Verifier largest on all four tasks; Extractor <= +2.0 everywhere  ok")

    print("Table 3: collection-disjoint evaluation")
    t3 = t["table3_collection_disjoint_eval"]
    tasks3 = ["MedQA", "AQuA-RAT", "MMLU-Pro", "GPQA"]
    for task in tasks3:
        n, cand, marg, gain, calls = t3[task]
        check(f"{task} gain = MARGENT - candidate", marg - cand, gain)
    for j, col in enumerate(["candidate", "margent", "gain_pp"], start=1):
        check(f"macro mean {col}", float(np.mean([t3[k][j] for k in tasks3])), t3["Macro mean"][j])
    check("macro mean calls", float(np.mean([t3[k][4] for k in tasks3])), t3["Macro mean"][4], tol=0.00051)
    check("macro gain vs abstract", t3["Macro mean"][3], PAPER_TEXT["disjoint_gain_macro"])
    check("macro calls vs abstract (0.53)", t3["Macro mean"][4], PAPER_TEXT["disjoint_calls_macro"], tol=0.0051)

    print("Table 4: low-call MARGENT vs always Verifier")
    t4 = t["table4_low_call_vs_always_verifier"]
    tasks4 = ["MedQA", "MMLU-Pro", "GPQA", "AQuA-RAT"]
    for task in tasks4:
        ma, mc, va, vc, fewer = t4[task]
        check(f"{task} fewer calls % = 100(1 - calls)", 100 * (vc - mc) / vc, fewer)
    for j, col in enumerate(["margent_acc", "margent_calls", "verifier_acc", "verifier_calls", "fewer_calls_pct"]):
        tol = 0.00051 if col == "margent_calls" else 0.051
        check(f"macro mean {col}", float(np.mean([t4[k][j] for k in tasks4])), t4["Macro mean"][j], tol=tol)
    check("calls 1.00 -> 0.565 (text)", t4["Macro mean"][1], PAPER_TEXT["t4_calls_from_to"][1])
    check("points given up vs always Verifier (text 1.6)",
          float(np.mean([t4[k][2] - t4[k][0] for k in tasks4])), PAPER_TEXT["t4_points_given_up"])

    print("Table 5: selected MARGENT vs force all")
    t5 = t["table5_selected_vs_force_all"]
    for task in tasks4:
        ma, mc, fa, fc, d = t5[task]
        check(f"{task} delta acc = MARGENT - force all", ma - fa, d)
    for j, col in enumerate(["margent_acc", "margent_calls", "force_all_acc", "force_all_calls", "delta_acc"]):
        tol = 0.00051 if col == "margent_calls" else 0.051
        check(f"macro mean {col}", float(np.mean([t5[k][j] for k in tasks4])), t5["Macro mean"][j], tol=tol)
    ma, mc, fa, fc, d = PAPER_TEXT["t5_macro"]
    check("macro acc vs text", t5["Macro mean"][0], ma)
    check("macro calls vs text (1.03)", t5["Macro mean"][1], mc, tol=0.0051)
    check("macro force-all acc vs text", t5["Macro mean"][2], fa)
    check("macro difference vs text (0.4)", abs(t5["Macro mean"][4]), d)
    check("MMLU-Pro match vs abstract", t5["MMLU-Pro"][0], PAPER_TEXT["t5_mmlu_match"])
    check("MMLU-Pro force all equals MARGENT", t5["MMLU-Pro"][2], t5["MMLU-Pro"][0], tol=1e-9)
    check("AQuA-RAT MARGENT vs abstract", t5["AQuA-RAT"][0], PAPER_TEXT["t5_aqua"][0])
    check("AQuA-RAT force all vs abstract", t5["AQuA-RAT"][2], PAPER_TEXT["t5_aqua"][1])

    print("Table 8: correction / corruption")
    t8 = t["table8_outcome_decomposition"]
    calls, corr, corrupt = t8["AQuA-RAT"]
    check("AQuA-RAT correction:corruption ratio (text 20:1)", corr / corrupt, PAPER_TEXT["t8_aqua_ratio"])
    check("AQuA-RAT correction % (text 48.0)", 100 * corr, PAPER_TEXT["t8_aqua_pct"][0])
    check("AQuA-RAT corruption % (text 2.4)", 100 * corrupt, PAPER_TEXT["t8_aqua_pct"][1])
    for task in tasks4:
        assert t8[task][1] > t8[task][2], f"{task}: net value not positive"
    print("  correction > corruption on every task (positive net value)  ok")

    print("Table 9: commit-to-rescue comparisons")
    t9 = {k: v for k, v in t["table9_rho_comparisons"].items() if not k.startswith("_")}
    for name, (rf, rt, af, at_, cf, ct, gf, gt) in t9.items():
        assert ct < cf, f"{name}: calls did not fall"
        assert gt > gf, f"{name}: call gap did not rise"
    print("  calls fall and call gap rises in every pair  ok")
    rf, rt, af, at_, cf, ct, gf, gt = t9["AQuA-RAT depth 3"]
    check("AQuA-RAT calls removed % (text 37.4)", 100 * (cf - ct) / cf, PAPER_TEXT["t9_aqua_removed_pct"])
    check("AQuA-RAT points lost (text 12.6)", af - at_, PAPER_TEXT["t9_aqua_points_lost"])
    accs = [(t9[k][2], t9[k][3]) for k in t9]
    assert any(b > a for a, b in accs) and any(b < a for a, b in accs), "accuracy change is monotone"
    print("  accuracy rises in some pairs and falls in others (not monotone)  ok")

    print("Table 10 and released 8B traces")
    t10 = t["table10_grpo_continuations"]
    for run, calls in PAPER_TEXT["t10_calls"].items():
        check(f"{run} calls", t10[run]["calls"], calls, tol=1e-9)
    s = t["medqa_three_seed_sft"]
    for key, val in zip(("accuracy_mean", "accuracy_sd", "calls_mean", "calls_sd"), PAPER_TEXT["seeds"]):
        check(f"three-seed {key}", s[key], val, tol=1e-9)
    check("Table 4 MedQA row = three-seed mean acc", t4["MedQA"][0], s["accuracy_mean"], tol=1e-9)
    check("Table 4 MedQA row = three-seed mean calls", t4["MedQA"][1], s["calls_mean"], tol=1e-9)
    lo, hi = s["calls_mean"] - s["calls_sd"], s["calls_mean"] + s["calls_sd"]
    macro_calls = t3["Macro mean"][4]
    assert lo <= macro_calls <= hi, f"Table 3 macro calls {macro_calls} outside the three-seed band"
    print(f"  Table 3 macro calls {macro_calls:.3f} lie inside the three-seed band [{lo:.3f}, {hi:.3f}]  ok")


# ----------------------------------------------------------------------------
# drawing helpers
# ----------------------------------------------------------------------------
def title_and_subtitle(ax, title: str, subtitle: str, pad: float = 24, y: float = 1.03) -> None:
    ax.set_title(title, pad=pad)
    ax.text(0, y, subtitle, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9.5, color=cs.INK2, linespacing=1.4)


def cap_label(ax, x, y, text, dy=1.2, fontsize=9, **kw):
    ax.text(x, y + dy, text, ha="center", va="bottom", fontsize=fontsize, color=cs.INK, zorder=4, **kw)


# ----------------------------------------------------------------------------
# Figure: oracle by depth (Table 1)
# ----------------------------------------------------------------------------
def fig_oracle_depth(t: dict) -> None:
    t1 = t["table1_oracle_by_depth"]
    tasks = ["MedQA", "AQuA-RAT", "MMLU-Pro", "GPQA"]
    series = [
        (1, r"Commit, $A_0$ (no delegation)", C_ORACLE["A0"]),
        (2, r"Best one call, $A_1$", C_ORACLE["A1"]),
        (3, r"Best measured, $A_D$ (depth $D$)", C_ORACLE["AD"]),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.7))
    x = np.arange(len(tasks))
    step, w = 0.27, 0.25
    cs.ygrid(ax, zero=False)
    for j, (col, label, colour) in enumerate(series):
        xs = x + (j - 1) * step
        vals = [t1[k][col] for k in tasks]
        ax.bar(xs, vals, width=w, color=colour, zorder=2, label=label)
        for xi, v in zip(xs, vals):
            cap_label(ax, xi, v, f"{v:.2f}")
    for xi, task in zip(x, tasks):
        g = t1[task][4]
        ax.text(xi, 1.015, f"Gain@1 {g:.1f}%", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=10, color=cs.INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}\nn = {t1[k][0]}" for k in tasks], linespacing=1.4)
    ax.set_xlim(-0.6, len(tasks) - 0.4)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Measured oracle accuracy (%)")
    ax.tick_params(axis="x", length=0)
    macro = t1["Macro mean"]
    title_and_subtitle(
        ax, "Most measured value is available after one delegation",
        f"Table 1, one shared collection per task. Gain@1 = $(A_1-A_0)\\,/\\,(A_D-A_0)$; "
        f"macro mean {macro[4]:.1f}%. The second call adds 0.5 to 4.5 points.",
        pad=44, y=1.10)
    # legend below the axis: a titled legend inside the plot would reach the GPQA labels
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.20), borderaxespad=0, ncol=3,
              handlelength=1.0, handleheight=0.8, columnspacing=1.6,
              title="Measured oracle (not a policy)", title_fontsize=9.5, alignment="left")
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.27, top=0.82)
    cs.save(fig, "fig_oracle_depth")


# ----------------------------------------------------------------------------
# Figure: net marginal value per sub-agent (Table 2)
# ----------------------------------------------------------------------------
def fig_net_marginal_value(t: dict) -> None:
    t2 = t["table2_net_marginal_value_pp"]
    tasks = ["MedQA", "MMLU-Pro", "GPQA", "AQuA-RAT"]
    agents = ["Extractor", "Reasoner", "Verifier"]
    fig, ax = plt.subplots(figsize=(7, 4.9))
    gap, h = 3.9, 0.82
    y_group = {task: -(g * gap) for g, task in enumerate(tasks)}
    ticks, tick_labels = [], []
    cs.xgrid(ax, zero=True)
    for task in tasks:
        for j, agent in enumerate(agents):
            y = y_group[task] + (1 - j) * 1.0
            v = t2[task][j]
            ax.barh(y, v, height=h, color=C_AGENT[agent], zorder=2)
            bold = agent == "Verifier"
            if v >= 0:
                ax.text(v + 0.5, y, f"{v:+.1f}", ha="left", va="center", fontsize=9.5,
                        color=cs.INK, fontweight="bold" if bold else "normal", zorder=4)
            else:
                ax.text(v - 0.5, y, f"{v:+.1f}".replace("-", "\u2212"), ha="right", va="center",
                        fontsize=9.5, color=cs.INK, zorder=4)
            ticks.append(y)
            tick_labels.append(agent)
        ax.text(-0.16, y_group[task], task, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=10.5, fontweight="bold", color=cs.INK)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=9.5)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_ylim(y_group[tasks[-1]] - 1.75, 1.75)
    ax.set_xlim(-9, 35)
    ax.set_xticks(range(-5, 36, 5))
    ax.set_xlabel(r"Net marginal value of one forced call, $100\,\bar{\Delta}_a$ (percentage points)")
    ax.spines["left"].set_visible(False)
    title_and_subtitle(
        ax, "The value of a call depends on which sub-agent is called",
        "Table 2. Positive: more rescue than corruption mass; negative: more corruption.\n"
        "The Verifier has the largest net value on all four tasks; the Extractor is close to zero or negative on all four.",
        pad=36, y=1.025)
    handles = [Patch(color=C_AGENT[a], label=a) for a in agents]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.3,
              handlelength=1.0, handleheight=0.8, labelspacing=0.5, title="Sub-agent (one forced call)",
              title_fontsize=9.5, alignment="left")
    ax.text(0.0, -0.21,
            "One-step records: 12 MedQA and 9 MMLU-Pro questions are rescued by the Reasoner but not the Verifier.",
            ha="left", va="top", fontsize=9, color=cs.MUTED, transform=ax.transAxes)
    fig.subplots_adjust(left=0.22, right=0.985, bottom=0.21, top=0.83)
    cs.save(fig, "fig_net_marginal_value")


# ----------------------------------------------------------------------------
# Figure: collection-disjoint evaluation (Table 3)
# ----------------------------------------------------------------------------
def fig_disjoint_eval(t: dict) -> None:
    t3 = t["table3_collection_disjoint_eval"]
    groups = ["MedQA", "AQuA-RAT", "MMLU-Pro", "GPQA", "Macro mean"]
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.6, 4.4), gridspec_kw={"width_ratios": [2.6, 1], "wspace": 0.42})

    # (a) paired bars candidate vs MARGENT, gain above each pair
    x = np.arange(len(groups))
    w, off = 0.36, 0.19
    cs.ygrid(ax_a, zero=False)
    cand = [t3[g][1] for g in groups]
    marg = [t3[g][2] for g in groups]
    ax_a.bar(x - off, cand, width=w, color=C_CANDIDATE, zorder=2, label="Candidate (before the first delegation)")
    ax_a.bar(x + off, marg, width=w, color=C_MARGENT, zorder=2, label="MARGENT (final output)")
    for xi, c, m, g in zip(x, cand, marg, groups):
        cap_label(ax_a, xi - off, c, f"{c:.1f}")
        cap_label(ax_a, xi + off, m, f"{m:.1f}")
        ax_a.text(xi, max(c, m) + 7.5, f"+{t3[g][3]:.1f}", ha="center", va="bottom",
                  fontsize=10, color=cs.INK, fontweight="bold", zorder=4)
    labels = [f"{g}\nn = {t3[g][0]}" if t3[g][0] else g for g in groups]
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels, linespacing=1.4)
    ax_a.set_xlim(-0.6, len(groups) - 0.4)
    ax_a.set_ylim(0, 100)
    ax_a.set_yticks(range(0, 101, 20))
    ax_a.set_ylabel("Exact-match accuracy (%)")
    ax_a.tick_params(axis="x", length=0)
    title_and_subtitle(ax_a, "(a) Accuracy, gain in pp above each pair",
                       "Table 3, evaluations disjoint from collection")
    ax_a.legend(loc="upper left", bbox_to_anchor=(0.0, -0.17), borderaxespad=0,
                handlelength=1.0, handleheight=0.8, labelspacing=0.4)

    # (b) calls per example, horizontal, same order top to bottom
    ys = np.arange(len(groups))[::-1]
    calls = [t3[g][4] for g in groups]
    cs.xgrid(ax_b, zero=False)
    ax_b.barh(ys, calls, height=0.55, color=C_MARGENT, zorder=2)
    for y, c in zip(ys, calls):
        ax_b.text(c + 0.03, y, f"{c:.3f}", ha="left", va="center", fontsize=9, color=cs.INK, zorder=4)
    ax_b.set_yticks(ys)
    ax_b.set_yticklabels(groups)
    ax_b.set_xlim(0, 1.0)
    ax_b.set_xticks([0, 0.5, 1.0])
    ax_b.set_ylim(-0.6, len(groups) - 0.4)
    ax_b.set_xlabel("Sub-agent calls per example")
    ax_b.tick_params(axis="y", length=0)
    ax_b.spines["left"].set_visible(False)
    title_and_subtitle(ax_b, "(b) Calls per example", "mean over the same evaluations")
    macro = t3["Macro mean"]
    fig.text(0.01, 0.985,
             f"Learned delegation: +{macro[3]:.1f} pp over the candidate at {macro[4]:.3f} calls per example",
             ha="left", va="top", fontsize=11.5, fontweight="bold", color=cs.INK)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.26, top=0.80)
    cs.save(fig, "fig_disjoint_eval")


# ----------------------------------------------------------------------------
# Figure: accuracy vs calls, small multiples (Tables 4 and 5)
# ----------------------------------------------------------------------------
# label placement per (task, point): (dx, dy) in points, ha, va
LABEL_POS = {
    ("MedQA", "low"): (0, -10, "center", "top"),
    ("MedQA", "sel"): (-7, 7, "right", "bottom"),
    ("MedQA", "ver"): (9, -8, "left", "top"),
    ("MedQA", "all"): (-7, 7, "right", "bottom"),
    ("MMLU-Pro", "low"): (0, -10, "center", "top"),
    ("MMLU-Pro", "sel"): (-7, 7, "right", "bottom"),
    ("MMLU-Pro", "ver"): (9, -8, "left", "top"),
    ("MMLU-Pro", "all"): (-7, 7, "right", "bottom"),
    ("GPQA", "low"): (0, -10, "center", "top"),
    ("GPQA", "sel"): (-7, 7, "right", "bottom"),
    ("GPQA", "ver"): (9, -8, "left", "top"),
    ("GPQA", "all"): (-7, 7, "right", "bottom"),
    ("AQuA-RAT", "low"): (-9, -2, "right", "center"),
    ("AQuA-RAT", "sel"): (-2, 9, "center", "bottom"),
    ("AQuA-RAT", "ver"): (10, -7, "left", "top"),
    ("AQuA-RAT", "all"): (-7, 7, "right", "bottom"),
}
# a small glyph of the mark sits beside each value label, so a label next to two
# coincident markers (MedQA and GPQA at about one call) still says which mark it
# belongs to; the text itself stays in ink
LABEL_KEY = {
    "low": dict(marker="o", ms=5, color=C_MARGENT, mec=C_MARGENT, mew=0),
    "sel": dict(marker="o", ms=5, color=C_MARGENT, mec=C_MARGENT, mew=0),
    "ver": dict(marker="D", ms=6, mfc="none", mec=C_VERIFIER_FORCED, mew=1.3),
    "all": dict(marker="s", ms=6, mfc="none", mec=C_FORCE_ALL, mew=1.3),
}


def label_with_key(ax, fig, key: str, x, y, text: str, dx, dy, ha, va, gap: float = 5.5) -> None:
    """Value label in ink plus the series' marker glyph beside it (offsets in points)."""
    dyc = dy + {"top": -4.2, "bottom": 4.2, "center": 0.0}[va]
    if ha == "left":
        gx, tx = dx + 3, dx + 3 + gap
    elif ha == "right":
        gx, tx = dx - 3, dx - 3 - gap
    else:  # centre the glyph + text pair on dx
        gx, tx, ha = dx - 13, dx - 13 + gap, "left"
    ax.plot([x], [y], ls="none", zorder=5, **LABEL_KEY[key],
            transform=offset_copy(ax.transData, fig=fig, x=gx, y=dyc, units="points"))
    ax.annotate(text, (x, y), xytext=(tx, dy), textcoords="offset points",
                ha=ha, va=va, fontsize=9, color=cs.INK, zorder=5)


def fig_accuracy_vs_calls(t: dict) -> None:
    t4 = t["table4_low_call_vs_always_verifier"]
    t5 = t["table5_selected_vs_force_all"]
    tasks = ["MedQA", "MMLU-Pro", "GPQA", "AQuA-RAT"]
    fig, axes = plt.subplots(1, 4, figsize=(8.5, 4.1), sharey=True, gridspec_kw={"wspace": 0.14})
    for ax, task in zip(axes, tasks):
        ma4, mc4, va4, vc4, _ = t4[task]
        ma5, mc5, fa5, fc5, _ = t5[task]
        pts = {"low": (mc4, ma4), "sel": (mc5, ma5), "ver": (vc4, va4), "all": (fc5, fa5)}
        ax.grid(axis="both", color=cs.GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        # MARGENT: low-call point joined to the selected operating point
        ax.plot([mc4, mc5], [ma4, ma5], color=C_MARGENT, lw=1.3, zorder=2)
        ax.plot([mc4, mc5], [ma4, ma5], marker="o", ms=8, color=C_MARGENT, mec=cs.SURFACE, mew=1.5,
                ls="none", zorder=4)
        # forced baselines: open markers so an overlapping MARGENT dot stays visible inside
        ax.plot(vc4, va4, marker="D", ms=11, mfc="none", mec=C_VERIFIER_FORCED, mew=2, ls="none", zorder=3)
        ax.plot(fc5, fa5, marker="s", ms=11, mfc="none", mec=C_FORCE_ALL, mew=2, ls="none", zorder=3)
        for key, (xv, yv) in pts.items():
            dx, dy, ha, va = LABEL_POS[(task, key)]
            label_with_key(ax, fig, key, xv, yv, f"{yv:.1f}", dx, dy, ha, va)
        ax.set_title(task)
        ax.set_xlim(0, 3.2)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_ylim(50, 90)
        ax.set_yticks(range(50, 91, 10))
        ax.tick_params(axis="both", length=0)
    axes[0].set_ylabel("Accuracy (%)")
    fig.supxlabel("Sub-agent calls per example", fontsize=10.5, color=cs.INK2, y=0.165)
    fig.text(0.01, 0.985, "About one selected call matches three forced calls",
             ha="left", va="top", fontsize=11.5, fontweight="bold", color=cs.INK)
    fig.text(0.01, 0.935, "Tables 4 and 5, development diagnostics. Labels are accuracy in %, each with "
             "its mark's glyph. AQuA-RAT: forcing all three sub-agents is worse than 1.03 selected calls.",
             ha="left", va="top", fontsize=9.5, color=cs.INK2)
    handles = [
        Line2D([], [], color=C_MARGENT, lw=1.3, marker="o", ms=8, mec=cs.SURFACE, mew=1.5,
               label="MARGENT: low-call policy (Table 4) to selected operating point (Table 5)"),
        Line2D([], [], marker="D", ms=10, mfc="none", mec=C_VERIFIER_FORCED, mew=2, ls="none",
               label="Always Verifier, 1 forced call (Table 4)"),
        Line2D([], [], marker="s", ms=10, mfc="none", mec=C_FORCE_ALL, mew=2, ls="none",
               label="Force all three sub-agents, 3 calls (Table 5)"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, 0.01), ncol=1,
               borderaxespad=0, handletextpad=0.6, labelspacing=0.4, columnspacing=1.5)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.27, top=0.84)
    cs.save(fig, "fig_accuracy_vs_calls")


# ----------------------------------------------------------------------------
# Figure: correction vs corruption (Table 8)
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Figure: commit-to-rescue ratio arrows (Table 9)
# ----------------------------------------------------------------------------
PAIR_STYLE = {
    # name: (marker, label anchor (x, y) in data units, ha, va).  Every point is a
    # learned MARGENT policy (only rho changes), so all pairs share C_MARGENT and the
    # task / depth is carried by the marker shape and the text label.
    "MedQA depth 2": ("o", (0.70, 86.4), "center", "bottom"),
    "MedQA depth 1": ("^", (0.665, 79.6), "right", "top"),
    "MMLU-Pro depth 1": ("s", (0.525, 64.5), "left", "center"),
    "AQuA-RAT depth 3": ("D", (0.84, 73.4), "left", "center"),
}
# the MedQA depth-1 end point (0.560, 82.5) sits on the MedQA depth-2 arrow: that arrow
# is drawn thinner and the end markers carry a wider surface ring, so the ring breaks the line
ARROW_LW = {"MedQA depth 2": 1.4}
MARKER_SIZE = {"^": (9.5, 10.0)}  # (start, end) sizes; the triangle reads smaller than a circle
RHO_LABEL_OFFSET = {
    # name: ((dx, dy, ha, va) for start, (dx, dy, ha, va) for end), in points
    "MedQA depth 2": ((8, 2, "left", "center"), (-9, 0, "right", "center")),
    "MedQA depth 1": ((9, 0, "left", "center"), (0, 9, "center", "bottom")),
    "MMLU-Pro depth 1": ((0, -10, "center", "top"), (0, 9, "center", "bottom")),
    "AQuA-RAT depth 3": ((8, 0, "left", "center"), (-9, 0, "right", "center")),
}


def fig_rho_sweep(t: dict) -> None:
    t9 = t["table9_rho_comparisons"]
    fig, ax = plt.subplots(figsize=(7, 4.9))
    ax.grid(axis="both", color=cs.GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    handles = []
    for name, (marker, (lx, ly), ha, va) in PAIR_STYLE.items():
        colour = C_MARGENT
        lw = ARROW_LW.get(name, 1.8)
        ms_start, ms_end = MARKER_SIZE.get(marker, (8.5, 9.0))
        rf, rt, af, at_, cf, ct, gf, gt = t9[name]
        ax.annotate("", xy=(ct, at_), xytext=(cf, af),
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=lw, shrinkA=7, shrinkB=7,
                                    mutation_scale=14), zorder=2)
        ax.plot(cf, af, marker=marker, ms=ms_start, mfc=cs.SURFACE, mec=colour, mew=1.8, ls="none", zorder=3)
        ax.plot(ct, at_, marker=marker, ms=ms_end, color=colour, mec=cs.SURFACE, mew=1.8, ls="none", zorder=4)
        (sdx, sdy, sha, sva), (edx, edy, eha, eva) = RHO_LABEL_OFFSET[name]
        ax.annotate(f"{RHO} = {rf}", (cf, af), xytext=(sdx, sdy), textcoords="offset points",
                    ha=sha, va=sva, fontsize=9, color=cs.INK2, zorder=5)
        ax.annotate(f"{RHO} = {rt}", (ct, at_), xytext=(edx, edy), textcoords="offset points",
                    ha=eha, va=eva, fontsize=9, color=cs.INK2, zorder=5)
        task, depth = name.split(" depth ")
        ax.text(lx, ly,
                f"{task}, depth {depth}: {RHO} {rf} {ARROW} {rt}\n"
                f"call gap {gf:.3f} {ARROW} {gt:.3f}".replace(" 0.", " .").replace("gap 0.", "gap ."),
                ha=ha, va=va, fontsize=9, color=cs.INK, linespacing=1.4, zorder=5)
        handles.append(Line2D([], [], marker=marker, ms=8, color=colour, mec=cs.SURFACE, mew=1.2,
                              ls="-", lw=lw, label=f"{task}, depth {depth}"))
    handles += [
        Line2D([], [], marker="o", ms=8, mfc=cs.SURFACE, mec=cs.INK2, mew=1.8, ls="none",
               label=f"before: {RHO} from"),
        Line2D([], [], marker="o", ms=8, color=cs.INK2, mec=cs.SURFACE, mew=1.2, ls="none",
               label=f"after: {RHO} to (more commit examples)"),
    ]
    ax.set_xlim(0, 1.25)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0, 1.25])
    ax.set_ylim(60, 90)
    ax.set_yticks(range(60, 91, 5))
    ax.set_xlabel("Sub-agent calls per example")
    ax.set_ylabel("Accuracy (%)")
    ax.tick_params(axis="both", length=0)
    rf, rt, af, at_, cf, ct, gf, gt = t9["AQuA-RAT depth 3"]
    removed = 100 * (cf - ct) / cf
    lost = af - at_
    title_and_subtitle(
        ax, "More commit examples: fewer calls, accuracy not monotone",
        f"Table 9, development data; every point is a learned policy and each arrow changes only {RHO} "
        f"within a fixed collection and recipe.\nCall gap rises in every pair. AQuA-RAT, {RHO} 0 {ARROW} all: "
        f"removes {removed:.1f}% of calls and loses {lost:.1f} points.",
        pad=36, y=1.025)
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.085, -0.005), ncol=3,
               borderaxespad=0, handletextpad=0.6, labelspacing=0.4, columnspacing=1.6)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.24, top=0.84)
    cs.save(fig, "fig_rho_sweep")


# ----------------------------------------------------------------------------
# Figure: outcome-only GRPO collapses to three calls (Table 10 + released traces)
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def main() -> None:
    cs.apply_style()
    t = load()
    verify(t)
    print()
    fig_oracle_depth(t)
    fig_net_marginal_value(t)
    fig_disjoint_eval(t)
    fig_accuracy_vs_calls(t)
    fig_rho_sweep(t)
    print()
    if MISMATCHES:
        print(f"{len(MISMATCHES)} check(s) disagree:")
        for m in MISMATCHES:
            print("  -", m)
    else:
        print("all checks agree with the paper tables and text")


if __name__ == "__main__":
    main()
