#!/usr/bin/env python3
"""Figures 7 and 8 of the project page (MedQA routing run ``medqa_marginal_v1``).

  fig_counterfactual   accuracy of the five fixed one-step policies on the 400
                       counterfactual training questions, Wilson 95 % CI,
                       rescued / corrupted counts per advisor
  fig_routing_behavior (a) draft -> final transition classes on dev and test
                       (b) advisor-call count distribution on dev and test

Everything is computed from the raw files in
``supplementary_code/results/medqa_marginal_v1/``; nothing is typed in by hand.
The computed values are asserted against the numbers in the content spec and
both are printed, so a silent drift in the data would fail loudly.

Run:  /opt/anaconda3/bin/python3 static/charts/make_medqa_figures.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # research_milestones/
RUN_DIR = ROOT / "supplementary_code" / "results" / "medqa_marginal_v1"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "supplementary_code" / "scripts"))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from analyze_results import mcnemar_exact, wilson  # noqa: E402  (stdlib only)
from chart_style import (  # noqa: E402
    BLUE, GRID, INK, INK2, MUTED, NEUTRAL, ORANGE, RULE, STATUS, SURFACE,
    apply_style, save, xgrid, ygrid,
)

ADVISORS = ("extractor", "reasoner", "verifier")

# ----------------------------------------------------------------------------
# expected values from the content spec (CONTENT_SPEC.md, "MedQA routing run")
# ----------------------------------------------------------------------------
SPEC_CF = {
    "n": 400,
    "direct": (55.0, 50.1, 59.8),
    "extractor": (58.5, 53.6, 63.2, 25, 11, 0.029),
    "reasoner": (78.8, 74.5, 82.5, 100, 5, 5e-24),
    "verifier": (84.5, 80.6, 87.7, 127, 9, 8e-28),
    "hindsight": (91.0, 87.8, 93.4),
    "preferred": {"commit": 220, "rescue": 144, "unsolved": 36},
    "first_advisor": {"verifier": 86, "reasoner": 51, "extractor": 7},
    "net_pp": {"extractor": 3.5, "reasoner": 23.75, "verifier": 29.5},
}
SPEC_EVAL = {
    "dev": {"n": 200, "draft": (55.5, 48.6, 62.2), "final": (67.5, 60.7, 73.6),
            "corrected": 36, "corrupted": 12, "p": 7.2e-4, "avg_calls": 3.000,
            "calls": {3: 200}},
    "test": {"n": 200, "draft": (63.5, 56.6, 69.9), "final": (76.5, 70.2, 81.8),
             "corrected": 38, "corrupted": 12, "p": 3.1e-4, "avg_calls": 2.985,
             "calls": {1: 1, 2: 1, 3: 198}},
}


# ----------------------------------------------------------------------------
# computation from the raw files
# ----------------------------------------------------------------------------
def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def acc_ci(k: int, n: int) -> tuple[float, float, float]:
    lo, hi = wilson(k, n)
    return 100 * k / n, 100 * lo, 100 * hi


def compute_counterfactual() -> dict:
    records = read_jsonl(RUN_DIR / "counterfactual_records.jsonl")
    n = len(records)
    direct = [bool(r["direct_correct"]) for r in records]

    # first depth-1 branch per advisor, per question
    forced: dict[str, list[bool]] = {a: [] for a in ADVISORS}
    for r in records:
        seen: set[str] = set()
        for br in r["branches"]:
            seq = tuple(br["sequence"])
            if len(seq) == 1 and seq[0] in forced and seq[0] not in seen:
                forced[seq[0]].append(bool(br["correct"]))
                seen.add(seq[0])
    for a in ADVISORS:
        assert len(forced[a]) == n, f"{a}: {len(forced[a])} one-step branches, expected {n}"

    out = {"n": n, "direct": acc_ci(sum(direct), n), "advisors": {}}
    for a in ADVISORS:
        rescued = sum(1 for d, f in zip(direct, forced[a]) if not d and f)
        corrupted = sum(1 for d, f in zip(direct, forced[a]) if d and not f)
        out["advisors"][a] = {
            "acc": acc_ci(sum(forced[a]), n),
            "rescued": rescued,
            "corrupted": corrupted,
            "net_pp": 100 * (rescued - corrupted) / n,
            "p": mcnemar_exact(rescued, corrupted),
        }
    oracle = sum(1 for i, d in enumerate(direct) if d or any(forced[a][i] for a in ADVISORS))
    out["hindsight"] = acc_ci(oracle, n)

    pref = Counter()
    first = Counter()
    for r in records:
        seq = r["preferred_sequence"]
        if seq is None:
            pref["unsolved"] += 1
        elif len(seq) == 0:
            pref["commit"] += 1
        else:
            pref["rescue"] += 1
            first[seq[0]] += 1
    out["preferred"] = dict(pref)
    out["first_advisor"] = dict(first)
    return out


def compute_eval(path: Path) -> dict:
    rows = read_jsonl(path)
    n = len(rows)
    draft = [bool(r["initial_draft_correct"]) for r in rows]
    final = [bool(r["correct"]) for r in rows]
    cls = Counter()
    for d, f in zip(draft, final):
        cls[(d, f)] += 1
    corrected = cls[(False, True)]
    corrupted = cls[(True, False)]
    calls = Counter(int(r["tool_calls"]) for r in rows)
    return {
        "n": n,
        "draft": acc_ci(sum(draft), n),
        "final": acc_ci(sum(final), n),
        "stayed_correct": cls[(True, True)],
        "corrected": corrected,
        "corrupted": corrupted,
        "stayed_wrong": cls[(False, False)],
        "p": mcnemar_exact(corrected, corrupted),
        "avg_calls": sum(int(r["tool_calls"]) for r in rows) / n,
        "calls": {k: calls[k] for k in range(4)},
    }


# ----------------------------------------------------------------------------
# checks against the spec
# ----------------------------------------------------------------------------
def check(label: str, computed, expected, tol: float = 0.051) -> None:
    """Assert computed == expected (floats at 1-decimal precision, p-values at 10 % rel.)."""
    if isinstance(expected, float) and expected < 0.01:  # p-values quoted to 1-2 sig. figs
        ok = abs(computed - expected) / expected < 0.1
    elif isinstance(expected, float):
        ok = abs(computed - expected) < tol
    else:
        ok = computed == expected
    print(f"  {label:<44} computed {computed!s:<22} spec {expected!s:<22} {'ok' if ok else 'MISMATCH'}")
    assert ok, f"{label}: computed {computed} != spec {expected}"


def verify(cf: dict, ev: dict[str, dict]) -> None:
    print("Counterfactual one-step branches")
    check("n", cf["n"], SPEC_CF["n"])
    for key in ("direct", "hindsight"):
        for stat, c, e in zip(("acc", "lo", "hi"), cf[key], SPEC_CF[key]):
            check(f"{key} {stat}", round(c, 1), e)
    for a in ADVISORS:
        s = cf["advisors"][a]
        e_acc, e_lo, e_hi, e_res, e_cor, e_p = SPEC_CF[a]
        for stat, c, e in zip(("acc", "lo", "hi"), s["acc"], (e_acc, e_lo, e_hi)):
            check(f"{a} {stat}", round(c, 1), e)
        check(f"{a} rescued", s["rescued"], e_res)
        check(f"{a} corrupted", s["corrupted"], e_cor)
        check(f"{a} McNemar p", s["p"], e_p)
        check(f"{a} net marginal pp", s["net_pp"], SPEC_CF["net_pp"][a], tol=1e-9)
    check("preferred depth", cf["preferred"], SPEC_CF["preferred"])
    check("first advisor among rescues", cf["first_advisor"], SPEC_CF["first_advisor"])
    for split in ("dev", "test"):
        e = SPEC_EVAL[split]
        c = ev[split]
        print(f"Final manager eval, {split}")
        check("n", c["n"], e["n"])
        for key in ("draft", "final"):
            for stat, cv, ev_ in zip(("acc", "lo", "hi"), c[key], e[key]):
                check(f"{key} {stat}", round(cv, 1), ev_)
        check("corrected", c["corrected"], e["corrected"])
        check("corrupted", c["corrupted"], e["corrupted"])
        check("McNemar p", c["p"], e["p"])
        check("avg calls", round(c["avg_calls"], 3), e["avg_calls"], tol=1e-9)
        check("call distribution", {k: v for k, v in c["calls"].items() if v}, e["calls"])
        total = c["stayed_correct"] + c["corrected"] + c["corrupted"] + c["stayed_wrong"]
        check("transition classes sum to n", total, c["n"])


# ----------------------------------------------------------------------------
# drawing helpers
# ----------------------------------------------------------------------------
def title_and_subtitle(ax, title: str, subtitle: str, pad: float = 24) -> None:
    ax.set_title(title, pad=pad)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9.5, color=INK2)


def fig_counterfactual(cf: dict) -> None:
    rows = [("Direct (commit immediately)", cf["direct"], None, NEUTRAL)]
    for a in ADVISORS:
        s = cf["advisors"][a]
        rows.append((f"Always call {a}", s["acc"], s, BLUE))
    rows.append(("Hindsight best one-step branch", cf["hindsight"], None, NEUTRAL))

    fig, ax = plt.subplots(figsize=(7, 3.6))
    n = len(rows)
    ys = [n - 1 - i for i in range(n)]  # first row on top
    direct_acc = cf["direct"][0]

    xgrid(ax, zero=False)
    # hairline through the tip of the Direct bar: the part of each advisor bar to
    # its right is the gain over committing immediately
    ax.axvline(direct_acc, color=MUTED, linewidth=0.8, zorder=1)
    ax.text(direct_acc + 0.6, n - 0.35, "direct", ha="left", va="center",
            fontsize=8.5, color=MUTED)

    for y, (label, (acc, lo, hi), stats, colour) in zip(ys, rows):
        ax.barh(y, acc, height=0.46, color=colour, zorder=2)
        ax.errorbar(acc, y, xerr=[[acc - lo], [hi - acc]], fmt="none", ecolor=INK2,
                    elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=3)
        ax.text(hi + 1.2, y, f"{acc:.1f}", ha="left", va="center", fontsize=9.5,
                color=INK, zorder=4)
        if stats is not None:
            ax.text(0.6, y - 0.42,
                    f"rescued {stats['rescued']}, corrupted {stats['corrupted']}, "
                    f"net {stats['net_pp']:+g} pp",
                    ha="left", va="center", fontsize=8.5, color=MUTED, zorder=4)

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.75, n - 0.15)
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Accuracy (%), 95 % Wilson CI")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    title_and_subtitle(
        ax, "Fixed one-step policies on the counterfactual training set",
        f"MedQA, {cf['n']} questions, depth 1, temperature 0\n"
        "rescued = direct wrong and branch right, corrupted = the reverse",
        pad=34,  # two subtitle lines
    )
    handles = [Patch(color=BLUE, label="One forced advisor call"),
               Patch(color=NEUTRAL, label="Reference: no advisor / hindsight oracle")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
              handlelength=1.0, handleheight=0.8, borderaxespad=0, columnspacing=2.0)
    save(fig, "fig_counterfactual")


TRANSITIONS = [
    # key,             legend label,                       colour,               label ink
    ("stayed_correct", "Stayed correct (draft right, final right)", "#b9d3f3",       INK),
    ("corrected",      "Corrected (draft wrong, final right)",      STATUS["good"],  SURFACE),
    ("stayed_wrong",   "Stayed wrong (draft wrong, final wrong)",   NEUTRAL,         INK),
    ("corrupted",      "Corrupted (draft right, final wrong)",      STATUS["critical"], SURFACE),
]
# dev / test are also told apart by stacking order, count labels and the legend,
# so two greys suffice; this keeps blue / orange for the roles fixed in chart_style
SPLIT_COLOUR = {"dev": INK2, "test": MUTED}


def fig_routing_behavior(ev: dict[str, dict]) -> None:
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7, 3.4), gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.4},
    )
    splits = ["dev", "test"]
    arrow = r"$\rightarrow$"  # mathtext: the UI font has no U+2192 glyph

    # ---- (a) 100 % stacked horizontal bars of the transition classes --------
    ys = {"dev": 1, "test": 0}
    h = 0.42
    for split in splits:
        e = ev[split]
        y = ys[split]
        left = 0.0
        for key, _, colour, ink in TRANSITIONS:
            count = e[key]
            width = 100 * count / e["n"]
            ax_a.barh(y, width, left=left, height=h, color=colour,
                      edgecolor=SURFACE, linewidth=1.2, zorder=2)
            xc = left + width / 2
            if width >= 7.5:  # label fits inside with padding
                ax_a.text(xc, y, str(count), ha="center", va="center",
                          fontsize=9, color=ink, zorder=4)
            else:  # too narrow: short leader down to a label under the segment
                ax_a.plot([xc, xc], [y - h / 2, y - h / 2 - 0.07], color=INK2,
                          linewidth=0.8, zorder=4, solid_capstyle="butt")
                ax_a.text(xc, y - h / 2 - 0.09, str(count), ha="center",
                          va="top", fontsize=9, color=INK, zorder=4)
            left += width
        ax_a.text(0, y + h / 2 + 0.06,
                  f"{split.capitalize()} (n = {e['n']}): draft {e['draft'][0]:.1f} %{arrow}"
                  f"final {e['final'][0]:.1f} %",
                  ha="left", va="bottom", fontsize=9.5, color=INK)

    ax_a.set_xlim(0, 100)
    ax_a.set_ylim(-0.6, 1.72)
    ax_a.set_xticks(range(0, 101, 25))
    ax_a.set_xlabel("Share of questions (%)")
    ax_a.set_yticks([])
    ax_a.spines["left"].set_visible(False)
    xgrid(ax_a, zero=False)
    title_and_subtitle(ax_a, f"(a) Draft{arrow}final transitions",
                       "counts per class, 200 questions per split")
    handles = [Patch(facecolor=c, label=lbl) for _, lbl, c, _ in TRANSITIONS]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.125, -0.08), ncol=2,
               handlelength=1.0, handleheight=0.8, fontsize=9, borderaxespad=0,
               columnspacing=1.6, labelspacing=0.35)

    # ---- (b) advisor-call count distribution, horizontal so end labels never
    #          collide (dev and test both pile up at k = 3) ------------------
    ks = [0, 1, 2, 3]
    yk = {k: len(ks) - 1 - k for k in ks}  # k = 0 on top
    hb = 0.34
    for i, split in enumerate(splits):
        e = ev[split]
        counts = [e["calls"][k] for k in ks]
        yy = [yk[k] + (0.5 - i) * (hb + 0.04) for k in ks]  # dev above test
        ax_b.barh(yy, counts, height=hb, color=SPLIT_COLOUR[split], zorder=2,
                  label=f"{split.capitalize()} (n = {e['n']})")
        for y, c in zip(yy, counts):
            ax_b.text(c + 4, y, str(c), ha="left", va="center", fontsize=9, color=INK, zorder=4)
    xgrid(ax_b, zero=False)
    ax_b.set_yticks([yk[k] for k in ks])
    ax_b.set_yticklabels([str(k) for k in ks])
    ax_b.set_ylabel("Advisor calls (k)")
    ax_b.set_xlabel("Episodes")
    ax_b.set_xlim(0, 240)
    ax_b.set_xticks(range(0, 201, 100))
    ax_b.set_ylim(-0.6, len(ks) - 0.4)
    ax_b.spines["left"].set_visible(False)
    ax_b.tick_params(axis="y", length=0)
    three = sum(ev[s]["calls"][3] for s in splits)
    total = sum(ev[s]["n"] for s in splits)
    title_and_subtitle(ax_b, "(b) Advisor calls per episode",
                       f"k = 3 in {three} of {total} episodes")
    ax_b.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2)
    save(fig, "fig_routing_behavior")


def main() -> None:
    apply_style()
    cf = compute_counterfactual()
    ev = {"dev": compute_eval(RUN_DIR / "eval_dev200.jsonl"),
          "test": compute_eval(RUN_DIR / "eval_test200.jsonl")}
    verify(cf, ev)
    for split in ("dev", "test"):
        e = ev[split]
        print(f"{split} transition classes: stayed correct {e['stayed_correct']}, "
              f"corrected {e['corrected']}, corrupted {e['corrupted']}, "
              f"stayed wrong {e['stayed_wrong']}")
    fig_counterfactual(cf)
    fig_routing_behavior(ev)


if __name__ == "__main__":
    main()
