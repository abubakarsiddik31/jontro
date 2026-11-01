"""Recompute every corpus statistic the Jontro paper cites.

Single source of truth: reads the released canonical JSONL, the split id
manifests, and the goal manifest, and writes

- outputs/stats/jontro2_stats.json   (all counts, machine-readable)
- outputs/stats/fig_*.pdf|png        (publication figures)

Nothing here hard-codes a number; if the corpus changes, re-run and the paper's
figures/tables change with it.

Usage:
    python scripts/jontro2_stats.py
"""
from __future__ import annotations

import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import numpy as np  # noqa: E402

CANONICAL = Path("outputs/release_multiturn/jontro_multiturn_2000.jsonl")
SPLIT_DIR = Path("outputs/splits")
OUT = Path("outputs/stats")
FIG_DIR = Path("paper/figures")  # what main.tex includes; kept byte-identical to OUT
TEXTWIDTH_IN = 6.57  # \textwidth of the manuscript (10pt A4 article, 0.85in margins)

# Okabe-Ito palette, shared with the manuscript's TikZ diagrams.
JBLUE, JORANGE, JGREEN, JGRAY = "#0072B2", "#E69F00", "#009E73", "#4D4D4D"
JRED, JSKY = "#D55E00", "#56B4E9"
JPINK = "#CC79A7"
# One colour per system class, used by every evaluation figure (validated for
# colour-vision deficiency): floors grey, zero-shot vermillion, fine-tuned blue/green.
C_FLOOR, C_ORACLE = "#8A8A8A", "#222222"
C_ZS, C_SFT, C_MIX = JRED, JBLUE, JGREEN
# Tool sectors for the inventory figure (domain -> sector); four hues, fixed order.
SECTORS = [
    ("Transport", JBLUE, {"bus_launch_booking", "railway_booking", "flight_booking", "city_transport", "travel_planning"}),
    ("Finance & telecom", JRED, {"mobile_finance", "telecom"}),
    ("Public services", JGREEN, {"govt_services", "healthcare", "education", "agri_weather"}),
    ("Commerce & lifestyle", JPINK, {"ecommerce", "food_delivery", "real_estate", "movie_cinema", "job_search", "news", "sports_scores"}),
]

DOMAIN_LABELS = {
    "agri_weather": "Agriculture/weather",
    "bus_launch_booking": "Bus/launch booking",
    "city_transport": "City transport",
    "ecommerce": "E-commerce",
    "education": "Education",
    "flight_booking": "Flight booking",
    "food_delivery": "Food delivery",
    "govt_services": "Government services",
    "healthcare": "Healthcare",
    "job_search": "Job search",
    "mobile_finance": "Mobile finance",
    "movie_cinema": "Movie/cinema",
    "news": "News",
    "railway_booking": "Railway booking",
    "real_estate": "Real estate",
    "sports_scores": "Sports scores",
    "telecom": "Telecom",
    "travel_planning": "Travel planning",
}

BANGLA_DIGITS = "০১২৩৪৫৬৭৮৯"
_BN_DIGIT_RE = re.compile(f"[{BANGLA_DIGITS}]")


def _setup_style() -> None:
    # Each figure is drawn at exactly the width \includegraphics gives it (a
    # fraction of TEXTWIDTH_IN) and saved uncropped, so the point sizes below are
    # the sizes on the printed page: 9pt axis titles, 8pt ticks, 7.5pt in-figure
    # numbers, sans-serif lettering as journal artwork guidelines ask for.
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "standard",
            "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],  # a list: glyph fallback (e.g. the arrow)
            "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.monospace": ["Menlo", "DejaVu Sans Mono", "Courier New"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "sans",
            "mathtext.it": "sans:italic",
            "mathtext.bf": "sans:bold",
            "pdf.fonttype": 42,  # TrueType in PDF; kills Type 3
            "ps.fonttype": 42,  # same for PostScript output
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.labelpad": 4,
            "axes.linewidth": 0.6,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#B0B0B0",
            "grid.alpha": 0.5,
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.5,
        }
    )


def _comma(v: float, _pos: int | None = None) -> str:
    """Tick formatter with thousands separators (e.g. 2,000)."""
    return f"{v:,.0f}"


def _tag(ax, label: str) -> None:
    """Bold panel tag, e.g. '(a)' — the LaTeX caption is the title."""
    ax.text(0.0, 1.02, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def _fig(width_frac: float, height_in: float, **subplots_kw):
    """Figure drawn at width_frac * \\textwidth and laid out so every label fits
    inside the canvas without post-hoc cropping: the include width then maps 1:1
    and the point sizes are what the reader sees."""
    fig, axes = plt.subplots(
        figsize=(TEXTWIDTH_IN * width_frac, height_in), layout="constrained", **subplots_kw
    )
    fig.get_layout_engine().set(w_pad=2 / 72, h_pad=2 / 72)
    return fig, axes


def _label_bars(ax, xs, vals, dy, fontsize: float = 7.5, skip_zero: bool = True) -> None:
    """Counts above bars; empty bins stay unlabelled rather than reading '0'."""
    for x, v in zip(xs, vals):
        if skip_zero and not v:
            continue
        ax.text(x, v + dy, f"{v:,}", ha="center", va="bottom", fontsize=fontsize, color="#333333")


def _save(fig, name: str) -> None:
    """Write to outputs/stats and mirror the identical bytes into paper/figures,
    so the manuscript cannot include a figure this script did not just draw."""
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        src = OUT / f"{name}.{ext}"
        fig.savefig(src)
        if FIG_DIR.is_dir():
            shutil.copyfile(src, FIG_DIR / f"{name}.{ext}")
    plt.close(fig)


def _pct(x: float, n: int) -> float:
    return round(100.0 * x / n, 2) if n else 0.0


def main() -> None:
    _setup_style()
    OUT.mkdir(parents=True, exist_ok=True)

    recs: list[dict] = []
    with CANONICAL.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    n = len(recs)

    domains: Counter = Counter()
    registers: Counter = Counter()
    scripts: Counter = Counter()
    cells: Counter = Counter()
    spl_lang: Counter = Counter()
    tech: Counter = Counter()
    closures: Counter = Counter()
    terminated: Counter = Counter()
    goals: Counter = Counter()
    tools_offered: Counter = Counter()
    tools_used_hist: Counter = Counter()
    user_turns: Counter = Counter()
    total_msgs: list[int] = []
    decision_points_per_traj: Counter = Counter()
    calls_per_step: Counter = Counter()
    dp_by_turn_index: Counter = Counter()
    role_totals: Counter = Counter()
    role_shapes: set[str] = set()

    tool_names: set[str] = set()
    tool_use_counts: Counter = Counter()
    tool_domains: dict[str, Counter] = defaultdict(Counter)

    n_decision_points = n_tool_calls = n_textonly_assistant = 0
    n_multi_call_messages = 0
    n_args_with_null = n_calls_total = 0
    bangla_digit_trajectories = 0
    ids: list[str] = []
    seeds: list[str] = []
    all_sim_seeds: list[str] = []

    validation_flags: dict[str, Counter] = defaultdict(Counter)
    validation_num: dict[str, list[float]] = defaultdict(list)

    for r in recs:
        ids.append(r["id"])
        domains[r["domain"]] += 1
        p = r.get("persona", {})
        registers[p.get("register", "apni")] += 1
        scripts[p.get("script", "bengali")] += 1
        cells[(p.get("register", "apni"), p.get("script", "bengali"))] += 1
        spl_lang[p.get("system_prompt_language", "bangla")] += 1
        tech[p.get("tech_literacy", "medium")] += 1

        md = r.get("metadata", {})
        closures[md.get("closure_style") or "unset"] += 1
        terminated[md.get("terminated") or "unset"] += 1
        goals[md.get("persona_goal") or "<none>"] += 1
        seeds.append(str(md.get("seed")))
        for v in (md.get("sim_seeds") or {}).values():
            all_sim_seeds.append(str(v))

        tools = r["tools"]
        tools_offered[len(tools)] += 1
        for t in tools:
            tool_names.add(t["name"])
        used = md.get("tools_used") or []
        tools_used_hist[len(used) if isinstance(used, list) else used] += 1
        for name in used if isinstance(used, list) else []:
            tool_use_counts[name] += 1
            tool_domains[name][r["domain"]] += 1

        msgs = r["messages"]
        total_msgs.append(len(msgs))
        role_totals.update(m["role"] for m in msgs)
        shape = ",".join(m["role"][0].upper() for m in msgs)
        role_shapes.add(shape)

        marks = "".join("A" if m["role"] == "assistant" else ("U" if m["role"] == "user" else ("T" if m["role"] == "tool" else "S")) for m in msgs)
        if "A" not in marks:
            role_shapes.add("__no_assistant__")

        ut = sum(1 for m in msgs if m["role"] == "user")
        user_turns[ut] += 1

        dp_here = 0
        seen_user = 0
        has_bn_digit = False
        for m in msgs:
            if m["role"] == "user":
                seen_user += 1
                if _BN_DIGIT_RE.search(m.get("content") or ""):
                    has_bn_digit = True
            if m["role"] == "assistant":
                tcs = m.get("tool_calls") or []
                if tcs:
                    dp_here += 1
                    calls_per_step[len(tcs)] += 1
                    dp_by_turn_index[seen_user] += 1
                    n_tool_calls += len(tcs)
                    if len(tcs) > 1:
                        n_multi_call_messages += 1
                    for tc in tcs:
                        n_calls_total += 1
                        args = tc.get("arguments") or {}
                        if any(v is None for v in args.values()):
                            n_args_with_null += 1
                else:
                    n_textonly_assistant += 1
        n_decision_points += dp_here
        decision_points_per_traj[dp_here] += 1
        if has_bn_digit:
            bangla_digit_trajectories += 1

        val = md.get("validation") or {}
        for k, v in val.items():
            if isinstance(v, bool):
                validation_flags[k][v] += 1
            elif isinstance(v, (int, float)):
                validation_num[k].append(float(v))

    # ---------------------------------------------------------------- splits
    def _ids(path: Path) -> list[str]:
        if not path.exists():
            return []
        return [json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()]

    train_ids = _ids(SPLIT_DIR / "train_ids.jsonl")
    test_ids = _ids(SPLIT_DIR / "test_ids.jsonl")
    by_id = {r["id"]: r for r in recs}
    train_goals = {by_id[i]["metadata"].get("persona_goal") for i in train_ids if i in by_id}
    test_goals = {by_id[i]["metadata"].get("persona_goal") for i in test_ids if i in by_id}

    def _split_summary(sel: list[str]) -> dict:
        dp = calls = argfree = 0
        doms: Counter = Counter()
        for i in sel:
            r = by_id.get(i)
            if not r:
                continue
            doms[r["domain"]] += 1
            for m in r["messages"]:
                if m["role"] == "assistant" and m.get("tool_calls"):
                    dp += 1
                    calls += len(m["tool_calls"])
                    if all(
                        not [v for v in (c.get("arguments") or {}).values() if v is not None]
                        for c in m["tool_calls"]
                    ):
                        argfree += 1
        return {
            "n_records": len(sel),
            "n_goals": len({by_id[i]["metadata"].get("persona_goal") for i in sel if i in by_id}),
            "n_decision_points": dp,
            "n_tool_calls": calls,
            "arg_free_decision_points": argfree,
            "arg_free_decision_points_pct": round(100 * argfree / dp, 2) if dp else 0.0,
            "domains": len(doms),
            "domain_counts": dict(doms.most_common()),
        }

    test_goal_manifest = []
    tg = SPLIT_DIR / "test_goals.jsonl"
    if tg.exists():
        test_goal_manifest = [json.loads(line) for line in tg.read_text().splitlines() if line.strip()]

    stats = {
        "corpus": {
            "n_trajectories": n,
            "n_unique_ids": len(set(ids)),
            "n_duplicate_ids": n - len(set(ids)),
            "n_domains": len(domains),
            "n_goals": len([g for g in goals if g != "<none>"]),
            "n_distinct_tools_offered": len(tool_names),
            "n_distinct_tools_called": len(tool_use_counts),
            "n_seeds_distinct": len({s for s in seeds if s != "None"}),
            "n_sim_seeds_total": len(all_sim_seeds),
            "n_sim_seeds_distinct": len(set(all_sim_seeds)),
            "n_decision_points": n_decision_points,
            "n_tool_calls": n_tool_calls,
            "n_textonly_assistant_messages": n_textonly_assistant,
            "n_multi_call_messages": n_multi_call_messages,
            "n_assistant_messages": role_totals["assistant"],
            "role_totals": dict(role_totals),
            "n_role_shape_patterns": len(role_shapes),
            "tool_use_rate_pct": _pct(sum(1 for r in recs if (r["metadata"].get("tools_used") or [])), n),
            "mean_calls_per_decision_point": round(n_tool_calls / n_decision_points, 3) if n_decision_points else 0,
            "null_valued_args_in_calls": n_args_with_null,
            "calls_with_null_args_pct": _pct(n_args_with_null, n_calls_total),
            "trajectories_with_bangla_digits_pct": _pct(bangla_digit_trajectories, n),
        },
        "domains": dict(domains.most_common()),
        "domain_labels": DOMAIN_LABELS,
        "registers": dict(registers.most_common()),
        "scripts": dict(scripts.most_common()),
        "register_script_cells": {f"{k[0]}|{k[1]}": v for k, v in sorted(cells.items())},
        "system_prompt_language": dict(spl_lang.most_common()),
        "tech_literacy": dict(tech.most_common()),
        "closure_style": dict(closures.most_common()),
        "terminated": dict(terminated.most_common()),
        "goals": {
            "n_distinct": len([g for g in goals if g != "<none>"]),
            "records_per_goal_min": min(goals.values()) if goals else 0,
            "records_per_goal_median": float(np.median(list(goals.values()))) if goals else 0,
            "records_per_goal_max": max(goals.values()) if goals else 0,
        },
        "user_turns_per_traj": dict(sorted(user_turns.items())),
        "user_turns_mean": round(float(np.mean([k for k, v in user_turns.items() for _ in range(v)])), 2) if user_turns else 0,
        "user_turns_min": min(user_turns) if user_turns else 0,
        "user_turns_max": max(user_turns) if user_turns else 0,
        "messages_per_traj": {
            "min": int(min(total_msgs)),
            "max": int(max(total_msgs)),
            "mean": round(float(np.mean(total_msgs)), 2),
            "median": float(np.median(total_msgs)),
        },
        "tools_offered_per_traj": dict(sorted(tools_offered.items())),
        "tools_offered_mean": round(float(np.mean([k for k, v in tools_offered.items() for _ in range(v)])), 2) if tools_offered else 0,
        "tools_used_per_traj": dict(sorted(tools_used_hist.items())),
        "decision_points_per_traj": dict(sorted(decision_points_per_traj.items())),
        "calls_per_step": dict(sorted(calls_per_step.items())),
        "decision_points_by_turn_index": dict(sorted(dp_by_turn_index.items())),
        "tool_use_counts": dict(tool_use_counts.most_common()),
        "validation_flags": {k: dict(v) for k, v in sorted(validation_flags.items())},
        "validation_numeric": {k: {"min": min(v), "max": max(v), "mean": round(float(np.mean(v)), 3)} for k, v in sorted(validation_num.items()) if v},
        "splits": {
            "train": _split_summary(train_ids),
            "test": _split_summary(test_ids),
            "id_overlap": len(set(train_ids) & set(test_ids)),
            "goal_overlap": len(train_goals & test_goals),
            "goal_manifest_entries": len(test_goal_manifest),
        },
    }

    (OUT / "jontro2_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------------------------------------------------------------- figures
    # 1. decision points by turn index (benchmark depth) — linear scale
    ks = sorted(dp_by_turn_index)
    turn_vals = [dp_by_turn_index[k] for k in ks]
    turn_tot = sum(turn_vals)
    fig, ax = _fig(0.60, 2.55)  # placed at 0.6\textwidth
    ax.bar([str(k) for k in ks], turn_vals, width=0.62,
           color=[C_FLOOR if k == ks[0] else JBLUE for k in ks], edgecolor="white", linewidth=0.6)
    for k, v in zip(ks, turn_vals):
        ax.text(
            str(k), v + turn_tot * 0.012, f"{v:,}\n({100.0 * v / turn_tot:.1f}%)",
            ha="center", va="bottom", fontsize=7.5, color="#333333", linespacing=1.15,
        )
    share_after = 100.0 * (turn_tot - dp_by_turn_index[ks[0]]) / turn_tot
    ax.text(
        0.98, 0.96, f"{share_after:.1f}% of decision points occur\nafter the first user turn",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, fontweight="bold", color=JBLUE,
    )
    ax.set_xlabel("Turn index $t$ (user turn the decision point follows)")
    ax.set_ylabel("Decision points")
    ax.set_ylim(0, max(turn_vals) * 1.22)
    ax.yaxis.set_major_formatter(FuncFormatter(_comma))
    _save(fig, "fig_decision_points_by_turn")

    # 10. tool inventory: how many conversations use each tool
    fig, ax = _fig(0.74, 3.7)  # placed at 0.74\textwidth
    names = [k for k, _ in tool_use_counts.most_common(16)][::-1]
    tool_vals = [tool_use_counts[k] for k in names]
    def _sector(tool: str) -> int:
        dom = tool_domains[tool].most_common(1)[0][0]
        return next(i for i, (_, _, doms) in enumerate(SECTORS) if dom in doms)
    sec = [_sector(n) for n in names]
    ax.barh(names, tool_vals, color=[SECTORS[i][1] for i in sec], height=0.72, edgecolor="white", linewidth=0.5)
    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(facecolor=c, label=lab) for lab, c, _ in SECTORS if any(SECTORS[i][0] == lab for i in sec)],
        loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=7.5, title="Sector of the calling domain", title_fontsize=7.5,
        alignment="left", handlelength=1.2, borderaxespad=0.2,
    )
    for i, v in enumerate(tool_vals):
        ax.text(v + max(tool_vals) * 0.012, i, f"{v:,}", va="center", fontsize=7.5, color="#333333")
    ax.set_yticks(range(len(names)), names, family="monospace")  # identifiers, set as in the text
    ax.set_xlabel("Conversations using the tool")
    ax.set_xlim(0, max(tool_vals) * 1.12)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.xaxis.set_major_formatter(FuncFormatter(_comma))
    ax.grid(False, axis="y")
    ax.grid(True, axis="x")
    _save(fig, "fig_tools")

    # ------------------------------------------------- evaluation-side figures
    scores_path = Path("outputs/eval/multiturn_scores.json")
    rean_path = Path("outputs/stats/eval_reanalysis.json")
    if scores_path.exists() and rean_path.exists():
        scores = json.loads(scores_path.read_text())
        reanalysis = json.loads(rean_path.read_text())
        _fig_forest(scores, reanalysis)
        _fig_turn_curve(scores)
        _fig_shape_panel(user_turns, decision_points_per_traj, calls_per_step, total_msgs)

    print(f"wrote {OUT / 'jontro2_stats.json'}")
    print(json.dumps({k: stats["corpus"][k] for k in stats["corpus"]}, indent=2, ensure_ascii=False))


def _fig_forest(scores: dict, reanalysis: dict) -> None:
    """Central results figure: (a) tool selection, (b) argument exactness by
    system, coloured by system class; (c) goal-clustered paired differences."""
    systems = [
        ("ORACLE-gold", "Oracle (gold replay)", C_ORACLE, True),
        ("majority-global", "Majority tool (global)", C_FLOOR, False),
        ("majority-domain", "Majority tool (domain)", C_FLOOR, False),
        ("first-offered", "First offered tool", C_FLOOR, False),
        ("random-offered", "Random offered tool", C_FLOOR, False),
        ("Qwen3.5-2B-zeroshot", "Qwen3.5-2B zero-shot", C_ZS, True),
        ("Qwen3.5-4B-zeroshot", "Qwen3.5-4B zero-shot", C_ZS, True),
        ("Qwen3.5-9B-zeroshot", "Qwen3.5-9B zero-shot", C_ZS, True),
        ("Qwen3.5-2B-SFT", "Qwen3.5-2B SFT", C_SFT, True),
        ("Qwen3.5-2B-SFT-mix", "Qwen3.5-2B SFT-mix", C_MIX, True),
    ]
    bands = [(1, 4, "baselines"), (5, 7, "zero-shot"), (8, 9, "fine-tuned")]
    ny = len(systems)
    fig, axes = _fig(0.80, 6.5, nrows=3, ncols=1, gridspec_kw={"height_ratios": [ny, ny, 7.6]})
    fig.get_layout_engine().set(hspace=0.06)
    ax_a, ax_b, ax_c = axes

    def _interval_panel(ax, rate_key: str, ci_key: str, tag: str, xlabel: str) -> None:
        for k, (i0, i1, _) in enumerate(bands):
            if k % 2 == 0:  # faint bands separate the system classes
                ax.axhspan(ny - 1 - i1 - 0.5, ny - 1 - i0 + 0.5, color="#F1F1F1", zorder=0, lw=0)
        for i, (key, _, col, filled) in enumerate(systems):
            o = scores[key]["overall"]
            est = 100.0 * o[rate_key]
            lo, hi = o[ci_key]
            y = ny - 1 - i
            ax.plot([lo, hi], [y, y], color=col, lw=1.6, solid_capstyle="round", zorder=2)
            ax.plot(est, y, marker="o", ms=5.2, mfc=col if filled else "white", mec=col, mew=1.2, zorder=3)
            if hi > 95:
                ax.text(lo - 2.2, y, f"{est:.1f}", ha="right", va="center", fontsize=7.5, color="#333333")
            else:
                ax.text(hi + 2.0, y, f"{est:.1f}", ha="left", va="center", fontsize=7.5, color="#333333")
        ax.set_yticks(list(range(ny))[::-1], [s[1] for s in systems])
        for tick, (_, _, col, _) in zip(ax.get_yticklabels(), systems):
            tick.set_color(col if col != C_FLOOR else "#555555")
        ax.set_ylim(-0.6, ny - 0.4)
        ax.set_xlim(-3, 106)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel(xlabel)
        ax.grid(False, axis="y")
        ax.grid(True, axis="x")
        _tag(ax, tag)

    _interval_panel(ax_a, "tool_selection_rate", "tool_selection_ci", "(a)", "Tool selection (%)")
    _interval_panel(ax_b, "arg_exact_rate", "arg_exact_ci", "(b)", "Argument exactness (%)")

    groups = [
        ("sft_minus_2b_zeroshot", "2B SFT $-$ 2B zero-shot", C_SFT),
        ("sft_minus_sft_mix", "2B SFT $-$ SFT-mix", C_MIX),
        ("4b_minus_2b_zeroshot", "4B $-$ 2B zero-shot", C_ZS),
        ("9b_minus_2b_zeroshot", "9B $-$ 2B zero-shot", C_ZS),
    ]
    metrics = [("tool_selection", "tool selection", "o", 0.2), ("arg_exact", "argument exactness", "s", -0.2)]
    diffs = reanalysis["paired_diffs"]
    ng = len(groups)
    for g, (gkey, _, col) in enumerate(groups):
        y0 = ng - 1 - g
        x_lab = max(diffs[gkey][mk]["ci95_pct"][1] for mk, _, _, _ in metrics) + 1.5
        for mk, _, marker, dy in metrics:
            d = diffs[gkey][mk]
            lo, hi = d["ci95_pct"]
            est = d["diff_pct"]
            ax_c.plot([lo, hi], [y0 + dy] * 2, color=col, lw=1.6, solid_capstyle="round", zorder=2)
            ax_c.plot(est, y0 + dy, marker=marker, ms=5.0, mfc=col if d["excludes_zero"] else "white",
                      mec=col, mew=1.2, zorder=3)
            ax_c.text(x_lab, y0 + dy, f"{est:+.1f}", ha="left", va="center", fontsize=7, color="#333333")
    ax_c.axvline(0, color="#555555", lw=0.9, zorder=1)
    ax_c.set_yticks(list(range(ng))[::-1], [g[1] for g in groups])
    for tick, (_, _, col) in zip(ax_c.get_yticklabels(), groups):
        tick.set_color(col)
    ax_c.set_ylim(-0.6, ng - 0.4)
    ax_c.set_xlim(-8, 57)
    ax_c.set_xticks([0, 10, 20, 30, 40, 50])
    ax_c.set_xlabel("Paired difference (points, goal-clustered 95% interval)")
    ax_c.grid(False, axis="y")
    ax_c.grid(True, axis="x")
    from matplotlib.lines import Line2D
    ax_c.legend(
        handles=[Line2D([], [], color="#555555", marker=m, ls="", ms=5, mfc="#555555", label=lab)
                 for _, lab, m, _ in metrics]
        + [Line2D([], [], color="#555555", marker="o", ls="", ms=5, mfc="white", label="interval spans zero")],
        loc="lower right", fontsize=7.5, handletextpad=0.3, borderaxespad=0.2,
    )
    _tag(ax_c, "(c)")
    _save(fig, "fig_forest")


def _fig_turn_curve(scores: dict) -> None:
    """Per-turn tool selection and argument exactness for the offered-tool
    floors and the 2B models (teacher-forced), at turn positions with n >= 30."""
    turn_keys = [
        t for t in range(12)
        if float(scores["first-offered"]["by_turn_index"][str(t)]["n"]) >= 30
    ]
    xs = [t + 1 for t in turn_keys]  # 0-based step index -> turn position
    series = [
        ("first-offered", "first offered (baseline)", C_FLOOR, "-", "o", "white"),
        ("random-offered", "random offered (baseline)", C_FLOOR, "--", "s", "white"),
        ("Qwen3.5-2B-zeroshot", "2B zero-shot", C_ZS, "-", "^", C_ZS),
        ("Qwen3.5-2B-SFT", "2B SFT", C_SFT, "-", "D", C_SFT),
    ]
    panels = [
        ("tool_selection_rate", "Tool selection (%)", "(a)"),
        ("arg_exact_rate", "Argument exactness (%)", "(b)"),
    ]
    fig, axes = _fig(1.0, 2.25, nrows=1, ncols=2, sharey=True)  # placed at full \textwidth
    fig.get_layout_engine().set(wspace=0.04)
    for ax, (rate_key, ylabel, tag) in zip(axes, panels):
        for skey, slab, col, ls, mk, face in series:
            bti = scores[skey]["by_turn_index"]
            ys = [100.0 * bti[str(t)][rate_key] for t in turn_keys]
            ax.plot(xs, ys, color=col, linestyle=ls, marker=mk, ms=4.5, mfc=face,
                    mec=col, linewidth=1.4, label=slab)
        ax.set_title(ylabel, fontsize=9, loc="center")
        ax.set_xlabel("Tool-calling turn position $t$")
        ax.set_ylim(-4, 104)
        ax.set_xlim(0.6, xs[-1] + 0.4)
        ax.set_xticks(xs)
        _tag(ax, tag)
    axes[0].set_ylabel("Score (%)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(0.0, 0.9), ncol=2, fontsize=7, handlelength=2.0,
                   columnspacing=0.8)
    _save(fig, "fig_turn_curve")


def _fig_shape_panel(
    user_turns: Counter,
    decision_points_per_traj: Counter,
    calls_per_step: Counter,
    total_msgs: list[int],
) -> None:
    """Optional 2x2 appendix panel: conversation-shape histograms, shared style."""
    fig, axes = _fig(1.0, 3.9, nrows=2, ncols=2)  # placed at full \textwidth
    fig.get_layout_engine().set(wspace=0.08, hspace=0.1)
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # (a) user turns
    ks = sorted(user_turns)
    vals = [user_turns[k] for k in ks]
    ax_a.bar([str(k) for k in ks], vals, color=JBLUE, width=0.62, edgecolor="white", linewidth=0.6)
    _label_bars(ax_a, [str(k) for k in ks], vals, max(vals) * 0.02)
    ax_a.set_ylim(0, max(vals) * 1.18)
    ax_a.set_xlabel("User turns")
    ax_a.set_ylabel("Conversations")

    # (b) decision points per conversation, full integer axis incl. empty bins
    ks = list(range(1, max(decision_points_per_traj) + 1))
    counts = [decision_points_per_traj.get(k, 0) for k in ks]
    ax_b.bar([str(k) for k in ks], counts, width=0.62, edgecolor="white", linewidth=0.6,
             color=[C_FLOOR if k == 1 else JGREEN for k in ks])
    multi = 100.0 * sum(c for k, c in zip(ks, counts) if k > 1) / sum(counts)
    ax_b.text(0.97, 0.95, f"{multi:.1f}% of conversations\nhave two or more", transform=ax_b.transAxes,
              ha="right", va="top", fontsize=7.5, fontweight="bold", color=JGREEN)
    _label_bars(ax_b, [str(k) for k in ks], counts, max(counts) * 0.02, fontsize=7)
    ax_b.set_ylim(0, max(counts) * 1.18)
    ax_b.set_xlabel("Tool-calling turns")
    ax_b.set_ylabel("Conversations")

    # (c) calls per decision point
    ks = sorted(calls_per_step)
    vals = [calls_per_step[k] for k in ks]
    ax_c.bar([str(k) for k in ks], vals, width=0.62, edgecolor="white", linewidth=0.6,
             color=[C_FLOOR if k == 1 else JRED for k in ks])
    par = 100.0 * sum(v for k, v in zip(ks, vals) if k > 1) / sum(vals)
    ax_c.text(0.97, 0.95, f"{par:.1f}% of tool-calling turns\nissue parallel calls", transform=ax_c.transAxes,
              ha="right", va="top", fontsize=7.5, fontweight="bold", color=JRED)
    _label_bars(ax_c, [str(k) for k in ks], vals, max(vals) * 0.02)
    ax_c.set_ylim(0, max(vals) * 1.16)
    ax_c.set_xlabel("Calls in one assistant turn")
    ax_c.set_ylabel("Assistant turns")
    ax_c.yaxis.set_major_formatter(FuncFormatter(_comma))

    # (d) messages per conversation, mean marked and labelled in place
    hist_counts, _, _ = ax_d.hist(
        total_msgs, bins=range(min(total_msgs), max(total_msgs) + 2), color=JSKY, edgecolor="white", linewidth=0.5
    )
    mean_msgs = float(np.mean(total_msgs))
    ax_d.set_ylim(0, hist_counts.max() * 1.14)
    ax_d.set_xlim(min(total_msgs) - 1, max(total_msgs) + 2)  # same breathing room as the bar panels
    ax_d.axvline(mean_msgs, color=JBLUE, linestyle="--", linewidth=1.1)
    ax_d.text(
        mean_msgs + 0.5, hist_counts.max() * 1.10, f"mean {mean_msgs:.2f}",
        ha="left", va="top", fontsize=7.5, color=JBLUE, fontweight="bold",
    )
    ax_d.set_xlabel("Messages")
    ax_d.set_ylabel("Conversations")
    ax_d.yaxis.set_major_formatter(FuncFormatter(_comma))

    for ax, tag in zip((ax_a, ax_b, ax_c, ax_d), ("(a)", "(b)", "(c)", "(d)")):
        _tag(ax, tag)
    _save(fig, "fig_shape_panel")


if __name__ == "__main__":
    main()
