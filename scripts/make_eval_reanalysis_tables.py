"""Render the register-x-script and error-taxonomy tables for the manuscript
from outputs/stats/eval_reanalysis.json (produced by the eval re-analysis).

Writes paper/tables/per_cell_results.tex and paper/tables/error_taxonomy.tex.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs/stats/eval_reanalysis.json"
PAIRED = ROOT / "outputs/stats/cross_cell_paired.json"
OUT = ROOT / "paper/tables"

REG_LABEL = {
    "apni/bengali": r"apni / Bengali",
    "apni/banglish": r"apni / Banglish",
    "tumi/bengali": r"tumi / Bengali",
    "tumi/banglish": r"tumi / Banglish",
    "tui/bengali": r"tui / Bengali",
    "tui/banglish": r"tui / Banglish",
}
MODELS = ["Qwen3.5-2B-zeroshot", "Qwen3.5-2B-SFT"]
MODEL_LABEL = {
    "Qwen3.5-2B-zeroshot": r"2B zero-shot",
    "Qwen3.5-2B-SFT": r"2B SFT",
}
ERR_TYPES = [
    ("omission", "Omission"),
    ("wrong_tool_set", "Wrong tools"),
    ("set_composition", "Set composition"),
    ("right_tools_wrong_values", "Right tools, wrong values"),
    ("order_only", "Order only"),
    ("correct_strict", "Correct"),
]
ERR_SYSTEMS = [
    ("Qwen3.5-2B-zeroshot", r"2B zero-shot"),
    ("Qwen3.5-4B-zeroshot", r"4B zero-shot"),
    ("Qwen3.5-9B-zeroshot", r"9B zero-shot"),
    ("Qwen3.5-2B-SFT", r"2B SFT"),
    ("Qwen3.5-2B-SFT-mix", r"2B SFT-mix"),
]


def fmt(x: dict) -> str:
    lo, hi = x["ci95"]
    return f"{x['rate']:.1f} [{lo:.1f}--{hi:.1f}]"


def _num(x: float) -> str:
    return f"{x:.1f}".replace("-", "$-$")


def main() -> None:
    data = json.loads(SRC.read_text())
    paired = json.loads(PAIRED.read_text())
    span = paired["goals_by_cells_spanned"]
    multi = sum(v for k, v in span.items() if int(k) > 1)
    tb = paired["models"]["Qwen3.5-2B-SFT"]["arg_exact"]["tui/bengali"]
    OUT.mkdir(parents=True, exist_ok=True)

    # --- per-cell results -------------------------------------------------
    cells = data["cells"]["metrics"]
    test = data["cells"]["test_split"]
    lines = [
        r"\begin{table}[!t]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrccccr}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Test split} & \multicolumn{2}{c}{Tool selection (\%)} & \multicolumn{2}{c}{Argument exactness (\%)} & Completion (\%) \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Cell & conv.\ / goals & $n$ & 2B zero-shot & 2B SFT & 2B zero-shot & 2B SFT & 2B SFT \\",
        r"\midrule",
    ]
    comp = data.get("task_completion", {}).get("sft_per_cell", {})
    for cell in REG_LABEL:
        n = test[cell]["decision_points"]
        z = cells[cell]["Qwen3.5-2B-zeroshot"]
        s = cells[cell]["Qwen3.5-2B-SFT"]
        cc = comp.get(cell, {}).get("completion_rate_pct")
        comp_s = f"{cc:.1f}" if cc is not None else "--"
        lines.append(
            f"{REG_LABEL[cell]} & {test[cell]['conversations']} / {test[cell]['goals']} & {n} & "
            f"{fmt(z['tool_selection_rate'])} & {fmt(s['tool_selection_rate'])} & "
            f"{fmt(z['arg_exact_rate'])} & {fmt(s['arg_exact_rate'])} & {comp_s} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\caption{Decision-point results by register$\times$script cell on the"
        r" test split (1{,}720 points, 48 goals). Intervals are 95\%"
        r" goal-clustered bootstrap intervals (2{,}000 resamples). Each cell"
        r" resamples only its own goals, so the smallest cells (tui$\times$Banglish,"
        r" 82 points over 12 goals, and tumi$\times$Banglish, 8 goals) have wide"
        r" intervals, some reaching the bounds. Goals are shared across cells"
        f" ({multi} of the 48 test goals appear in two or three cells), so the"
        r" goal counts sum to more than 48. After fine-tuning, tool selection is"
        r" similar in all cells. Argument exactness is lowest in the"
        r" \emph{tui}$\times$\emph{Bengali} cell, but on the goals it shares with"
        f" other cells it is only {-tb['mean_diff']:.1f} points lower than those"
        f" goals' conversations elsewhere (95\\% interval {_num(tb['ci95'][0])} to {_num(tb['ci95'][1])},"
        r" no detectable difference). The overall gap"
        f" comes from goals that occur only in this cell ({tb['points_unshared_goals']}"
        f" points at {tb['rate_unshared_goals']:.1f}\\%; \\S\\ref{{sec:results}}).}}",
        r"\label{tab:percell}",
        r"\end{table}",
    ]
    (OUT / "per_cell_results.tex").write_text("\n".join(lines) + "\n")

    # --- error taxonomy ----------------------------------------------------
    errs = data["error_taxonomy"]["per_system"]
    lines = [
        r"\begin{table}[!t]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r" & \multicolumn{6}{c}{\% of the 1{,}720 test decision points} \\",
        r"\cmidrule(lr){2-7}",
        r"System & Omission & Wrong tools & Set compos. & Wrong values & Order only & Correct \\",
        r"\midrule",
    ]
    for system, label in ERR_SYSTEMS:
        e = errs[system]
        row = [label]
        for key, _ in ERR_TYPES:
            row.append(f"{e[key]['pct_of_points']:.1f}")
        lines.append(" & ".join(row) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Error types per system, teacher-forced. \emph{Omission}: no"
        r" call predicted. This includes parse failures, which the scorer"
        r" records as an empty call set, so decoding failures and deliberate"
        r" replies without a call are counted together. \emph{Wrong tools}: predicted names share no tool"
        r" with gold. \emph{Set composition}: partial overlap (missing, extra or"
        r" duplicated calls). \emph{Wrong values}: name multiset matches gold but"
        r" some argument value differs. \emph{Order only}: identical call set,"
        r" different order (penalised only by first-call accuracy). Rows sum to"
        r" 100. \emph{Correct} plus \emph{Order only} equals the set-based"
        r" argument exactness of Table~\ref{tab:baselines}, up to a rounding"
        r" difference of 0.1. After fine-tuning, most omissions become wrong"
        r" values or set-composition errors.}",
        r"\label{tab:errtax}",
        r"\end{table}",
    ]
    (OUT / "error_taxonomy.tex").write_text("\n".join(lines) + "\n")

    print("wrote paper/tables/per_cell_results.tex")
    print("wrote paper/tables/error_taxonomy.tex")


if __name__ == "__main__":
    main()
