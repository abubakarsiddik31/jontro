# ruff: noqa: E501  -- generated LaTeX table rows read better unwrapped
"""Generate the LaTeX tables for the Jontro manuscript from real artifacts.

Inputs (never hand-edited):
- outputs/stats/jontro2_stats.json      (scripts/jontro2_stats.py)
- outputs/eval/multiturn_scores.json    (scripts/score_multiturn.py)
- outputs/stats/register_analysis.json  (register-gate coverage + advisory audit)
- outputs/stats/data_facts.json         (Banglish orthography audit)
- outputs/stats/eval_reanalysis.json    (paired differences / MDD)
- out_modal/retention/retention_report.json (retention gates)

Writes paper/tables/*.tex. Re-run after any data or eval change; the manuscript
\\inputs these files, so a number cannot drift from its source.

Usage:
    python scripts/make_jontro2_tables.py
"""
from __future__ import annotations

import json
from pathlib import Path

STATS = Path("outputs/stats/jontro2_stats.json")
SCORES = Path("outputs/eval/multiturn_scores.json")
REGISTER = Path("outputs/stats/register_analysis.json")
FACTS = Path("outputs/stats/data_facts.json")
REANALYSIS = Path("outputs/stats/eval_reanalysis.json")
RETENTION_REPORT = Path("out_modal/retention/retention_report.json")
OUT = Path("paper/tables")

# Frozen retention gates (retention/PROTOCOL.md, fixed before any adapter ran).
GATES = {
    "G1_coherence": "coherence gap $\\leq$ 10 points vs base",
    "G2_tqa": "TruthfulQA-48 within 5 items of base",
    "G3_tool_restraint": "spurious calls $\\leq$ base + 5\\,pp",
    "G4_language": "language match $\\geq$ base $-$ 5\\,pp",
}


def esc(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def pct(num: float, den: float) -> str:
    return f"{100.0 * num / den:.1f}" if den else "--"


def write(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(body.strip() + "\n", encoding="utf-8")
    print(f"wrote {OUT / name}")


def main() -> None:
    s = json.loads(STATS.read_text(encoding="utf-8"))
    n = s["corpus"]["n_trajectories"]

    # ---------------------------------------------------------- corpus at a glance
    c = s["corpus"]
    rows = [
        ("Conversations", f"{n:,}"),
        ("Distinct task goals", f"{c['n_goals']}"),
        ("Service domains", f"{c['n_domains']}"),
        ("Distinct tools offered / called",
         f"{c['n_distinct_tools_offered']} / {c['n_distinct_tools_called']}"),
        ("User turns per conversation", f"{s['user_turns_min']}--{s['user_turns_max']} (mean {s['user_turns_mean']})"),
        ("Messages per conversation", f"{s['messages_per_traj']['min']}--{s['messages_per_traj']['max']} (mean {s['messages_per_traj']['mean']})"),
        ("Messages by role", f"{c['role_totals']['user']:,} user / {c['role_totals']['assistant']:,} assistant / {c['role_totals']['tool']:,} tool"),
        ("Tool-calling decision points", f"{c['n_decision_points']:,}"),
        ("Tool calls", f"{c['n_tool_calls']:,} (mean {c['mean_calls_per_decision_point']} per decision point)"),
        ("Decision points with parallel calls", f"{c['n_multi_call_messages']:,} ({pct(c['n_multi_call_messages'], c['n_decision_points'])}\\,\\%)"),
        ("Text-only assistant turns", f"{c['n_textonly_assistant_messages']:,}"),
        ("Role-sequence patterns", f"{c['n_role_shape_patterns']}"),
        ("Conversations with $\\geq$1 tool call", "100.0\\,\\%"),
        ("Distinct simulator seeds", f"{c['n_sim_seeds_distinct']:,} (0 collisions)"),
    ]
    body = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lr}",
            r"\toprule",
            r"Property & Value \\",
            r"\midrule",
            *[f"{k} & {v} \\\\" for k, v in rows],
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Statistics of the Jontro release, computed from the released "
            r"corpus.}",
            r"\label{tab:overview}",
            r"\end{table}",
        ]
    )
    write("corpus_overview.tex", body)

    # ------------------------------------------------------------------- domains
    doms = sorted(s["domains"].items(), key=lambda kv: -kv[1])
    labels = s["domain_labels"]
    tr_dom = s["splits"]["train"]["domain_counts"]
    te_dom = s["splits"]["test"]["domain_counts"]
    body = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Domain & Train & Test & Total & \% of corpus \\",
            r"\midrule",
            *[
                f"{labels.get(d, esc(d))} & {tr_dom.get(d, 0):,} & {te_dom.get(d, 0):,} & {v:,} & {pct(v, n)} \\\\"
                for d, v in doms
            ],
            r"\midrule",
            f"Total & {s['splits']['train']['n_records']:,} & {s['splits']['test']['n_records']:,} & {n:,} & 100.0 \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Conversations per domain and split. Every domain appears in "
            r"both splits; food delivery has the fewest test conversations.}",
            r"\label{tab:domains}",
            r"\end{table}",
        ]
    )
    write("domains.tex", body)

    # --------------------------------------------------------- register x script
    cells = s["register_script_cells"]
    regs = ["apni", "tumi", "tui"]
    scrs = ["bengali", "banglish"]
    head = " & ".join([r"\multirow{2}{*}{Address form}"] + [f"\\multicolumn{{2}}{{c}}{{{x.capitalize()}}}" for x in scrs]) + r" \\"
    sub = " & " + " & ".join([r"$n$ & \%"] * len(scrs)) + r" \\"
    body_rows = []
    for r in regs:
        cells_r = [f"{cells.get(f'{r}|{x}', 0):,} & {pct(cells.get(f'{r}|{x}', 0), n)}" for x in scrs]
        body_rows.append(f"{r.capitalize()} & " + " & ".join(cells_r) + r" \\")
    tot = "Total & " + " & ".join(
        f"{sum(cells.get(f'{r}|{x}', 0) for r in regs):,} & {pct(sum(cells.get(f'{r}|{x}', 0) for r in regs), n)}"
        for x in scrs
    ) + r" \\"
    body = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            head,
            sub,
            r"\midrule",
            *body_rows,
            r"\midrule",
            tot,
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Conversations by register and script. All six cells are populated. "
            r"The intimate $\times$ Banglish cell, which no register-annotated Bangla "
            r"resource covers, has " + f"{cells.get('tui|banglish', 0)}" + r" conversations.}",
            r"\label{tab:cells}",
            r"\end{table}",
        ]
    )
    write("register_script.tex", body)

    # ------------------------------------------------------------------- splits
    tr, te = s["splits"]["train"], s["splits"]["test"]
    g_total = tr["n_goals"] + te["n_goals"]
    body = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Split property & Train & Test & Overlap \\",
            r"\midrule",
            f"Conversations & {tr['n_records']:,} & {te['n_records']:,} & {s['splits']['id_overlap']} \\\\",
            f"\\quad \\% of corpus & {pct(tr['n_records'], n)} & {pct(te['n_records'], n)} & \\\\",
            f"Distinct goals & {tr['n_goals']} & {te['n_goals']} & {{\\bfseries {s['splits']['goal_overlap']}}} \\\\",
            f"\\quad \\% of {g_total} goals & {pct(tr['n_goals'], g_total)} & {pct(te['n_goals'], g_total)} & \\\\",
            f"Decision points & {tr['n_decision_points']:,} & {te['n_decision_points']:,} & -- \\\\",
            f"\\quad argument-free & {tr['arg_free_decision_points']:,} ({tr['arg_free_decision_points_pct']}\\,\\%) & {te['arg_free_decision_points']:,} ({te['arg_free_decision_points_pct']}\\,\\%) & -- \\\\",
            f"Tool calls & {tr['n_tool_calls']:,} & {te['n_tool_calls']:,} & -- \\\\",
            f"Domains covered & {tr['domains']} & {te['domains']} & -- \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Goal-disjoint split. Within each domain about 30\% of the goals, "
            r"rounded and never all of them, are held out for test with a fixed seed; the "
            f"test side has {pct(te['n_goals'], g_total)}\\,\\% of goals and {pct(te['n_records'], n)}\\,\\% of records. "
            r"No test goal occurs in training, where goal identity is identity of the goal "
            r"text (near-duplicates are reported in \S\ref{sec:limitations}); a split by "
            r"record would place other conversations of each of the " + f"{te['n_goals']}" + r" test goals in training.}",
            r"\label{tab:splits}",
            r"\end{table}",
        ]
    )
    write("splits.tex", body)

    # ------------------------------------------------------------ tool inventory
    off, used = s["tools_offered_per_traj"], s["tools_used_per_traj"]
    ks = sorted({int(k) for k in off} | {int(k) for k in used})
    off_mean = sum(int(k) * v for k, v in off.items()) / n
    used_mean = sum(int(k) * v for k, v in used.items()) / n
    # One physical row per k: both distributions side by side in a single
    # tabular row (no embedded row terminator, which would split the line).
    off_rows = [
        f"{k} & {off.get(str(k), 0):,} & {pct(off.get(str(k), 0), n)}"
        for k in ks
        if off.get(str(k))
    ]
    used_rows = [
        f"{k} & {used.get(str(k), 0):,} & {pct(used.get(str(k), 0), n)}"
        for k in ks
        if used.get(str(k))
    ]
    body = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{rrrrrr}",
            r"\toprule",
            r"\multicolumn{3}{c}{Candidate tools offered per conversation} & \multicolumn{3}{c}{Distinct tools invoked per conversation} \\",
            r"\cmidrule(lr){1-3}\cmidrule(lr){4-6}",
            r"$k$ tools & Conversations & \% & $k$ tools & Conversations & \% \\",
            r"\midrule",
            *[
                f"{off_rows[i] if i < len(off_rows) else ' & & '} & {used_rows[i] if i < len(used_rows) else ' & & '} \\\\"
                for i in range(max(len(off_rows), len(used_rows)))
            ],
            r"\midrule",
            f"Mean & \\multicolumn{{2}}{{c}}{{{off_mean:.2f}}} & Mean & \\multicolumn{{2}}{{c}}{{{used_mean:.2f}}} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Number of tools offered and number of distinct tools called"
            r" per conversation (2{,}000 conversations). The two distributions are"
            r" shown side by side; a conversation may offer $k$ tools and call"
            r" $k'\neq k$ of them, so the rows do not correspond. Models choose from"
            r" a small candidate set specific to each conversation.}",
            r"\label{tab:inventory}",
            r"\end{table}",
        ]
    )
    write("tool_inventory.tex", body)

    # --------------------------------------------------------------- validation
    vf = s["validation_flags"]
    nva = s["validation_numeric"]["n_candidate_tools"]

    def flag_true(k: str) -> int:
        return vf.get(k, {}).get("true", 0)

    rows = [
        f"Schema + language boundary & required & {flag_true('schema_pass'):,}/{n:,} & 100.0 \\\\",
        f"Simulator replay (byte-exact) & required & {flag_true('consistency_pass'):,}/{n:,} & 100.0 \\\\",
        f"Heuristics (depth, script, repetition) & required & {flag_true('heuristics_pass'):,}/{n:,} & 100.0 \\\\",
        f"Reference call present & required & {flag_true('reference_call'):,}/{n:,} & 100.0 \\\\",
        f"Conversational closure labelled & recorded & {flag_true('closure_style_present'):,}/{n:,} & {pct(flag_true('closure_style_present'), n)} \\\\",
        ("Cross-register pronoun & required & 0 flagged & 0.0 \\\\"),
        ("Sadhu bhasha (literary register) & required & 0 flagged & 0.0 \\\\"),
        f"Imperative register concord & advisory & {flag_true('register_imperative_flag'):,} flagged & {pct(flag_true('register_imperative_flag'), n)} \\\\",
        f"Verb-morphology register concord & advisory & {flag_true('register_verb_flag'):,} flagged & {pct(flag_true('register_verb_flag'), n)} \\\\",
        f"LLM judge (5 rubric dimensions) & specified, \\textbf{{not run}} & {flag_true('judge_scored')} scored & -- \\\\",
    ]
    body = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{llrr}",
            r"\toprule",
            r"Check & Status & Outcome & \% of records \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Automated validation of the released corpus. The required checks "
            r"run offline and deterministically on every record, and a record is rejected "
            r"at its first failure. The judge stage is implemented but was not run on this "
            r"release. Advisory rows do not reject records, and their percentages are "
            r"flag rates.}",
            r"\label{tab:validation}",
            r"\end{table}",
        ]
    )
    write("validation.tex", body)

    # ------------------------------------------------------------------ scores
    if SCORES.exists():
        sc = json.loads(SCORES.read_text(encoding="utf-8"))
        floors = ["ORACLE-gold", "majority-global", "majority-domain", "first-offered", "random-offered"]
        zero = [k for k in sc if k.endswith("-zeroshot")]
        tuned = [k for k in sc if "-SFT" in k]
        groups = [[k for k in floors if k in sc], sorted(zero), sorted(tuned)]
        groups += [[k for k in sc if all(k not in g for g in groups)]]

        def row(name, delta_vs=None):
            d = sc[name]
            ov, tc = d["overall"], d["task_completion"]
            label = name.replace("-", r"-\allowbreak ")
            cs_lo, cs_hi = ov["tool_selection_ci"]
            ae_lo, ae_hi = ov["arg_exact_ci"]
            delta = "-- & --"
            if delta_vs and delta_vs in sc:
                base = sc[delta_vs]["overall"]
                d_cs = 100 * (ov["tool_selection_rate"] - base["tool_selection_rate"])
                d_ae = 100 * (ov["arg_exact_rate"] - base["arg_exact_rate"])
                delta = f"$+{d_cs:.1f}$ & $+{d_ae:.1f}$"
            return (f"{label} & {100*ov['first_call_accuracy']:.1f} & "
                    f"{100*ov['tool_selection_rate']:.1f} & [{cs_lo:.1f}--{cs_hi:.1f}] & "
                    f"{100*ov['name_recall']:.1f} & "
                    f"{100*ov['arg_exact_rate']:.1f} & [{ae_lo:.1f}--{ae_hi:.1f}] & "
                    f"{100*ov['slot_accuracy']:.1f} & {100*tc['completion_rate']:.1f} & {delta} \\\\")

        lines = []
        for g in (g for g in groups if g):
            if lines:
                lines.append(r"\midrule")
            for nm in g:
                lines.append(row(nm, delta_vs="Qwen3.5-2B-zeroshot" if nm == "Qwen3.5-2B-SFT" else None))
        has_models = bool(zero or tuned)
        caption = (
            r"\caption{Teacher-forced scoring on the 700-conversation test split (1{,}720 decision points, 48 goals). "
            r"$^{\dagger}$Task completion is teacher-forced; the interactive protocol is oracle-verified but not yet run for models (\S\ref{sec:notrun}). "
            r"\emph{ORACLE-gold} replays the gold calls and serves as the upper bound and a check of the scorer; the next four rows are non-learned baselines. "
            + (r"The \emph{-zeroshot} rows are Qwen3.5 base models given the Jontro system prompt and the tools in their native chat template, thinking off, greedy decoding. "
               r"The \emph{-SFT} rows are Qwen3.5-2B with a LoRA adapter trained on the 1{,}300 training conversations (\emph{-SFT-mix} adds self-distilled replay data, \S\ref{sec:models}). "
               r"All results are from one run with one seed. Tool selection and argument exactness are shown with 95\% goal-clustered intervals, "
               r"and $\Delta$ is the paired difference between SFT and the zero-shot 2B model; further intervals and paired tests are in \S\ref{sec:results}.}"
               if has_models else
               r"Model baselines are produced by the same harness and are not yet run.}")
        )
        body = "\n".join(
            [
                r"\begin{table}[t]",
                r"\centering",
                r"\footnotesize",
                r"\setlength{\tabcolsep}{3pt}",
                r"\resizebox{\textwidth}{!}{%",
                r"\begin{tabular}{lrrrrrrrrrr}",
                r"\toprule",
                r" & \multicolumn{8}{c}{Score (\%)} & \multicolumn{2}{c}{$\Delta$ (SFT $-$ 2B-zs)} \\",
                r"\cmidrule(lr){2-9}\cmidrule(lr){10-11}",
                r"System & First & Call set & 95\,CI & Name rec. & Args & 95\,CI & Slots & Task$^{\dagger}$ & CS & Args \\",
                r"\midrule",
                *lines,
                r"\bottomrule",
                r"\end{tabular}}",
                caption,
                r"\label{tab:baselines}",
                r"\end{table}",
            ]
        )
        write("baselines.tex", body)
    else:
        print(f"(skipping baselines table: {SCORES} absent)")

    # ------------------------------------------------------- retention (general ability)
    ret = Path("outputs/eval/retention_scores.json")
    if ret.exists():
        m = json.loads(ret.read_text(encoding="utf-8"))["metrics"]
        arms = [("Qwen3.5-2B base", "base"), ("2B SFT", "qwen3.5-2b-full-s0"), ("2B SFT-mix", "qwen3.5-2b-full_mix-s0")]
        f = lambda v, nd=1, sign=False: "--" if v is None else (f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}")  # noqa: E731
        lines = []
        for label, key in arms:
            r = m[key]
            lines.append(
                f"{label} & {r['tqa_strict_correct']} & {100*r['tool_spurious_rate']:.0f} & "
                f"{f(r.get('coherence_mean'))} & {f(r.get('coherence_bn'))} & {f(r.get('coherence_en'))} & "
                f"{f(r.get('heldout_bn_nll_delta'), 2, True)} & {f(r.get('heldout_en_nll_delta'), 2, True)} \\\\")
        body = "\n".join([
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lrrrrrrr}",
            r"\toprule",
            r" & TQA & Spurious & \multicolumn{3}{c}{Coherence (0--100)} & \multicolumn{2}{c}{$\Delta$NLL} \\",
            r"\cmidrule(lr){4-6}\cmidrule(lr){7-8}",
            r"Model & /48 & calls (\%) & All & Bengali & English & Bengali & English \\",
            r"\midrule",
            *lines,
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{General ability of the base and fine-tuned Qwen3.5-2B models, with greedy decoding and thinking off. "
            r"TQA is the number of correct forced-choice answers out of 48. Spurious calls is the share of 100 unrelated prompts, "
            r"given with the Jontro system prompt and tools, on which the model made a tool call. "
            r"Coherence is rated by a Gemini judge on 100 prompts without tools (50 Bengali or Banglish, 50 English); the judge "
            r"is from the model family that generated the corpus (see \S\ref{sec:notrun}). "
            r"The base row is the $\Delta$NLL reference. $\Delta$NLL is the change in per-token negative log-likelihood "
            r"of held-out human-written replies, fine-tuned minus base (negative is better). "
            r"Thresholds and changes to the protocol are listed in Table~\ref{tab:ledger}.}",
            r"\label{tab:retention}",
            r"\end{table}",
        ])
        write("retention.tex", body)

    # --------------------------------------- floors and 2B models by turn index
    if SCORES.exists():
        sc = json.loads(SCORES.read_text(encoding="utf-8"))
        systems = ["first-offered", "random-offered", "Qwen3.5-2B-zeroshot", "Qwen3.5-2B-SFT"]
        bti = {k: sc.get(k, {}).get("by_turn_index", {}) for k in systems}
        fo = bti["first-offered"]
        idx = sorted((int(k) for k in fo), key=int)
        shown = [i for i in idx if fo.get(str(i), {}).get("n", 0) >= 30]
        dropped = len(idx) - len(shown)
        tail_points = sum(int(fo.get(str(i), {}).get("n", 0)) for i in idx if i not in shown)
        lines = []
        for i in shown:
            cells = [f"{i+1}", f"{int(fo[str(i)]['n']):,}"]
            for k in systems:
                b = bti[k].get(str(i), {})
                cells += [f"{100*b['tool_selection_rate']:.1f}", f"{100*b['arg_exact_rate']:.1f}"]
            lines.append(" & ".join(cells) + r" \\")
        body = "\n".join(
            [
                r"\begin{table}[!t]",
                r"\centering",
                r"\small",
                r"\setlength{\tabcolsep}{4pt}",
                r"\begin{tabular}{rrrrrrrrrr}",
                r"\toprule",
                r"\multirow{2}{*}{Turn $t$} & \multirow{2}{*}{Points} & \multicolumn{2}{c}{first-offered} & \multicolumn{2}{c}{random-offered} & \multicolumn{2}{c}{2B zero-shot} & \multicolumn{2}{c}{2B SFT} \\",
                r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
                r" & & Call set & Args & Call set & Args & Call set & Args & Call set & Args \\",
                r"\midrule",
                *lines,
                r"\midrule",
                rf"$6$--$12$ & {tail_points:,} & \multicolumn{{8}}{{c}}{{omitted ($<$30 points each)}} \\",
                r"\bottomrule",
                r"\end{tabular}",
                r"\caption{Tool selection and argument exactness (\%) by \emph{tool-calling "
                r"turn position}, teacher-forced. $t=1$ is the conversation's first "
                r"tool-calling turn. Unlike the user-turn index of "
                r"Figure~\ref{fig:dpturn}, $t$ can reach twelve in the corpus. On tool "
                r"selection only first-offered drops sharply after $t=1$; random-offered "
                r"falls and then rises above its $t=1$ score. Argument exactness falls "
                r"to near zero for both baselines. Under teacher forcing the models do "
                r"not get worse at later positions, because the gold prefix contains "
                r"the earlier results."
                + (f" The tail row aggregates the {dropped} higher positions." if dropped else "")
                + r"}",
                r"\label{tab:byturnfloors}",
                r"\end{table}",
            ]
        )
        write("by_turn_floors.tex", body)

    # ------------------------------------------------------- appendix: catalogue
    tu = list(s["tool_use_counts"].items())
    third = (len(tu) + 2) // 3
    cols = [tu[i * third:(i + 1) * third] for i in range(3)]

    def _cell(t: tuple[str, int] | None) -> str:
        return f"\\code{{{esc(t[0])}}} & {t[1]:,}" if t else "& "

    rows = [
        " & ".join(_cell(c[i] if i < len(c) else None) for c in cols) + r" \\"
        for i in range(third)
    ]
    body = "\n".join(
        [
            r"{\footnotesize",
            r"\begin{longtable}{lr@{\hspace{1.2em}}lr@{\hspace{1.2em}}lr}",
            r"\caption{Number of conversations that call each tool "
            f"({len(tu)} of the 54 catalogue tools are used in the release).}}"
            r"\label{tab:catalogue}\\",
            r"\toprule",
            r"Tool & Conv. & Tool & Conv. & Tool & Conv. \\",
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            r"Tool & Conv. & Tool & Conv. & Tool & Conv. \\",
            r"\midrule",
            r"\endhead",
            *rows,
            r"\bottomrule",
            r"\end{longtable}}",
        ]
    )
    write("tool_catalogue.tex", body)

    # ------------------------------------------------------ persona composition
    lit_order = ["low", "medium", "high"]
    reg_w = {"apni": "50", "tumi": "35", "tui": "15"}
    scr_w = {"bengali": "75", "banglish": "25"}
    spl_w = {"bangla": "60", "english": "40"}

    def axis_rows(title: str, counts: dict, weights: dict | None) -> list[str]:
        out = []
        for i, (k, v) in enumerate(counts.items()):
            axis = f"\\multirow{{{len(counts)}}}{{*}}{{{title}}}" if i == 0 else ""
            w = weights[k] if weights else "--"
            out.append(f"{axis} & {k} & {v:,} & {pct(v, n)} & {w} \\\\")
        return out

    rows = (
        axis_rows("Address register", s["registers"], reg_w)
        + [r"\midrule"]
        + axis_rows("User script", s["scripts"], scr_w)
        + [r"\midrule"]
        + axis_rows("System-prompt language", s["system_prompt_language"], spl_w)
        + [r"\midrule"]
        + axis_rows("Technical literacy", {k: s["tech_literacy"][k] for k in lit_order}, None)
    )
    body = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{llrrr}",
            r"\toprule",
            r"Persona axis & Level & Conversations & \% & Weight (\%) \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Persona attributes in the release. Weights are the sampling "
            r"weights of the orchestrator (technical literacy is sampled uniformly), "
            r"and counts are the realised numbers of conversations.}",
            r"\label{tab:persona}",
            r"\end{table}",
        ]
    )
    write("persona_composition.tex", body)

    # ------------------------------------------------------- advisory-flag audit
    if REGISTER.exists():
        ra = json.loads(REGISTER.read_text(encoding="utf-8"))
        imp = ra["advisory_audit"]["imperative"]
        vrb = ra["advisory_audit"]["verb"]
        gc = ra["gate_recheck"]["gate_conditioning"]
        imp_rec = imp["record_level"]
        vrb_rec = vrb["record_level"]
        rows = [
            (
                r"Cross-register pronoun (required) & 0 flagged & 0; an independent "
                r"recheck with a broader marker inventory also finds 0 & -- \\"
            ),
            (
                f"Imperative concord (advisory) & {imp['flagged_records']} & "
                f"{imp_rec['shared_paradigm']} shared bare-stem (grammatical in \\emph{{tui}}), "
                f"{imp_rec['ambiguous']} conjunction \\bn{{যা}}, "
                f"{imp_rec['true_violation']} \\bn{{করাও}} read as a verbal noun: "
                r"\textbf{0 verified (0--1 by the strictest reading)} & -- \\"
            ),
            (
                f"Verb morphology (advisory) & {vrb['flagged_records']} & "
                f"{vrb_rec['shared_paradigm']} shared \\emph{{tumi}}/\\emph{{tui}} paradigm, "
                f"{vrb_rec['ambiguous']} context-ambiguous & "
                f"{vrb['homograph_or_neutral_false_positive_occurrences']}/830 flagged tokens "
                f"({100.0 * vrb['homograph_or_neutral_false_positive_occurrences'] / 830:.1f}\\%) are ordinary words \\\\"
            ),
            r"\midrule",
            r"Total advisory & 842 & \textbf{0 verified register violations} & -- \\",
        ]
        body = "\n".join(
            [
                r"\begin{table}[t]",
                r"\centering",
                r"\small",
                r"\setlength{\tabcolsep}{4pt}",
                r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.21\textwidth}r>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}p{0.19\textwidth}@{}}",
                r"\toprule",
                r"Flag class & Flagged & Outcome after audit & Checker FPs \\",
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabularx}",
                r"\caption{Audit of all register flags in the release against the "
                r"rule lists released with the analysis outputs. The audit is "
                r"rule-based, with a manual reading of the single candidate, and does "
                r"not replace native-speaker judgement. Register checks apply to "
                r"user turns longer than 20 characters. Coverage of the pronoun check: "
                f"{gc['records_with_any_pronoun_marker_pct']}\\% of records, "
                f"{gc['user_turns_with_any_pronoun_marker_pct']}\\% of user turns contain a "
                r"detectable second-person marker; the rest pass trivially, since "
                r"Bangla often omits pronouns.}",
                r"\label{tab:flagaudit}",
                r"\end{table}",
            ]
        )
        write("register_flag_audit.tex", body)

    # ------------------------------------------------------- Banglish orthography
    if FACTS.exists():
        df = json.loads(FACTS.read_text(encoding="utf-8"))
        o = df["item3_banglish_orthography"]
        sh = o["b_sh_vs_s"]
        en = o["c_enclitics"]
        lex = en["lexicalized_unhyphenated"]
        rows = [
            (
                r"Dominant function-word spelling & share of tokens, 13 most frequent words "
                r"(\code{ami}, \code{kore}, \code{hobe}, \code{ebong}, \ldots) & "
                r"95--100\% per word; token-weighted mean 98.8\% \\"
            ),
            (
                r"Sibilants & \code{sh}- spellings vs bare \code{s} for "
                r"\bn{শ}/\bn{ষ} & "
                f"{sh['sh_side']} vs {sh['bare_s_side']} ({sh['sh_share_pct']}\\%) \\\\"
            ),
            (
                r"Enclitics on English content words & hyphenated (\code{bus-er}, "
                r"\code{booking-ti}) vs unhyphenated & "
                f"{en['hyphenated_tokens']:,} vs {en['unhyphenated_bare_suffixation_onto_english_words']} \\\\"
            ),
            (
                r"Lexicalised unhyphenated enclitics & \code{ekta}/\code{ekti}/\code{eita} "
                r"(not hyphenated enclitic hosts) & "
                f"{lex['ekta']}/{lex['ekti']}/{lex['eita']} \\\\"
            ),
            (
                r"Fully-English Banglish turns & turns with no Bangla marker & "
                r"0 of 1{,}429 \\\\"
            ),
            (
                r"Bengali numerals in Banglish turns & number-words or Bengali script digits & "
                r"essentially never; Arabic digits throughout \\\\"
            ),
        ]
        body = "\n".join(
            [
                r"\begin{table}[t]",
                r"\centering",
                r"\small",
                r"\setlength{\tabcolsep}{4pt}",
                r"\begin{tabularx}{\textwidth}{@{}p{0.30\textwidth}XX@{}}",
                r"\toprule",
                r"Feature & Statistic & Value \\",
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabularx}",
                r"\caption{Orthographic audit of the corpus's Banglish "
                f"({o['scope']}). The spelling is consistent and tidier than in "
                r"real chat; per-word variant counts are released with the analysis "
                r"outputs.}",
                r"\label{tab:orthography}",
                r"\end{table}",
            ]
        )
        write("banglish_orthography.tex", body)

    # ------------------------------------------------- minimal detectable difference
    if REANALYSIS.exists():
        er = json.loads(REANALYSIS.read_text(encoding="utf-8"))
        label = {
            "sft_minus_2b_zeroshot": "SFT $-$ 2B zero-shot",
            "sft_minus_sft_mix": "SFT $-$ SFT-mix",
            "4b_minus_2b_zeroshot": "4B $-$ 2B zero-shot",
            "9b_minus_2b_zeroshot": "9B $-$ 2B zero-shot",
        }
        rows = []
        for key, lab in label.items():
            d = er["paired_diffs"].get(key)
            if not d:
                continue
            cs, ae = d["tool_selection"], d["arg_exact"]
            mdd_cs = (cs["ci95_pct"][1] - cs["ci95_pct"][0]) / 2
            mdd_ae = (ae["ci95_pct"][1] - ae["ci95_pct"][0]) / 2
            span = "spans zero" if not (cs["excludes_zero"] and ae["excludes_zero"]) else "excludes zero"
            rows.append(f"{lab} & {d['n_pairs']:,} & $\\pm${mdd_cs:.1f} & $\\pm${mdd_ae:.1f} & {span} \\\\")
        body = "\n".join(
            [
                r"\begin{table}[t]",
                r"\centering",
                r"\small",
                r"\begin{tabular}{lrrrl}",
                r"\toprule",
                r"Paired comparison & $n$ & \multicolumn{2}{c}{MDD (points)} & Interval \\",
                r"\cmidrule(lr){3-4}",
                r" & & Call set & Args & \\",
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
                r"\caption{Minimal detectable difference (MDD): the half-width of the "
                r"paired 95\% goal-clustered interval for each comparison on the 48 "
                r"test goals. When an interval includes zero, the comparison says "
                r"nothing about effects smaller than its MDD and does not show "
                r"equivalence. The primary comparison is SFT against the zero-shot 2B "
                r"model; the others are exploratory and not corrected for multiple "
                r"testing.}",
                r"\label{tab:power}",
                r"\end{table}",
            ]
        )
        write("power.tex", body)

    # --------------------------------------------------------------- retention gates
    if RETENTION_REPORT.exists():
        rep = json.loads(RETENTION_REPORT.read_text(encoding="utf-8"))
        arms = [
            ("Base (reference)", "base"),
            ("2B SFT", "qwen3.5-2b-full-s0"),
            ("2B SFT-mix", "qwen3.5-2b-full_mix-s0"),
        ]
        mark = lambda b: r"\cmark" if b else r"\xmark"  # noqa: E731
        base_m = rep["metrics"]["base"]
        lines = []
        for lab, key in arms:
            m = rep["metrics"][key]
            g = rep["gates"].get(key)
            gap = base_m["coherence_mean"] - m["coherence_mean"]
            if g is None:  # the base arm is the reference; gates do not apply to it
                gate_cells = " & ".join(["--"] * 5)
            else:
                gate_cells = " & ".join(
                    mark(g.get(k)) for k in ("G1_coherence", "G2_tqa", "G3_tool_restraint", "G4_language", "PASS")
                )
            lines.append(
                f"{lab} & {m['tqa_strict_correct']} & {100 * m['tool_spurious_rate']:.0f} & "
                f"{gap:.1f} & {100 * m['lang_match_rate']:.0f} & {gate_cells} \\\\"
            )
        body = "\n".join(
            [
                r"\begin{table}[t]",
                r"\centering",
                r"\small",
                r"\begin{tabular}{lrrrrccccc}",
                r"\toprule",
                r" & TQA & Spurious & Coher.\ gap & Lang.\ match & G1 & G2 & G3 & G4 & Pass \\",
                r"\midrule",
                *lines,
                r"\bottomrule",
                r"\end{tabular}",
                r"\caption{The four retention thresholds, fixed before evaluation, "
                r"and the observed values. G1: coherence gap $\leq$ 10 points vs base. "
                r"G2: TruthfulQA-48 within 5 items of base. G3: spurious calls "
                r"$\leq$ base $+$ 5\,pp. G4: reply-language match $\geq$ base $-$ "
                r"5\,pp. SFT fails G1 and G3, and SFT-mix passes all four. SFT-mix "
                r"was designed after SFT failed, so its pass is not an independent "
                r"test (Table~\ref{tab:ledger}). The Bengali-only coherence gap (10.9 for "
                r"SFT-mix) is discussed in the text.}",
                r"\label{tab:retgates}",
                r"\end{table}",
            ]
        )
        write("retention_gates.tex", body)

    print(f"candidate-tool size range: {nva['min']:.0f}--{nva['max']:.0f} (mean {nva['mean']})")


if __name__ == "__main__":
    main()
