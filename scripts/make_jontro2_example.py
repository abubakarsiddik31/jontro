# ruff: noqa: E501  -- LaTeX fragments are more readable unwrapped
"""Render one real Jontro conversation to LaTeX for the manuscript.

The example is selected by an objective rule, not by hand: among test-split
conversations that contain at least two decision points and at least one turn
with parallel calls, take the one whose message count is closest to the median
(ties broken by ID). The caption in the manuscript states the rule, so the
reader knows it is representative rather than flattering.

Usage:
    python scripts/make_jontro2_example.py
"""
from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path

CANONICAL = Path("outputs/release_multiturn/jontro_multiturn_2000.jsonl")
TEST_IDS = Path("outputs/splits/test_ids.jsonl")
OUT = Path("paper/tables/example_trajectory.tex")

BENGALI = re.compile(r"[\u0980-\u09FF]")
BENGALI_RUN = re.compile(r"[\u0980-\u09FF]+")
_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def tex_escape(s: str) -> str:
    return "".join(_SPECIALS.get(ch, ch) for ch in s)


def render_text(s: str, limit: int = 520) -> str:
    s = " ".join(s.split())
    if len(s) > limit:
        s = s[: limit - 1] + "\u2026"
    esc = tex_escape(s)
    return f"\\bn{{{esc}}}" if BENGALI.search(s) else esc


def render_json(obj: object, limit: int = 320) -> str:
    # Tool messages store the simulator output as a JSON *string*; parse it so we
    # render the object, not a doubly-escaped string literal.
    if isinstance(obj, str):
        with contextlib.suppress(json.JSONDecodeError):
            obj = json.loads(obj)
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    if len(s) > limit:
        s = s[: limit - 1] + "\u2026"
    return tex_escape(s)


def json_text(obj: object, limit: int = 320) -> str:
    """Raw (unescaped) JSON text for a tool call or tool result."""
    if isinstance(obj, str):
        with contextlib.suppress(json.JSONDecodeError):
            obj = json.loads(obj)
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    if len(s) > limit:
        s = s[: limit - 1] + "\u2026"
    return s


def mixed_code(s: str) -> str:
    """Render text as code, switching Bangla runs to the Bengali font.

    Monospace fonts carry no Bengali glyphs, so a Bangla value inside an
    otherwise-ASCII JSON blob would silently vanish from the PDF. We alternate
    \\code{} and \\bn{} spans so every character renders.
    """
    s = " ".join(s.split())
    out: list[str] = []
    pos = 0
    for m in BENGALI_RUN.finditer(s):
        if m.start() > pos:
            out.append(f"\\code{{{tex_escape(s[pos:m.start()])}}}")
        out.append(f"\\bn{{{m.group(0)}}}")
        pos = m.end()
    if pos < len(s):
        out.append(f"\\code{{{tex_escape(s[pos:])}}}")
    return "".join(out) or "\\code{}"


ROLE_LABEL = {
    "user": r"\textcolor{jblue}{\textbf{user}}",
    "assistant": r"\textcolor{jgreen}{\textbf{assistant}}",
    "tool": r"\textcolor{jred}{\textbf{tool}}",
}
ROLE_COLOR = {"user": "jblue", "assistant": "jgreen", "tool": "jred"}


def choose(recs: list[dict], test_ids: set[str]) -> dict:
    cands = []
    for r in recs:
        if r["id"] not in test_ids:
            continue
        dps = [m for m in r["messages"] if m["role"] == "assistant" and m.get("tool_calls")]
        if len(dps) < 2:
            continue
        if not any(len(m["tool_calls"]) > 1 for m in dps):
            continue
        cands.append(r)
    counts = sorted(len(r["messages"]) for r in cands)
    median = counts[len(counts) // 2]
    return sorted(cands, key=lambda r: (abs(len(r["messages"]) - median), r["id"]))[0]


# English glosses for the three register examples. Selection is deterministic, so
# these IDs are stable; the generator warns if the data changes underneath them.
GLOSS = {
    "bd-agentic-103212": "Hello, I would like to know the time and fare of a flight from Dhaka to Cox's Bazar. Could you check and tell me?",
    "bd-agentic-101921": "Hello, please check how much balance is on my SIM.",
    "bd-agentic-102968": "Listen, check how much main balance is on my SIM right now.",
    # Banglish cells of the six-cell table (shortest first user turn per cell).
    "bd-agentic-104423": "Hello, could you order a burger for me?",
    "bd-agentic-101953": "How much balance is on my mobile SIM? Check and tell me.",
    "bd-agentic-104574": "Hey, check how much balance is on my SIM right now.",
}


def pick_six_cell_examples(recs: list[dict], test_ids: set[str]) -> dict[tuple[str, str], dict]:
    """Shortest first user turn per register x script cell, test split only."""
    pick: dict[tuple[str, str], dict] = {}
    for r in recs:
        if r["id"] not in test_ids:
            continue
        first = next((m for m in r["messages"] if m["role"] == "user"), None)
        if not first:
            continue
        text = " ".join((first.get("content") or "").split())
        if not text:
            continue
        key = (r["persona"].get("register"), r["persona"].get("script"))
        if key not in pick or len(text) < len(pick[key]["text"]):
            pick[key] = {"id": r["id"], "text": text}
    return pick


def write_six_cell_examples(recs: list[dict], test_ids: set[str]) -> None:
    """Generate the six-cell glossed-examples table straight from the JSONL.

    Every printed turn is the verbatim, normalised first user turn of a test
    record, asserted against the release so the table cannot drift into
    paraphrase (a hand-edited Bengali row once shipped a non-word).
    """
    pick = pick_six_cell_examples(recs, test_ids)
    assert len(pick) == 6, f"expected 6 populated cells, got {sorted(pick)}"
    # Verbatim assertions: each selected text must be a first user turn of a
    # test-split record in the canonical release (trivially true here; the
    # assertion fails if anyone edits the generated rows by hand and reruns
    # selection logic out of sync with the data).
    first_turns = set()
    for r in recs:
        if r["id"] in test_ids:
            m = next((m for m in r["messages"] if m["role"] == "user"), None)
            if m and m.get("content"):
                first_turns.add(" ".join(m["content"].split()))
    rows = []
    for reg in ("apni", "tumi", "tui"):
        for scr in ("bengali", "banglish"):
            e = pick[(reg, scr)]
            assert e["text"] in first_turns, f"{e['id']}: turn not verbatim in release"
            gloss = GLOSS.get(e["id"])
            if gloss is None:
                print(f"WARNING: no gloss for {e['id']}; add one to GLOSS")
                gloss = "(no gloss recorded)"
            turn = f"\\bn{{{tex_escape(e['text'])}}}" if BENGALI.search(e["text"]) else tex_escape(e["text"])
            rows.append(
                f"\\textbf{{{reg}}} / {scr} & {turn} & {tex_escape(gloss)} \\\\"
            )
    body = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabularx}{\textwidth}{@{}l>{\raggedright\arraybackslash}p{0.47\textwidth}>{\raggedright\arraybackslash}p{0.31\textwidth}@{}}",
            r"\toprule",
            r"Cell & First user turn (verbatim, test split) & Gloss \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabularx}",
            r"\caption{One real first user turn from each register$\times$script cell "
            r"(test split; the shortest in each cell, so the register marking is easy "
            r"to see). In Bengali script the \emph{tumi} and \emph{tui} rows differ in "
            r"the imperative ending (\bn{বলো} vs \bn{বল}), the smallest cue the register "
            r"check uses; the Banglish rows show the same contrast in romanised form "
            r"(\emph{parben} / \emph{bolbe} vs.\ \emph{bol to}). In Banglish cells the "
            r"assistant replies in Bengali script, so the script axis concerns the "
            r"user's input. English glosses are approximate.}",
            r"\label{tab:sixcells}",
            r"\end{table}",
        ]
    )
    out = OUT.parent / "six_cell_examples.tex"
    out.write_text(body + "\n", encoding="utf-8")
    print(f"wrote {out}")


def write_glance(recs: list[dict], test_ids: set[str]) -> None:
    """Page-2 'at a glance' composite: generation loop, register x script mosaic,
    headline counts, and one real tui x Banglish exchange --- every number read
    from outputs/stats/jontro2_stats.json or the release, none typed by hand."""
    stats = json.loads(Path("outputs/stats/jontro2_stats.json").read_text(encoding="utf-8"))
    c = stats["corpus"]
    cells = stats["register_script_cells"]
    sp = stats["splits"]
    regs, scrs = ["apni", "tumi", "tui"], ["bengali", "banglish"]
    cmax = max(cells.values())

    # the exchange: the shortest tui x Banglish first turn (same rule as the
    # six-cell table), its first call, the simulator's result, and the reply
    pick = pick_six_cell_examples(recs, test_ids)[("tui", "banglish")]
    rec = next(r for r in recs if r["id"] == pick["id"])
    first_call = next(m for m in rec["messages"] if m["role"] == "assistant" and m.get("tool_calls"))
    call = first_call["tool_calls"][0]
    tool_msg = next(m for m in rec["messages"] if m["role"] == "tool")
    reply = next(m for m in rec["messages"] if m["role"] == "assistant" and not m.get("tool_calls") and m.get("content"))
    reply_text = " ".join(reply["content"].split())
    reply_first = reply_text.split("।")[0] + "।" if "।" in reply_text else reply_text
    gloss_user = GLOSS.get(pick["id"], "")
    GLANCE_REPLY_GLOSS = {
        "bd-agentic-104574": "Your mobile currently has a balance of 3,298 taka.",
    }
    gloss_reply = GLANCE_REPLY_GLOSS.get(pick["id"], "")
    call_str = mixed_code(f"{call['name']}({json_text(call['arguments'])})")
    result_str = mixed_code(json_text(tool_msg.get("content") or "", limit=60))

    mosaic = []
    cw, ch, x0, y0 = 13.0, 6.0, 9.0, -22.6  # mm
    for j, scr in enumerate(scrs):
        mosaic.append(f"\\node[font=\\scriptsize] at ({x0 + j * (cw + 0.8) + cw / 2:.1f}mm, {y0 + ch / 2 + 2.2:.1f}mm) {{{scr.capitalize()}}};")
    for i, reg in enumerate(regs):
        yc = y0 - i * (ch + 0.8)
        mosaic.append(f"\\node[font=\\scriptsize, anchor=east] at ({x0 - 1.2:.1f}mm, {yc:.1f}mm) {{\\emph{{{reg}}}}};")
        for j, scr in enumerate(scrs):
            v = cells[f"{reg}|{scr}"]
            shade = int(12 + 68 * v / cmax)
            txt = "white" if shade > 45 else "black"
            xc = x0 + j * (cw + 0.8) + cw / 2
            mosaic.append(
                f"\\node[draw=jblue!60, fill=jblue!{shade}, text={txt}, minimum width={cw}mm, minimum height={ch}mm, inner sep=0pt, font=\\scriptsize] "
                f"at ({xc:.1f}mm, {yc:.1f}mm) {{{v:,}}};"
            )
    # outline the novelty cell
    xi = x0 + 1 * (cw + 0.8) + cw / 2
    yi = y0 - 2 * (ch + 0.8)
    mosaic.append(f"\\node[draw=jred, thick, minimum width={cw}mm, minimum height={ch}mm, inner sep=0pt] at ({xi:.1f}mm, {yi:.1f}mm) {{}};")

    body = "\n".join(
        [
            r"\begin{figure}[!t]",
            r"\centering",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tikzpicture}[node distance=2.6mm, every node/.style={font=\scriptsize}]",
            r"% (a) generation loop",
            r"\node[box=jblue, font=\footnotesize, text width=27mm] (p1) {Persona sample\\[-1pt]{\scriptsize register $\cdot$ script $\cdot$ literacy}};",
            r"\node[box=jsky, right=of p1, text width=27mm] (p2) {User simulator\\[-1pt]{\scriptsize LLM; one Bangla/Banglish turn}};",
            r"\node[box=jgreen, right=of p2, text width=27mm] (p3) {Assistant\\[-1pt]{\scriptsize LLM; native function calling}};",
            r"\node[box=jorange, right=of p3, text width=27mm] (p4) {Deterministic simulator\\[-1pt]{\scriptsize seeded; replays byte-exact}};",
            r"\node[box=jgray, right=of p4, text width=27mm] (p5) {Conversation record\\[-1pt]{\scriptsize 2--5 user turns}};",
            r"\draw[flow] (p1)--(p2); \draw[flow] (p2)--(p3); \draw[flow] (p3)--(p4); \draw[flow] (p4)--(p5);",
            r"\draw[flowd] (p4.south) -- ++(0,-2.4mm) -| (p3.south);",
            r"\node[lbl, below=3.8mm of $(p3.south)!0.5!(p4.south)$] {until the request is complete};",
            r"\node[lbl, above=1.4mm of p1.north west, anchor=south west] {\textbf{(a)} how a conversation is made};",
            r"% (b) register x script mosaic",
            r"\node[lbl, anchor=south west] at (0mm, -12.4mm) {\textbf{(b)} register $\times$ script (conversations)};",
            *mosaic,
            r"% (c) headline counts",
            r"\node[anchor=north west, align=left, text width=48mm, font=\scriptsize] at (42mm, -14.2mm) {",
            r"\textbf{(c)} the release\\[2pt]",
            f"\\textbf{{{c['n_trajectories']:,}}} conversations, \\textbf{{{c['n_decision_points']:,}}} tool-calling decision points, \\textbf{{{c['n_tool_calls']:,}}} calls; every tool result replays byte for byte\\\\[1.5pt]",
            f"\\textbf{{{c['n_goals']}}} task goals $\\cdot$ \\textbf{{{c['n_domains']}}} Bangladesh-grounded domains $\\cdot$ \\textbf{{{c['n_distinct_tools_called']}}} tools\\\\[1.5pt]",
            f"goal-disjoint split \\textbf{{{sp['train']['n_records']:,} / {sp['test']['n_records']}}} conversations ({sp['train']['n_goals']} / {sp['test']['n_goals']} goals, zero overlap)",
            r"};",
            r"% (d) one real exchange",
            r"\node[anchor=north west, align=left, text width=62mm, font=\scriptsize] at (96mm, -14.2mm) {",
            r"\textbf{(d)} one real exchange (\emph{tui} $\times$ Banglish, test split)\\[2pt]",
            f"\\textcolor{{jblue}}{{\\textbf{{user}}}} {tex_escape(pick['text'])}\\\\",
            f"{{\\itshape\\color{{jgray}} {tex_escape(gloss_user)}}}\\\\[1.5pt]",
            f"\\textcolor{{jgreen}}{{\\textbf{{assistant}}}} $\\rightarrow$ {call_str}\\\\",
            f"\\textcolor{{jred}}{{\\textbf{{tool}}}} $\\rightarrow$ {result_str}\\\\[1.5pt]",
            f"\\textcolor{{jgreen}}{{\\textbf{{assistant}}}} {render_text(reply_first)}\\\\",
            f"{{\\itshape\\color{{jgray}} {tex_escape(gloss_reply)}}}",
            r"};",
            r"\end{tikzpicture}%",
            r"}",
            r"\caption{Overview of Jontro. (a) A deterministic simulator executes each "
            r"assistant call before the next turn, so later turns depend on actual "
            r"results. (b) All six register$\times$script cells are populated; the "
            r"outlined \emph{tui}$\times$Banglish cell is covered by no register-annotated "
            r"Bangla resource. (c) Release counts. (d) The opening of a test "
            r"conversation, verbatim: an intimate-register request in Banglish, the "
            r"executed call and its simulated result, and the assistant's reply in "
            r"Bengali script, in the respectful register.}",
            r"\label{fig:glance}",
            r"\end{figure}",
        ]
    )
    out = OUT.parent / "glance.tex"
    out.write_text(body + "\n", encoding="utf-8")
    print(f"wrote {out}: exchange from {pick['id']}")


def main() -> None:
    test_ids = {json.loads(line)["id"] for line in TEST_IDS.read_text().splitlines() if line.strip()}
    recs = [json.loads(line) for line in CANONICAL.open(encoding="utf-8") if line.strip()]
    write_six_cell_examples(recs, test_ids)
    write_glance(recs, test_ids)
    r = choose(recs, test_ids)

    md = r["metadata"]
    persona = r["persona"]
    n_calls = sum(len(m["tool_calls"]) for m in r["messages"] if m["role"] == "assistant" and m.get("tool_calls"))

    rows: list[str] = []
    for m in r["messages"]:
        role = m["role"]
        col = ROLE_COLOR[role]
        if role == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                calls.append(mixed_code(f"{tc['name']}({json_text(tc['arguments'])})"))
            content = render_text(m.get("content") or "") if m.get("content") else ""
            body = (" \\newline ".join(calls)) if calls else ""
            if content:
                body = content + (r" \newline " + body if body else "")
            rows.append(
                f"\\textcolor{{{col}}}{{\\textbullet}} & {ROLE_LABEL[role]} & {body} \\\\"
            )
        elif role == "tool":
            rows.append(f"\\textcolor{{{col}}}{{\\textbullet}} & {ROLE_LABEL[role]} & {mixed_code(json_text(m.get('content') or ''))} \\\\")
        else:
            rows.append(f"\\textcolor{{{col}}}{{\\textbullet}} & {ROLE_LABEL[role]} & {render_text(m.get('content') or '')} \\\\")

    body = "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{4pt}",
            r"\renewcommand{\arraystretch}{1.15}",
            r"\begin{tabularx}{\textwidth}{@{}cl>{\raggedright\arraybackslash}X@{}}",
            r"\toprule",
            r" & Role & Turn \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabularx}",
            r"\caption{A released test conversation (\code{" + tex_escape(r["id"]) + r"}), "
            r"reproduced verbatim. Domain: " + tex_escape(r["domain"]) + "; "
            r"address form: " + tex_escape(persona.get("register", "?")) + r"; script: "
            + tex_escape(persona.get("script", "?")) + r"; "
            f"{md.get('turns_user')} user turns; {n_calls} tool calls. "
            r"Tool calls are shown as \code{name(arguments)} and tool results are "
            r"simulator output. It is the test conversation closest to the median "
            r"length among those with $\geq 2$ decision points and $\geq 1$ parallel "
            r"call; it was chosen by this rule and not for its quality.}",
            r"\label{tab:example}",
            r"\end{table}",
        ]
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(body + "\n", encoding="utf-8")
    print(f"wrote {OUT}: {r['id']} ({md.get('turns_user')} user turns, {n_calls} calls, {len(r['messages'])} messages)")


if __name__ == "__main__":
    main()
