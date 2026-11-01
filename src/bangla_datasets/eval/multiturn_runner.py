"""Run a model over the multi-turn test split in one of two protocols.

- ``teacher_forced``: one prediction per (trajectory, decision point). The model
  sees the **gold** prefix at each step, so errors do not compound; this is the
  per-step decision-quality measurement and the source of the per-turn-index
  curve. Resumable by ``(example_id, step)``.
- ``interactive``: one prediction per trajectory. The model drives every step;
  each predicted call is executed by the deterministic simulator and the result
  is fed back before the next gold user turn, so errors compound and the result
  is an end-to-end task-completion measurement. Resumable by ``example_id``.

Both support ``oracle=True``, which skips the client and echoes the gold calls.
That yields the 100% upper bound and is the offline self-check that the runner
and scorer agree — no API budget needed.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from bangla_datasets.eval.multiturn_task import (
    build_step_messages,
    build_step_tools,
    iter_decision_points,
    message_to_openai,
    openai_call,
    simulate_seed,
)
from bangla_datasets.eval.runner import DEFAULT_MIN_DELAY, RunnerReport
from bangla_datasets.schema import ToolCall, Trajectory
from bangla_datasets.tools.simulators import simulate

if TYPE_CHECKING:
    from bangla_datasets.eval.client import Prediction

_log = logging.getLogger("bangla_datasets.eval")


class _ClientProto(Protocol):
    model: str

    def predict_messages(
        self, messages: list[dict], tools: list[dict] | None = None, tag: str = ""
    ) -> Prediction: ...


def _model_name(client: _ClientProto | None, oracle: bool) -> str:
    if oracle:
        return "ORACLE-gold"
    return getattr(client, "model", "unknown")


def _done_keys(path: Path, key: str = "example_id") -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = r.get(key, "")
            if key == "example_id" and r.get("step") is not None:
                k = f"{k}#{r['step']}"
            out.add(k)
    return out


def run_teacher_forced(
    client: _ClientProto | None,
    trajectories: list[Trajectory],
    out_path: Path,
    min_delay: float = DEFAULT_MIN_DELAY,
    oracle: bool = False,
) -> RunnerReport:
    """Predict every decision point, given the gold prefix at each step."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys(out_path)
    model = _model_name(client, oracle)
    n = n_errors = 0
    with out_path.open("a", encoding="utf-8") as f:
        for t in trajectories:
            for dp in iter_decision_points(t):
                key = f"{t.id}#{dp.index}"
                if key in done:
                    n += 1
                    continue
                if oracle:
                    calls, raw, err = dp.gold_calls, "", None
                else:
                    if n > 0 and min_delay > 0:
                        time.sleep(min_delay)
                    pred = client.predict_messages(  # type: ignore[union-attr]
                        build_step_messages(t, dp), build_step_tools(t), tag=key
                    )
                    calls, raw, err = pred.tool_calls, pred.raw_text, pred.error
                f.write(
                    json.dumps(
                        {
                            "example_id": t.id,
                            "step": dp.index,
                            "protocol": "teacher_forced",
                            "model": model,
                            "tool_calls": [c.model_dump(mode="json") for c in calls],
                            "raw_text": raw,
                            "error": err,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                f.flush()
                n += 1
                if err:
                    n_errors += 1
                if n % 50 == 0:
                    _log.info("%s: %d steps (errors=%d)", model, n, n_errors)
    return RunnerReport(model=model, n=n, n_errors=n_errors, out_path=out_path)


def run_interactive(
    client: _ClientProto | None,
    trajectories: list[Trajectory],
    out_path: Path,
    min_delay: float = DEFAULT_MIN_DELAY,
    oracle: bool = False,
) -> RunnerReport:
    """Drive each trajectory end to end, feeding back simulated tool results."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys(out_path, key="example_id")
    model = _model_name(client, oracle)
    n = n_errors = 0
    with out_path.open("a", encoding="utf-8") as f:
        for t in trajectories:
            if t.id in done:
                n += 1
                continue
            tools = build_step_tools(t)
            messages: list[dict] = [{"role": "system", "content": t.system_prompt}]
            steps_out: list[dict] = []
            row_err: str | None = None
            for dp in iter_decision_points(t):
                for m in dp.pending_user:
                    messages.append(message_to_openai(m))
                if oracle:
                    calls: list[ToolCall] = dp.gold_calls
                    raw, perr = "", None
                else:
                    if n > 0 and min_delay > 0:
                        time.sleep(min_delay)
                    pred = client.predict_messages(  # type: ignore[union-attr]
                        messages, tools, tag=f"{t.id}#{dp.index}"
                    )
                    calls, raw, perr = pred.tool_calls, pred.raw_text, pred.error
                    row_err = row_err or perr
                amsg: dict = {"role": "assistant", "content": raw or ""}
                if calls:
                    amsg["tool_calls"] = [openai_call(c, i) for i, c in enumerate(calls)]
                messages.append(amsg)
                results: list[str] = []
                for i, c in enumerate(calls):
                    call_id = openai_call(c, i)["id"]
                    try:
                        res = simulate(c.name, c.arguments, simulate_seed(t.id, dp.index, i))
                    except Exception as exc:  # noqa: BLE001 - unknown tool -> error payload
                        res = json.dumps({"error": str(exc)})
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": res})
                    results.append(res)
                steps_out.append(
                    {
                        "step": dp.index,
                        "tool_calls": [c.model_dump(mode="json") for c in calls],
                        "tool_results": results,
                        "error": perr,
                    }
                )
            f.write(
                json.dumps(
                    {
                        "example_id": t.id,
                        "protocol": "interactive",
                        "model": model,
                        "steps": steps_out,
                        "error": row_err,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()
            n += 1
            if row_err:
                n_errors += 1
            if n % 20 == 0:
                _log.info("%s: %d trajs (errors=%d)", model, n, n_errors)
    return RunnerReport(model=model, n=n, n_errors=n_errors, out_path=out_path)
