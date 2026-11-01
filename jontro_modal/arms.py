"""Arm grammar shared by the retention and benchmark jobs.

    base                        the untouched base model
    <run>                       a run's final adapter           (/out/adapters/<run>)
    <run>@step_N                a training checkpoint           (/out/ckpt/<run>/step_N/adapter)
    <run>*0.5                   the adapter with its LoRA scale multiplied by 0.5 (interpolation toward base)
    <run>@step_N*0.5            both
"""
from __future__ import annotations


def parse_arm(arm: str) -> tuple[str, str | None, float]:
    alpha = 1.0
    if "*" in arm:
        arm, a = arm.split("*", 1)
        alpha = float(a)
    run, _, step = arm.partition("@")
    return run, (step or None), alpha


def label(arm: str) -> str:
    return arm.replace("@", "__").replace("*", "_x")


def adapter_dir(arm: str) -> str:
    run, step, _ = parse_arm(arm)
    return f"/out/ckpt/{run}/{step}/adapter" if step else f"/out/adapters/{run}"


def attach(base, arms: list[str]):
    """Load every non-base arm as its own PEFT adapter on one base model; return the PeftModel (or None)."""
    from peft import PeftModel

    pm = None
    for a in arms:
        if a == "base":
            continue
        name = label(a).replace(".", "_")  # PEFT adapter names cannot contain '.'
        if pm is None:
            pm = PeftModel.from_pretrained(base, adapter_dir(a), adapter_name=name)
        else:
            pm.load_adapter(adapter_dir(a), adapter_name=name)
        alpha = parse_arm(a)[2]
        if alpha != 1.0:
            for m in pm.modules():
                sc = getattr(m, "scaling", None)
                if isinstance(sc, dict) and name in sc:
                    sc[name] *= alpha
    return pm


def adapter_name(arm: str) -> str:
    return label(arm).replace(".", "_")
