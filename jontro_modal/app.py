"""Modal wiring only: image, volumes, GPU functions, CLI entrypoints.

    # CPU-only check of the template, masking and lengths (no secret needed)
    JONTRO_NO_SECRET=1 uvx modal run jontro_modal/app.py::inspect_data --model Qwen/Qwen3.5-2B
    # smoke test, then a real run
    uvx modal run jontro_modal/app.py::main --model Qwen/Qwen3.5-2B --smoke
    uvx modal run jontro_modal/app.py::main --model Qwen/Qwen3.5-2B --variant full
    # all five variants of one model, in parallel
    uvx modal run jontro_modal/app.py::sweep --model Qwen/Qwen3.5-2B

Each run writes its own ``mlflow-<run>.db`` plus adapters/checkpoints to the ``jontro-out`` Volume
Pull it and browse the runs locally:
    uvx modal volume get jontro-out / out_modal/
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal

from jontro_modal.config import MODELS, VARIANTS, TrainConfig

CACHE = "/hf_cache"

app = modal.App("jontro-sft")
hf_cache = modal.Volume.from_name("jontro-hf-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("jontro-out", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    # versions pinned to an environment proven on Modal (Gemma 4 / Qwen3.5 support)
    .pip_install("torch==2.14.0", "transformers==5.16.1", "peft==0.20.0", "accelerate==1.14.0",
                 "mlflow==3.16.0", "liger-kernel==0.8.2", "huggingface_hub", "sentencepiece", "tqdm",
                 "flash-linear-attention")  # fast Gated DeltaNet kernels for Qwen3.5; pure-torch fallback ran at ~700 tok/s
    .env({"HF_HOME": CACHE, "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_file("outputs/release_multiturn/jontro_multiturn_2000_openai.jsonl",
                    "/data/jontro_multiturn_2000_openai.jsonl")
    .add_local_file("outputs/splits/train_ids.jsonl", "/data/train_ids.jsonl")
    .add_local_file("retention/data/general_chat.jsonl", "/retention/general_chat.jsonl")
    .add_local_file("retention/data/tool_restraint.jsonl", "/retention/tool_restraint.jsonl")
    .add_local_file("retention/data/heldout_text.jsonl", "/retention/heldout_text.jsonl")
    # questions only: the answer key never enters the container (blinded)
    .add_local_file("retention/data/truthfulqa48/questions.jsonl", "/retention/truthfulqa48_questions.jsonl")
    .add_local_file("mix/prompts_v1.jsonl", "/mix/prompts_v1.jsonl")
    .add_local_python_source("jontro_modal")
)
if Path("build/benchmark/teacher_forced_test.jsonl").exists():  # scripts/build_benchmark_prompts.py
    image = image.add_local_file("build/benchmark/teacher_forced_test.jsonl", "/benchmark/teacher_forced_test.jsonl")
if Path("mix/mix_v1.jsonl").exists():  # built by scripts/build_mix.py after the generation job
    image = image.add_local_file("mix/mix_v1.jsonl", "/data/mix_v1.jsonl")
# JONTRO_NO_SECRET=1 lets CPU-only inspection run before the hf-token secret exists.
secrets = [] if os.environ.get("JONTRO_NO_SECRET") else [modal.Secret.from_name("hf-token")]
_train_kw = dict(image=image, secrets=secrets, volumes={CACHE: hf_cache, "/out": out_vol}, timeout=3 * 3600)


def _run(cfg: dict) -> dict:
    from jontro_modal.train_core import run_training

    try:
        return run_training(TrainConfig(**cfg))
    finally:
        out_vol.commit()


@app.function(gpu="L40S", **_train_kw)
def train_l40s(cfg: dict) -> dict:
    return _run(cfg)


@app.function(gpu="A100-40GB", **_train_kw)
def train_a100_40(cfg: dict) -> dict:
    return _run(cfg)


@app.function(gpu="A100-80GB", **_train_kw)
def train_a100_80(cfg: dict) -> dict:
    return _run(cfg)


@app.function(gpu="H100", **_train_kw)
def train_h100(cfg: dict) -> dict:
    return _run(cfg)


@app.function(gpu="H100", **_train_kw)
def train_h100_multi(cfgs: list[dict]) -> list[dict]:
    """Several runs concurrently on ONE H100, one process each.

    A 2B LoRA run uses ~15 GB and a fraction of the GPU's compute, so co-running 3-4 runs raises
    aggregate throughput. Each run keeps its own MLflow db, checkpoint dir and adapter.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    from jontro_modal.train_core import run_cfg_dict

    try:
        with ProcessPoolExecutor(len(cfgs), mp_context=mp.get_context("spawn")) as ex:
            return list(ex.map(run_cfg_dict, cfgs))
    finally:
        out_vol.commit()


_BY_GPU = {"L40S": train_l40s, "A100-40GB": train_a100_40, "A100-80GB": train_a100_80, "H100": train_h100}


@app.function(image=image, volumes={CACHE: hf_cache}, timeout=1800)
def inspect_data(model: str, variant: str = "full", max_len: int = 16384) -> dict:
    """CPU-only: sample counts, length percentiles and a masked-label sample for *model*."""
    from transformers import AutoTokenizer

    from jontro_modal.sft_data import diagnose
    from jontro_modal.train_core import load_rows, tokenise

    tok = AutoTokenizer.from_pretrained(model)
    rows, _ = load_rows(TrainConfig(model=model, variant=variant))
    cfg = TrainConfig(model=model, variant=variant)
    kept, rejected = tokenise(rows, tok, max_len, cfg.text_turn_keep)
    if not kept:
        return {"error": "all samples rejected", **diagnose(rows[0], tok)}
    lens = sorted(len(e["input_ids"]) for e in kept)
    pct = lambda p: lens[min(len(lens) - 1, int(p * len(lens)))]  # noqa: E731
    labelled = [t for t, l in zip(kept[0]["input_ids"], kept[0]["labels"]) if l != -100]
    out = {"model": model, "variant": variant, "records": len(rows), "samples": len(kept),
           "rejected": rejected, "len_p50": pct(0.5), "len_p95": pct(0.95), "len_max": lens[-1],
           "label_tokens": sum(sum(x != -100 for x in e["labels"]) for e in kept),
           "total_tokens": sum(lens), "masked_sample": tok.decode(labelled)[:1200]}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


@app.function(gpu="H100", **_train_kw)
def retention_eval(base_model: str, arms: list[str]) -> dict:
    from jontro_modal.retention import run_retention

    try:
        return run_retention(base_model, arms)
    finally:
        out_vol.commit()


@app.local_entrypoint()
def retention(model: str = "Qwen/Qwen3.5-2B", arms: str = "qwen3.5-2b-full-s0") -> None:
    """Generate all retention suites for base + the given adapter run names (see retention/PROTOCOL.md)."""
    print(json.dumps(retention_eval.remote(model, [a for a in arms.split(",") if a]), indent=1))


@app.function(gpu="H100", **_train_kw)
def decode_check_eval(base_model: str, arms: list[str], rep_penalty: float) -> dict:
    from jontro_modal.retention import run_decode_check

    try:
        return run_decode_check(base_model, arms, rep_penalty)
    finally:
        out_vol.commit()


@app.local_entrypoint()
def decode_check(model: str = "Qwen/Qwen3.5-2B", arms: str = "qwen3.5-2b-full-s0,qwen3.5-2b-full_mix-s0",
                 rep_penalty: float = 1.15) -> None:
    """General-chat suite with a repetition penalty, for base + adapters (no training)."""
    print(json.dumps(decode_check_eval.remote(model, [a for a in arms.split(",") if a], rep_penalty), indent=1))


@app.function(gpu="H100", **_train_kw)
def mix_generate(base_model: str = "Qwen/Qwen3.5-2B") -> dict:
    from jontro_modal.mix_gen import run_mix

    try:
        return run_mix(base_model)
    finally:
        out_vol.commit()


@app.local_entrypoint()
def mix(model: str = "Qwen/Qwen3.5-2B") -> None:
    """Generate the base model's own answers for the full_mix prompts (see docs/.../full-mix-plan.md)."""
    print(json.dumps(mix_generate.remote(model), indent=1))


def _benchmark(base_model: str, arms: list[str], stride: int) -> dict:
    from jontro_modal.benchmark import run_benchmark

    try:
        return run_benchmark(base_model, arms, stride)
    finally:
        out_vol.commit()


@app.function(gpu="H100", **_train_kw)
def benchmark_eval(base_model: str, arms: list[str], stride: int) -> dict:
    return _benchmark(base_model, arms, stride)


@app.function(gpu="L40S", **_train_kw)
def benchmark_eval_l40s(base_model: str, arms: list[str], stride: int) -> dict:
    return _benchmark(base_model, arms, stride)


@app.local_entrypoint()
def benchmark(model: str = "Qwen/Qwen3.5-2B", arms: str = "base", stride: int = 4, gpu: str = "H100") -> None:
    """Teacher-forced Jontro benchmark for base + adapters (arm grammar in jontro_modal/arms.py). gpu: H100|L40S."""
    fn = benchmark_eval_l40s if gpu.upper() == "L40S" else benchmark_eval
    print(json.dumps(fn.remote(model, [a for a in arms.split(",") if a], stride), indent=1))


@app.function(gpu="A100-80GB", **_train_kw)
def baseline_benchmark_a100(base_model: str, bs: int) -> dict:
    """Zero-shot benchmark of another base model; outputs go to /out/benchmark_baselines/<slug>/base.jsonl."""
    import os

    slug = base_model.split("/")[-1].lower()
    os.environ["JONTRO_BENCH_OUT"] = f"/out/benchmark_baselines/{slug}"
    from jontro_modal.benchmark import run_benchmark

    try:
        return run_benchmark(base_model, ["base"], 1, bs=bs)
    finally:
        out_vol.commit()


@app.local_entrypoint()
def baseline(model: str = "Qwen/Qwen3.5-9B", bs: int = 8) -> None:
    """Full 1,720-step zero-shot benchmark of a base model on an A100-80GB."""
    print(json.dumps(baseline_benchmark_a100.remote(model, bs), indent=1))


def _sha() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


@app.local_entrypoint()
def main(model: str, variant: str = "full", seed: int = 0, epochs: int = 2, lr: float = 2e-4,
         rank: int = 32, smoke: bool = False, push: bool = True, dev: bool = False,
         gpu: str = "", micro_batch: int = 4, grad_accum: int = 4,
         grad_ckpt: bool = True) -> None:
    cfg = TrainConfig(model=model, variant=variant, seed=seed, epochs=epochs, lr=lr, rank=rank,
                      smoke=smoke, push=push, dev=dev, git_sha=_sha(),
                      micro_batch=micro_batch, grad_accum=grad_accum, grad_ckpt=grad_ckpt)
    print(_BY_GPU[gpu or MODELS[model]].remote(cfg.as_dict()))


@app.local_entrypoint()
def sweep(model: str, variants: str = ",".join(VARIANTS), seeds: str = "0", epochs: int = 2,
          smoke: bool = False, micro_batch: int = 4, grad_accum: int = 4) -> None:
    """All variants x seeds of one model, concurrently on a single H100 (one container)."""
    cfgs = [TrainConfig(model=model, variant=v.strip(), seed=int(sd), epochs=epochs, smoke=smoke,
                        micro_batch=micro_batch, grad_accum=grad_accum, git_sha=_sha()).as_dict()
            for v in variants.split(",") if v.strip() for sd in seeds.split(",")]
    for res in train_h100_multi.remote(cfgs):
        print(res)
