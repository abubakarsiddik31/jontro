"""LoRA SFT loop for Jontro.

Same design: a custom loop (not HF Trainer), a deterministic resume-safe step schedule,
length-grouped micro-batches, fused linear+cross-entropy so full-vocab logits are never
materialised, gradient clip 1.0, cosine schedule with 3% warm-up, and the verified
checkpoint helper. Heavy imports are local so this module imports on a laptop for tests.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

from jontro_modal import sft_data
from jontro_modal.config import TrainConfig

OUT = Path("/out")  # jontro-out volume: per-run mlflow db, checkpoints, adapters
DATA_RELEASE = Path("/data/jontro_multiturn_2000_openai.jsonl")
DATA_TRAIN_IDS = Path("/data/train_ids.jsonl")
DATA_MIX = Path("/data/mix_v1.jsonl")
WARMUP_RATIO = 0.03


def load_rows(cfg: TrainConfig) -> tuple[list[dict], list[dict]]:
    train = sft_data.load_train(DATA_RELEASE, DATA_TRAIN_IDS)
    dev_rows: list[dict] = []
    if cfg.dev:
        train, dev_rows = sft_data.split_dev(train)
    mix = sft_data.load_mix(DATA_MIX) if cfg.variant == "full_mix" else None
    return sft_data.make_variant(train, cfg.variant, cfg.seed, mix), dev_rows


def tokenise(rows: list[dict], tok, max_len: int, text_turn_keep: float = 1.0) -> tuple[list[dict], int]:
    kept: list[dict] = []
    rejected = 0
    for r in rows:
        ex, rej = sft_data.build_examples(r, tok, max_len, text_turn_keep)
        kept += ex
        rejected += rej
    return kept, rejected


def collate(batch: list[dict], pad_id: int):
    import torch

    n = max(len(b["input_ids"]) for b in batch)
    ids, lab, att = [], [], []
    for b in batch:
        k = n - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * k)
        lab.append(b["labels"] + [-100] * k)
        att.append([1] * len(b["input_ids"]) + [0] * k)
    return torch.tensor(ids), torch.tensor(lab), torch.tensor(att)


def build_schedule(n: int, lengths: list[int], cfg: TrainConfig, epochs: int):
    """schedule[epoch] = [optimizer step = [micro-batch = [example idx]]]; deterministic in seed.

    Each effective-batch chunk is sorted by length (if enabled) so micro-batches pad little,
    and chunks never mix across steps.
    """
    import torch

    g = torch.Generator().manual_seed(cfg.seed)
    chunk_rng = random.Random(cfg.seed + 1)
    eff = cfg.micro_batch * cfg.grad_accum
    schedule = []
    for _ in range(epochs):
        idx = torch.randperm(n, generator=g).tolist()
        chunks = [idx[i:i + eff] for i in range(0, len(idx), eff)]
        chunk_rng.shuffle(chunks)
        steps = []
        for chunk in chunks:
            if cfg.length_group:
                chunk.sort(key=lambda i: -lengths[i])
            micros = [chunk[j:j + cfg.micro_batch] for j in range(0, len(chunk), cfg.micro_batch)]
            for k in range(0, len(micros), cfg.grad_accum):
                steps.append(micros[k:k + cfg.grad_accum])
        schedule.append(steps)
    return schedule


def build_model(cfg: TrainConfig):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0})
    if cfg.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lora = LoraConfig(
        r=cfg.rank, lora_alpha=cfg.rank, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules="all-linear", exclude_modules=r".*(visual|vision).*")
    model = get_peft_model(model, lora)
    trained = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trained and not any("visual" in n or "vision" in n for n in trained), trained[:5]
    census = lora_census(model)
    print("LoRA module census:", census, flush=True)
    # Qwen3.5 is 3/4 Gated DeltaNet: its projections are not q/k/v/o, so check they were adapted
    assert "in_proj_qkv" in census and "q_proj" in census, f"missing target modules: {census}"
    # tied lm_head/embedding must not accumulate a dense gradient
    model.get_input_embeddings().weight.requires_grad_(False)
    model.train()  # activates LoRA dropout and gates gradient checkpointing
    return model


def lora_census(model) -> str:
    """Count of adapted modules by name, e.g. 'q_proj:6,in_proj_qkv:18,...' (logged to MLflow)."""
    from collections import Counter

    c = Counter(n.split(".lora_A")[0].split(".")[-1] for n, p in model.named_parameters()
                if p.requires_grad and ".lora_A" in n)
    return ",".join(f"{k}:{v}" for k, v in sorted(c.items()))


class FusedLoss:
    """hidden @ lm_head^T + CE over labelled positions only, logits never fully materialised.

    Uses Liger's fused kernel when available, else plain CE on the selected
    positions (still ~8x smaller than full-sequence logits since only ~13% of tokens are labelled).
    """

    def __init__(self, model) -> None:
        import torch.nn.functional as F

        self.core = model.get_base_model()
        self.inner = self.core.model  # returns last_hidden_state
        self.w = self.core.get_output_embeddings().weight
        self.F = F
        try:
            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

            self.flce = LigerFusedLinearCrossEntropyLoss(reduction="sum")
        except Exception:  # noqa: BLE001 - optional accelerator
            self.flce = None

    def __call__(self, input_ids, attn, labels):
        hs = self.inner(input_ids=input_ids, attention_mask=attn).last_hidden_state
        tgt = labels[:, 1:]  # causal shift happens here (HF does it inside its own loss)
        mask = tgt != -100
        h = hs[:, :-1][mask]
        t = tgt[mask]
        if self.flce is not None:
            total = self.flce(self.w, h, t)
        else:
            total = self.F.cross_entropy((h @ self.w.T).float(), t, reduction="sum")
        return total, int(mask.sum())


def commit_volume() -> None:
    """Persist everything written to /out so far. Modal volumes only keep committed writes, so a
    crash or timeout would otherwise lose the checkpoints written since the last commit."""
    try:
        import modal

        modal.Volume.from_name("jontro-out").commit()
    except Exception as exc:  # noqa: BLE001 - never let bookkeeping kill a training run
        print(f"volume commit failed (continuing): {exc}", flush=True)


def push_adapter(cfg: TrainConfig, folder: Path, subdir: str = "") -> bool:
    """Upload an adapter folder to the HF repo; a failure is logged, never fatal."""
    if not cfg.push or cfg.smoke:
        return False
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(cfg.hf_repo, private=True, exist_ok=True)
        api.upload_folder(folder_path=str(folder), repo_id=cfg.hf_repo, path_in_repo=subdir or ".")
        print(f"pushed {folder} -> {cfg.hf_repo}/{subdir}", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001 - adapter stays on the volume; scripts/push_adapters_hf.py retries
        print(f"HF push failed (adapter kept on volume): {exc}", flush=True)
        return False


def sha_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def run_training(cfg: TrainConfig) -> dict:
    import mlflow
    import numpy as np
    import torch
    import transformers
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    from jontro_modal.training_checkpoint import load_checkpoint, save_checkpoint

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    tok = AutoTokenizer.from_pretrained(cfg.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rows, _ = load_rows(cfg)
    examples, rejected = tokenise(rows, tok, cfg.max_len, cfg.text_turn_keep)
    epochs = cfg.epochs
    if cfg.smoke:  # stress test: the longest samples, so an OOM shows up here and not hours in
        examples = sorted(examples, key=lambda e: -len(e["input_ids"]))[:cfg.micro_batch * cfg.grad_accum * 3]
        epochs = 1
    lengths = [len(e["input_ids"]) for e in examples]
    supervised = sum(sum(x != -100 for x in e["labels"]) for e in examples)
    data_hash = hashlib.sha256(json.dumps([r["id"] for r in rows]).encode()).hexdigest()[:16]

    experiment = "jontro-sft-smoke" if cfg.smoke else "jontro-sft"
    uri = f"sqlite:///{OUT}/mlflow-{cfg.run_name}.db"  # one db per run: no cross-container writers
    ckpt_dir = OUT / "ckpt" / cfg.run_name
    if cfg.smoke:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)

    model = build_model(cfg)
    fused = FusedLoss(model)
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0, foreach=False)
    schedule = build_schedule(len(examples), lengths, cfg, epochs)
    total_steps = sum(len(s) for s in schedule)
    sched = get_cosine_schedule_with_warmup(opt, int(total_steps * WARMUP_RATIO), total_steps)

    identity = {**cfg.as_dict(), "data_hash": data_hash, "epochs": epochs, "n_examples": len(examples),
                "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    losses: list[dict] = []
    resume_step = 0
    resumed = load_checkpoint(ckpt_dir, identity, total_steps)
    if resumed is not None:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        meta, st, adapter_path = resumed
        resume_step = meta["step"]
        set_peft_model_state_dict(model, load_file(adapter_path, device="cuda"))
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        torch.set_rng_state(st["cpu_rng"])
        random.setstate(st["python_rng"])
        np.random.set_state(st["numpy_rng"])
        if st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda_rng"]])
        losses = st["losses"]
        print(f"resumed verified checkpoint at step {resume_step}/{total_steps}", flush=True)

    step, t0 = 0, time.time()
    epoch_ends = list(np.cumsum([len(s) for s in schedule]))
    tokens_seen = 0
    with mlflow.start_run(run_name=cfg.run_name):
        mlflow.log_params({
            **cfg.as_dict(), "epochs": epochs, "n_examples": len(examples), "n_rejected": rejected,
            "n_records": len(rows), "supervised_tokens_per_epoch": supervised,
            "lora_alpha": cfg.rank, "lora_module_census": lora_census(model), "n_trainable_params": n_train, "thinking": "off",
            "effective_batch": cfg.micro_batch * cfg.grad_accum, "grad_clip": 1.0,
            "liger_flce": fused.flce is not None, "gpu": torch.cuda.get_device_name(0),
            "resumed_from_step": resume_step,
        })
        mlflow.set_tags({
            "experiment": experiment, "smoke": str(cfg.smoke).lower(), "git_sha": cfg.git_sha,
            "data_hash": data_hash, "hf_repo": cfg.hf_repo,
            "transformers": transformers.__version__, "torch": torch.__version__,
        })
        for epoch in range(epochs):
            for micros in schedule[epoch]:
                if step < resume_step:  # steps already done before a kill: skip forward
                    step += 1
                    continue
                n_items = sum(sum(x != -100 for x in examples[i]["labels"][1:]) for m in micros for i in m)
                step_loss = 0.0
                for m_idxs in micros:
                    ids, lab, att = (t.to("cuda") for t in collate([examples[i] for i in m_idxs], pad_id))
                    total, _ = fused(ids, att, lab)
                    (total / n_items).backward()  # token-weighted mean across the whole step
                    step_loss += total.item()
                    tokens_seen += int(att.sum())
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                m_loss = step_loss / n_items
                if step % 5 == 0 or step == total_steps or step <= resume_step + 5:
                    mem = torch.cuda.max_memory_allocated() / 1e9
                    losses.append({"step": step, "loss": round(m_loss, 4)})
                    mlflow.log_metrics({"loss": round(m_loss, 4), "gpu_mem_gb": round(mem, 3)}, step=step)
                    print(f"epoch {epoch + 1} step {step}/{total_steps} loss {m_loss:.4f} "
                          f"gpu {mem:.1f}GB ({time.time() - t0:.0f}s)", flush=True)
                if cfg.ckpt_every and not cfg.smoke and (
                        step % cfg.ckpt_every == 0 or step in epoch_ends or step == total_steps):
                    save_checkpoint(model, opt, sched, losses, step, total_steps, identity, ckpt_dir)
                    commit_volume()
                    print(f"checkpoint saved at step {step}", flush=True)
            if epoch + 1 < epochs and not cfg.smoke:  # protect finished epochs on HF as well
                snap = OUT / "adapters" / f"{cfg.run_name}-epoch{epoch + 1}"
                snap.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(snap)
                push_adapter(cfg, snap, f"epoch-{epoch + 1}")
        wall = time.time() - t0
        tps = tokens_seen / max(wall, 1e-9)
        mlflow.log_metrics({"tokens_per_sec": tps, "train_runtime_s": wall,
                            "final_loss": losses[-1]["loss"] if losses else float("nan")})
        out = save_adapter(model, tok, cfg, data_hash, losses, wall)
        mlflow.log_artifacts(str(out), "adapter")
        commit_volume()  # adapter + mlflow db are safe on the volume before any network push
        pushed = push_adapter(cfg, out)
        mlflow.set_tag("hf_pushed", str(pushed).lower())
    commit_volume()
    return {"run": cfg.run_name, "repo": cfg.hf_repo if pushed else None, "hf_pushed": pushed,
            "final_loss": losses[-1]["loss"] if losses else None, "examples": len(examples),
            "steps": total_steps, "tokens_per_sec": round(tps), "train_runtime_s": round(wall),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}


def save_adapter(model, tok, cfg: TrainConfig, data_hash: str, losses: list[dict], wall: float) -> Path:
    out = OUT / "adapters" / cfg.run_name
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "run_manifest.json").write_text(json.dumps(
        {**cfg.as_dict(), "data_hash": data_hash, "wall_clock_s": round(wall, 1), "loss_trace": losses}, indent=1))
    return out


def run_cfg_dict(cfg: dict) -> dict:
    """Process-pool entry point: one training run per spawned process (clean CUDA state each)."""
    return run_training(TrainConfig(**cfg))
