"""Single source of truth for model registry, hyperparameters and naming."""
from __future__ import annotations

from dataclasses import asdict, dataclass

HF_USER = "abubakar-siddik"
VARIANTS = ("full", "full_mix", "bengali_only", "first_turn", "no_tui")  # `half` dropped: weakest claim

# model id -> GPU. LoRA + gradient checkpointing; sized with headroom for 4k-token samples.
MODELS: dict[str, str] = {  # H100: measured cheapest per token (L40S 3.2k, A100 5k, H100 9.6k tok/s)
    "Qwen/Qwen3.5-2B": "H100",
    "Qwen/Qwen3.5-4B": "H100",
    "Qwen/Qwen3.5-9B": "H100",
    
    
    
}


@dataclass(frozen=True)
class TrainConfig:
    model: str
    variant: str = "full"
    seed: int = 0
    epochs: int = 2
    text_turn_keep: float = 0.25  # fraction of non-tool-call assistant turns trained on (cost control)
    lr: float = 2e-4
    rank: int = 32
    max_len: int = 8192
    micro_batch: int = 4
    grad_accum: int = 4  # micro_batch * grad_accum = effective batch 16
    grad_ckpt: bool = True  # needed once micro-batches grow
    length_group: bool = True  # sort within each effective batch to cut padding
    ckpt_every: int = 50
    dev: bool = False  # carve a goal-disjoint dev set out of train (tuning only)
    smoke: bool = False  # 24 records, 1 epoch, nothing pushed
    push: bool = True
    git_sha: str = ""

    @property
    def short(self) -> str:
        return self.model.split("/")[-1].lower()

    @property
    def run_name(self) -> str:
        return f"{self.short}-{self.variant}-s{self.seed}" + ("-smoke" if self.smoke else "")

    @property
    def hf_repo(self) -> str:
        seed = "" if self.seed == 0 else f"-s{self.seed}"
        return f"{HF_USER}/jontro-{self.short}-{self.variant}{seed}-lora"

    def as_dict(self) -> dict:
        return asdict(self)
