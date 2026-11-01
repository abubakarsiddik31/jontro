# jontro_modal — small-model LoRA training on Modal

| File | Role |
|---|---|
| `config.py` | model→GPU registry, `TrainConfig`, HF repo naming |
| `sft_data.py` | variants (`full/half/bengali_only/first_turn/no_tui`), dev split, per-turn sample builder |
| `train_core.py` | LoRA SFT, MLflow logging, adapter push (runs in the container) |
| `app.py` | Modal image/volumes/functions and CLI entrypoints (`inspect_data`, `main`, `sweep`) |

Run order and commands are in the `app.py` docstring. One-time setup: create the Modal secret
`hf-token` (`HF_TOKEN=<write token>`). Adapters are pushed private to
`abubakar-siddik/jontro-<model>-<variant>-lora`.
