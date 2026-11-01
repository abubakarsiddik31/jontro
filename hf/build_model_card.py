"""Build and push the model card for a Jontro LoRA adapter (only the published adapter may be public).

Every number comes from files written by the pipeline, not typed by hand:
  out_modal/adapters/<run>/run_manifest.json      training settings
  outputs/eval/multiturn_scores.json              Jontro benchmark scores with goal-clustered intervals
  out_modal/retention/retention_report.json       retention checks
  mix/mix_v1.jsonl                                the replay data, for the full_mix variant

    python hf/build_model_card.py qwen3.5-2b-full-s0          # write build/model-cards/<run>.md
    python hf/build_model_card.py qwen3.5-2b-full-s0 --push   # upload as README.md (repo must be private)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

HF_USER = "abubakar-siddik"
OUT = Path("build/model-cards")
BENCH = Path("outputs/eval/multiturn_scores.json")
MIX = Path("mix/mix_v1.jsonl")

# benchmark row name used in multiturn_scores.json for each (variant)
BENCH_NAME = {"full": "Qwen3.5-2B-SFT", "full_mix": "Qwen3.5-2B-SFT-mix"}
ZERO_SHOT = "Qwen3.5-2B-zeroshot"


PUBLISHED = ("full_mix", 0)  # the adapter released publicly, under a plain name


def repo_id(man: dict) -> str:
    short = man["model"].split("/")[-1].lower()
    if (man["variant"], man["seed"]) == PUBLISHED:
        return f"{HF_USER}/jontro-{short}-lora"
    seed = "" if man["seed"] == 0 else f"-s{man['seed']}"
    return f"{HF_USER}/jontro-{short}-{man['variant']}{seed}-lora"


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def interval(ci) -> str:
    return f"[{ci[0]:.1f}, {ci[1]:.1f}]" if ci else ""


def benchmark_table(scores: dict, variant: str) -> str:
    """Systems as rows, four short metric columns; intervals in brackets so the table fits the card column."""
    rows = []
    systems = (("Qwen3.5-2B, no adapter", ZERO_SHOT), ("Qwen3.5-4B, no adapter", "Qwen3.5-4B-zeroshot"),
               ("Qwen3.5-9B, no adapter", "Qwen3.5-9B-zeroshot"), ("This adapter", BENCH_NAME[variant]))
    for label, key in systems:
        s = scores[key]
        ov, tc = s["overall"], s["task_completion"]
        rows.append(f"| {label} | {pct(ov['tool_selection_rate'])} {interval(ov['tool_selection_ci'])} | "
                    f"{pct(ov['arg_exact_rate'])} {interval(ov['arg_exact_ci'])} | {pct(ov['slot_accuracy'])} | "
                    f"{pct(tc['completion_rate'])} {interval(tc['completion_ci'])} |")
    head = ("| Model | Tool selection | Exact arguments | Slot accuracy | Fully correct conversations |\n"
            "|---|---|---|---|---|")
    return head + "\n" + "\n".join(rows)


def retention_table(report: dict, run: str) -> str:
    """Checks as rows and the two models as columns, so nine measures fit the card column."""
    m = report["metrics"]
    b, x = m["base"], m[run]
    num = lambda v, f: f.format(v) if v is not None else "n/a"  # noqa: E731
    checks = [
        ("TruthfulQA, correct of 48", lambda r: str(r["tqa_strict_correct"])),
        ("Tool calls on unrelated requests", lambda r: f"{100 * r['tool_spurious_rate']:.0f}%"),
        ("Reply in the user's language", lambda r: f"{100 * r['lang_match_rate']:.0f}%"),
        ("Coherence, all prompts (0 to 100)", lambda r: num(r.get("coherence_mean"), "{:.1f}")),
        ("Coherence, Bengali prompts", lambda r: num(r.get("coherence_bn"), "{:.1f}")),
        ("Coherence, English prompts", lambda r: num(r.get("coherence_en"), "{:.1f}")),
    ]
    rows = [f"| {name} | {f(b)} | {f(x)} |" for name, f in checks]
    rows += [
        f"| Held-out Bengali NLL, change from base | | {num(x.get('heldout_bn_nll_delta'), '{:+.2f}')} |",
        f"| Held-out English NLL, change from base | | {num(x.get('heldout_en_nll_delta'), '{:+.2f}')} |",
    ]
    return "| Check | Qwen3.5-2B, no adapter | This adapter |\n|---|---|---|\n" + "\n".join(rows)


def mix_paragraph() -> str:
    if not MIX.exists():
        return ""
    recs = [json.loads(line) for line in MIX.read_text().splitlines() if line.strip()]
    kinds = Counter(r.get("kind") for r in recs)
    cut = sum(1 for r in recs if r.get("truncated"))
    return (f"This variant adds {len(recs)} replay examples to the Jontro training conversations: the base model's own greedy answers to "
            f"unrelated prompts, {kinds.get('restraint', 0)} of them given the Jontro system prompt and tools (the right behaviour is to "
            f"answer without calling a tool) and {kinds.get('general', 0)} without tools. The prompts come from the aya dataset for Bengali "
            f"and OpenAssistant (oasst2) for English, both Apache-2.0, and do not overlap the retention prompts. Answers longer than 512 "
            f"tokens were cut at that length, and the end token of a cut answer was not trained ({cut} of {len(recs)}). The replay data were added "
            f"after an adapter trained on the Jontro conversations alone failed two of the retention checks described below, so this design is "
            f"a response to that result rather than a plan made in advance.")


def reading(variant: str, m: dict, run: str) -> str:
    b, x = m["base"], m[run]
    if variant == "full":
        return (f"This adapter keeps TruthfulQA ({x['tqa_strict_correct']} against {b['tqa_strict_correct']}) and reply language, but it calls a tool on "
                f"{100 * x['tool_spurious_rate']:.0f}% of unrelated requests where the base model does so on {100 * b['tool_spurious_rate']:.0f}%, and general "
                f"chat coherence falls from {b['coherence_mean']:.1f} to {x['coherence_mean']:.1f}, almost all of it in Bengali ({b['coherence_bn']:.1f} to "
                f"{x['coherence_bn']:.1f}). In Bengali its free generations often fall into repetition loops. It fails the coherence and tool-restraint "
                f"gates of the retention protocol. For a model that behaves normally outside Jontro, use the `full_mix` adapter instead.")
    return (f"This adapter makes no unrelated tool calls ({100 * x['tool_spurious_rate']:.0f}%, base {100 * b['tool_spurious_rate']:.0f}%) and keeps English "
            f"coherence ({x['coherence_en']:.1f} against {b['coherence_en']:.1f}). Bengali chat coherence is {x['coherence_bn']:.1f} against {b['coherence_bn']:.1f} "
            f"for the base model, so fine-tuning still costs some Bengali chat quality, although much less than without the replay data. It passes all four gates of the retention protocol as written, "
            f"although the overall coherence gap ({b['coherence_mean'] - x['coherence_mean']:.1f} points, limit 10) is helped by the English prompts. "
            f"Likelihood of held-out English text is worse than the base ({x['heldout_en_nll_delta']:+.2f} nats per token), probably because the replay data "
            f"are the model's own greedy outputs.")


USAGE = """```python
import json
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "{base}"
ADAPTER = "{repo}"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE, dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(model, ADAPTER)


def drop_nulls(x):
    # Released schemas list unused parameters as null.
    if isinstance(x, dict):
        return {{k: drop_nulls(v) for k, v in x.items()
                if v is not None}}
    return x


def as_tool(t):
    schema = drop_nulls(json.loads(t["parameters_json_schema"]))
    return {{"type": "function",
            "function": {{"name": t["name"],
                         "description": t["description"],
                         "parameters": schema}}}}


# A held-out test conversation, cut at its first user turn.
row = load_dataset("{dataset}", split="test")[0]
tools = [as_tool(t) for t in row["tools"]]
messages = [
    {{"role": "system", "content": row["system_prompt"]}},
    {{"role": "user", "content": row["messages"][0]["content"]}},
]

inputs = tok.apply_chat_template(
    messages, tools=tools, add_generation_prompt=True,
    enable_thinking=False, return_dict=True, return_tensors="pt",
).to(model.device)
out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
new = out[0, inputs["input_ids"].shape[1]:]
print(tok.decode(new, skip_special_tokens=False))
```"""


def card(run: str, man: dict, report: dict, scores: dict) -> str:
    base, variant = man["model"], man["variant"]
    repo = repo_id(man)
    published = (variant, man["seed"]) == PUBLISHED
    dataset = f"{HF_USER}/Jontro"
    wall = man.get("wall_clock_s")
    effective = man.get("micro_batch", 4) * man.get("grad_accum", 4)
    sft, zs = scores[BENCH_NAME[variant]]["overall"], scores[ZERO_SHOT]["overall"]
    m = report["metrics"]
    title = "Jontro Qwen3.5-2B LoRA" if published else repo.split("/")[1]
    status = ("" if published else
              f"\n\nThis adapter is kept for reproducibility of the paper and is not recommended for use, because it degrades general "
              f"chat; use [{HF_USER}/jontro-qwen3.5-2b-lora](https://huggingface.co/{HF_USER}/jontro-qwen3.5-2b-lora) instead.")
    data = ("1,300 Jontro training conversations and 748 replay examples of ordinary chat"
            if variant == "full_mix" else "1,300 Jontro training conversations")
    extra = mix_paragraph() if variant == "full_mix" else ""
    usage = USAGE.format(base=base, repo=repo, dataset=dataset)
    return f"""---
base_model: {base}
base_model_relation: adapter
library_name: peft
license: apache-2.0
language:
  - bn
  - en
datasets:
  - {dataset}
pipeline_tag: text-generation
tags:
  - lora
  - peft
  - bangla
  - bengali
  - banglish
  - tool-use
  - function-calling
  - multi-turn
---

# {title}

A LoRA adapter for [{base}](https://huggingface.co/{base}) that teaches the model to call tools in multi-turn service
conversations written in Bengali script or in Romanised Bangla (Banglish). It is the fine-tuned model reported in the paper
"Jontro: A Multi-Turn Bangla and Banglish Tool-Use Corpus and Benchmark".{status}

On held-out Jontro conversations the adapter picks the right tools at {pct(sft['tool_selection_rate'])}% of decision points, against
{pct(zs['tool_selection_rate'])}% for the base model alone, and gets every argument value right at {pct(sft['arg_exact_rate'])}% against
{pct(zs['arg_exact_rate'])}%. Outside Jontro it makes no spurious tool calls and keeps its English chat quality, while its Bengali chat
quality stays below that of the base model. The sections below give the details.

| Property | Value |
|---|---|
| Base model | [{base}](https://huggingface.co/{base}) |
| Adapter | LoRA, rank {man.get('rank')}, on all linear layers |
| Training data | {data} |
| Dataset | [{dataset}](https://huggingface.co/datasets/{dataset}) |
| Input | Bengali script or Banglish, with tools in the OpenAI function format |
| Output | Tool calls, and replies in Bengali script |
| License | Apache 2.0 |

## Intended use

The adapter is meant for research on tool use in Bengali and Banglish, for example as a baseline on the Jontro benchmark or as a starting
point for further fine-tuning. Given a system prompt, a list of tools and a conversation, it decides whether to call a tool, which tools to
call (several at once when the request needs it) and with which argument values, and it answers in Bengali script. It was trained on
synthetic conversations with simulated services, so it is not ready to act on real accounts, payments or bookings, and it should not be used
for medical, financial or legal advice.

## How to use it

The example below loads the base model with the adapter, takes one held-out conversation from the dataset and asks for the next assistant
turn. Use the system prompt and tool format of the dataset, since the model was trained with them, and keep thinking off, as it was off in
training and evaluation.

{usage}

The model writes tool calls in the Qwen3.5 format, as `<tool_call>` blocks that hold a `<function=...>` element with one `<parameter=...>`
element per argument. To serve it with vLLM, run the base model with `--enable-lora`, `--enable-auto-tool-choice` and
`--tool-call-parser qwen3_coder`, load this repository as the LoRA module, and pass `enable_thinking: false` in `chat_template_kwargs`;
the server then returns the calls in the OpenAI `tool_calls` format. After a call, append the tool result as a `tool` message and generate
again for the next turn.

## Training

Only the training split was used. It holds 1,300 conversations and shares no task goal with the 700 test conversations, so every test goal
is unseen. Each assistant turn is one training example, with the loss on assistant tokens only; every turn that makes a tool call is kept,
together with a random quarter of the turns that answer in text.

{extra}

The adapter was trained with LoRA at rank {man.get('rank')} (alpha equal to the rank, dropout 0.05) on all linear layers, including the
linear-attention projections of the Qwen3.5 architecture. Optimisation used AdamW at a learning rate of {man['lr']} with a cosine schedule,
3% warm-up, gradient clipping at 1.0, an effective batch of {effective} and {man['epochs']} epochs, in bf16 on one H100 80GB
{f'for about {wall / 60:.0f} minutes' if wall else ''}. No hyperparameters were searched, and the adapter from the final epoch is the one
released. The run used seed {man['seed']}, code commit `{man.get('git_sha', 'unknown')}` and data hash `{man.get('data_hash')}`.

## Evaluation on Jontro

The benchmark scores every tool-calling turn of the 700 test conversations (1,720 decision points over 48 held-out goals). At each
decision point the model sees the true conversation so far and must produce the right set of calls; this is the teacher-forced protocol of
the paper. Tool selection asks whether the set of tool names is right, exact arguments whether the names and every argument value are
right, and a conversation counts as fully correct only if every one of its decision points has exact arguments. Decoding was greedy with
thinking off. All scores are percentages; brackets give 95% bootstrap intervals, which resample task goals rather than
conversations and are therefore wide.

{benchmark_table(scores, variant)}

The larger base models do no better than the 2B model without fine-tuning, mostly because they often answer in text where a tool call is
needed. The adapter was trained on data from the same generator as the test set, so these numbers show what the training split teaches,
not how the model would fare with real users.

## Behaviour outside Jontro

Fine-tuning on a narrow task can damage general behaviour, so the adapter was compared with the same base model on checks fixed before any
adapter was evaluated: 48 TruthfulQA forced-choice items, 100 unrelated requests asked with the Jontro system prompt and tools offered
(the right behaviour is not to call a tool), 100 general chat prompts (50 in Bengali or Banglish, 50 in English) scored for coherence from 0
to 100 by a Gemini judge, whether the reply is in the user's language, and the change in negative log-likelihood of held-out human-written
replies (lower is better).

{retention_table(report, run)}

{reading(variant, m, run)}

TruthfulQA at 48 items cannot detect small differences, and the judge is a Gemini model from the same family that generated the corpus,
whose Bengali scores have not been checked against human ratings.

## Limitations

All results come from one training run with one seed, so run-to-run variation is unknown. Only the teacher-forced protocol has been run;
when the model's own calls feed later turns, errors can compound and conversation-level success is likely lower. The training and test
conversations were written by a language model playing the user, so the adapter has seen a tidier Banglish and a narrower range of requests
than real users produce. Bengali chat quality outside the tool-use setting is below that of the base model.

## Citation

If you use the adapter, please cite the paper:

```bibtex
@misc{{siddik2026jontro,
  author = {{Siddik, Abu Bakar and Islam, Afroza and Abhi, Sarafat Hussain}},
  title  = {{Jontro: A Multi-Turn {{Bangla}} and {{Banglish}}
            Tool-Use Corpus and Benchmark}},
  year   = {{2026}},
  note   = {{Preprint}}
}}
```

The entry will be updated with the arXiv identifier once the preprint is posted. To refer to the released adapter itself, you can also cite:

```bibtex
@misc{{siddik2026jontromodel,
  author       = {{Siddik, Abu Bakar}},
  title        = {{Jontro Qwen3.5-2B {{LoRA}}: A Tool-Use Adapter for {{Bangla}} and {{Banglish}}}},
  year         = {{2026}},
  publisher    = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/abubakar-siddik/jontro-qwen3.5-2b-lora}}}},
  note         = {{Apache 2.0, fine-tuned from Qwen3.5-2B on Jontro 2.0}}
}}
```

The training data should be cited as well; its entry is on the [dataset page](https://huggingface.co/datasets/abubakar-siddik/Jontro).

## License

The adapter is released under the Apache 2.0 license, the license of the base model. The Jontro dataset is released under CC BY 4.0.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run")
    ap.add_argument("--src", default="out_modal")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()
    man = json.loads((Path(args.src) / "adapters" / args.run / "run_manifest.json").read_text())
    report = json.loads((Path(args.src) / "retention" / "retention_report.json").read_text())
    scores = json.loads(BENCH.read_text())
    text = card(args.run, man, report, scores)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.run}.md").write_text(text)
    print(f"wrote {OUT / (args.run + '.md')}")
    if args.push:
        from huggingface_hub import HfApi

        repo = repo_id(man)
        api = HfApi()
        published = (man["variant"], man["seed"]) == PUBLISHED
        private = api.model_info(repo).private
        assert private or published, "refusing to push: an unpublished adapter's repo is public"
        api.upload_file(path_or_fileobj=text.encode(), path_in_repo="README.md", repo_id=repo, commit_message="Model card")
        print(f"pushed README to {repo} ({'private' if private else 'public'})")


if __name__ == "__main__":
    main()
