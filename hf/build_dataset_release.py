"""Build (and optionally push / publish) the Jontro 2.0 Hugging Face dataset repo.

The card's numbers are read from outputs/stats/jontro2_stats.json, never typed. The repo is
created/kept private; --push first tags the current main as ``singleturn-v2-archive`` (so the old
9,158-trajectory release stays recoverable) and then replaces main with the Jontro 2.0 tree in one commit.

    python hf/build_dataset_release.py            # build build/hf-dataset/ only
    python hf/build_dataset_release.py --push     # archive tag + replace main (stays private)
    python hf/build_dataset_release.py --publish  # push card update while private, tag v2.0, THEN make public
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

REPO_ID = "abubakar-siddik/Jontro"
OUT = Path("build/hf-dataset")
ARCHIVE_TAG = "singleturn-v2-archive"

COPY = {  # local -> path in repo
    "outputs/parquet/jontro_train.parquet": "data/train/jontro_train.parquet",
    "outputs/parquet/jontro_test.parquet": "data/test/jontro_test.parquet",
    "outputs/parquet/jontro_train_sharegpt.parquet": "data/train/jontro_train_sharegpt.parquet",
    "outputs/parquet/jontro_test_sharegpt.parquet": "data/test/jontro_test_sharegpt.parquet",
    "outputs/release_multiturn/jontro_multiturn_2000.jsonl": "raw/jontro_multiturn_2000.jsonl",
    "outputs/release_multiturn/jontro_multiturn_2000_openai.jsonl": "raw/jontro_multiturn_2000_openai.jsonl",
    "outputs/release_multiturn/jontro_multiturn_2000_sharegpt.jsonl": "raw/jontro_multiturn_2000_sharegpt.jsonl",
    "outputs/release_multiturn/jontro_multiturn_100.jsonl": "raw/jontro_multiturn_100_golden.jsonl",
    "outputs/release_multiturn/generation_report.json": "reports/generation_report.json",
    "outputs/splits/train_ids.jsonl": "splits/train_ids.jsonl",
    "outputs/splits/test_ids.jsonl": "splits/test_ids.jsonl",
    "outputs/splits/test_goals.jsonl": "splits/test_goals.jsonl",
    "outputs/stats/jontro2_stats.json": "reports/jontro2_stats.json",
    "outputs/eval/multiturn_scores.json": "baselines/multiturn_scores.json",
    "outputs/eval/multiturn_results.md": "baselines/multiturn_results.md",
    "outputs/eval/multiturn_by_turn.png": "baselines/multiturn_by_turn.png",
    "outputs/stats/fig_domains.png": "figures/domains.png",
    "outputs/stats/fig_register_script.png": "figures/register_script.png",
    "outputs/stats/fig_user_turns.png": "figures/user_turns.png",
    "outputs/stats/fig_closure.png": "figures/closure.png",
}


def table(d: dict, head: tuple[str, str], fmt=lambda k: k) -> str:
    rows = "\n".join(f"| {fmt(k)} | {v:,} |" for k, v in d.items())
    return f"| {head[0]} | {head[1]} |\n|---|---:|\n{rows}"


MODEL_ID = "abubakar-siddik/jontro-qwen3.5-2b-lora"

BIBTEX = """@misc{siddik2026jontro,
  author = {Siddik, Abu Bakar and Islam, Afroza and Abhi, Sarafat Hussain},
  title  = {Jontro: A Multi-Turn {Bangla} and {Banglish} Tool-Use Corpus and Benchmark},
  year   = {2026},
  note   = {Preprint}
}"""

BIBTEX_DATA = """@misc{siddik2026jontrodata,
  author       = {Siddik, Abu Bakar},
  title        = {Jontro 2.0: A Multi-Turn {Bangla} and {Banglish} Tool-Use Corpus},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\\url{https://huggingface.co/datasets/abubakar-siddik/Jontro}},
  note         = {Version 2.0, CC BY 4.0}
}"""

CFF = """cff-version: 1.2.0
message: If you use this dataset, please cite the paper given under preferred-citation.
type: dataset
title: "Jontro 2.0: A Multi-Turn Bangla and Banglish Tool-Use Corpus"
version: "2.0"
date-released: "2026-10-19"
license: CC-BY-4.0
url: "https://huggingface.co/datasets/abubakar-siddik/Jontro"
repository: "https://huggingface.co/datasets/abubakar-siddik/Jontro"
authors:
  - family-names: Siddik
    given-names: Abu Bakar
    orcid: "https://orcid.org/0009-0003-7834-7469"
keywords:
  - Bangla
  - Banglish
  - tool use
  - function calling
  - multi-turn dialogue
preferred-citation:
  type: article
  title: "Jontro: A Multi-Turn Bangla and Banglish Tool-Use Corpus and Benchmark"
  year: 2026
  notes: "Preprint"
  authors:
    - family-names: Siddik
      given-names: Abu Bakar
      orcid: "https://orcid.org/0009-0003-7834-7469"
      affiliation: "Rajshahi University of Engineering & Technology"
    - family-names: Islam
      given-names: Afroza
      affiliation: "Rajshahi University of Engineering & Technology"
    - family-names: Abhi
      given-names: Sarafat Hussain
      affiliation: "Rajshahi University of Engineering & Technology"
"""


def card(s: dict) -> str:
    c, sp = s["corpus"], s["splits"]
    cells = s["register_script_cells"]
    grid = "\n".join(
        f"| {r} | {cells[f'{r}|bengali']} | {cells[f'{r}|banglish']} |" for r in ("apni", "tumi", "tui"))
    vf = s["validation_flags"]
    closures = ", ".join(f"{k.replace('_', ' ')} ({v})" for k, v in s["closure_style"].items() if k != "unset")
    return f"""---
license: cc-by-4.0
language:
  - bn
task_categories:
  - text-generation
size_categories:
  - 1K<n<10K
pretty_name: Jontro 2.0
tags:
  - bangla
  - bengali
  - banglish
  - tool-use
  - function-calling
  - multi-turn
  - agent
  - bangladesh
configs:
  - config_name: default
    default: true
    data_files:
      - split: train
        path: data/train/jontro_train.parquet
      - split: test
        path: data/test/jontro_test.parquet
  - config_name: sharegpt
    data_files:
      - split: train
        path: data/train/jontro_train_sharegpt.parquet
      - split: test
        path: data/test/jontro_test_sharegpt.parquet
---

# Jontro 2.0

Jontro (Bengali: যন্ত্র, "instrument") is a dataset of {c['n_trajectories']:,} multi-turn tool-use conversations in Bengali script
and Romanised Bangla ("Banglish"). The conversations cover {c['n_domains']} service domains grounded in Bangladesh, such as
healthcare, telecom and railway booking, and use {c['n_distinct_tools_offered']} tools ({c['n_distinct_tools_called']} of them are called at least once).
An assistant issues tool calls and a deterministic simulator returns the results, so every tool result can be regenerated exactly
from its seed. Each conversation has 2 to 5 user turns (mean {s['user_turns_mean']}), and calls are chained across turns.

This card describes version 2.0. The dataset and its benchmark are described in the paper "Jontro: A Multi-Turn Bangla and
Banglish Tool-Use Corpus and Benchmark" by Abu Bakar Siddik, Afroza Islam, and Sarafat Hussain Abhi (preprint); see the citation section below. A LoRA
adapter for Qwen3.5-2B fine-tuned on the training split is public at [{MODEL_ID}](https://huggingface.co/{MODEL_ID}).

## Relation to the earlier release

An earlier release of this repository contained 9,158 single-turn trajectories. It is kept under the git tag
`{ARCHIVE_TAG}`. Every record in it had exactly one user turn, so it was not a multi-turn dataset. Version 2.0 replaces it, and
results on the two releases should not be compared.

## Summary

| Property | Value |
|---|---|
| Trajectories | {c['n_trajectories']:,} (no duplicate ids, {c['n_seeds_distinct']:,} distinct seeds) |
| User, assistant and tool messages | {c['role_totals']['user']:,}, {c['role_totals']['assistant']:,} and {c['role_totals']['tool']:,} |
| Tool calls | {c['n_tool_calls']:,}, in {c['n_decision_points']:,} tool-calling decision points |
| Decision points with parallel calls | {c['n_multi_call_messages']:,} |
| Distinct task goals | {c['n_goals']} |
| Split | {sp['train']['n_records']:,} train and {sp['test']['n_records']:,} test, with no goal overlap |
| Trajectories with at least one tool call | {c['tool_use_rate_pct']:.0f}% |

## Composition

Trajectories per domain:

{table(s['domains'], ('Domain', 'Trajectories'))}

Users address the assistant with the intimate *tui*, the familiar *tumi* or the respectful *apni* form. All six combinations of
register and script are populated, including tui in Banglish, which earlier Bangla resources leave empty.

| Register | Bengali script | Banglish |
|---|---:|---:|
{grid}

Conversation endings (`metadata.closure_style`): {closures}; {s['closure_style'].get('unset', 0)} are unset.

## Format

The `default` config holds the canonical records with the fields `id`, `domain`, `persona`, `messages`, `tools` and `metadata`.
Tool arguments, tool parameter schemas and `metadata.sim_seeds` are stored as JSON strings so that nested keys survive the Parquet
round trip exactly; parse them with `json.loads`. The Parquet files were checked against the JSONL release and match on all {c['n_trajectories']:,} records.
The `sharegpt` config holds the same conversations in ShareGPT format. The `raw/` directory has the JSONL files, including an
OpenAI-style tool-calling version and a 100-record anchor set.

`persona` records the register, script, technical literacy, profession, age, location and system-prompt language
(Bangla in {s['system_prompt_language']['bangla']:,} records, English in {s['system_prompt_language']['english']:,}). `metadata` records the closure style,
the task goal, the scenario type, the generation seeds, how the conversation ended, the tools used and a `validation` block.

## Splits

`splits/` contains the id lists and the {sp['test']['n_goals']} held-out goals. The split holds out whole goals, so the test set measures
transfer to tasks not seen in training. Because many records share a goal, the effective sample size is the number of goals
({sp['test']['n_goals']} in the test split) and not the number of records ({sp['test']['n_records']}), and goal-clustered confidence intervals are appropriate.
The test split has {sp['test']['n_decision_points']:,} decision points. The evaluation code (teacher-forced and interactive protocols, with
argument values scored against gold values) is in the code repository. `baselines/` contains a gold-replay check, which scores
100% on every metric by construction, four non-learned reference baselines, three zero-shot Qwen3.5 base models (2B, 4B, 9B) and two
Qwen3.5-2B models fine-tuned on the training split (the fine-tuned models are not released), all scored with the teacher-forced protocol.

## Validation

Every record passes the schema check, the internal consistency check (including a byte-level replay of the simulator) and the
language-boundary heuristics. No record uses a pronoun from the wrong register, and none is written in Sadhu bhasha.
{vf['register_verb_flag']['true']:,} records carry an advisory flag for register-specific verb endings and {vf['register_imperative_flag']['true']:,} for imperatives.
These flags are recorded and were not used to filter, because short suffixes also end ordinary words.

Two checks were not completed. An LLM-judged naturalness rubric was specified but not run, so `judge_scored` is false for every
record. Human review covered a sub-sample and was done by one reviewer, so there is no inter-annotator agreement.

## Generation and limitations

- Both sides of each conversation were generated by a Gemini-family model driven by a persona-based orchestrator. The model
  identifier was not stored per record. The simulators, seeds, schemas and scoring are reproducible, but the text is not
  reproducible bit for bit.
- User turns are model-generated, so they reflect the generator's picture of a Bangladeshi user and not observed behaviour.
- The domains are narrow and the simulators are simple. There is no legal or medical advice setting, no multi-party conversation
  and no regional dialect such as Chittagonian or Sylheti.
- The {c['n_goals']} goals repeat across records (median {s['goals']['records_per_goal_median']:.0f} records per goal, maximum {s['goals']['records_per_goal_max']}), so the
  {c['n_trajectories']:,} trajectories are fewer independent tasks than the count suggests.
- Argument matching is strict. Banglish has no standard spelling, so two spellings of the same name count as different values.
- Every trajectory contains a tool call. A model fine-tuned only on this data has no examples of requests that should be answered
  without a tool, and in our own experiments such models call tools on unrelated requests. Mixing in ordinary conversation
  without tools is advisable when general behaviour matters.

## Ethics and licence

All people, phone numbers, accounts and records in the data are synthetic and no personal data was collected. Several domains
involve health, finance and government services. The data is meant for research on tool-using agents and is not a source of
advice in those areas.

The data is released under CC BY 4.0. You may use, share and adapt it, including commercially, provided you credit the author, link to the
licence and indicate any changes. This licence covers the corpus as released and does not override any terms that the provider of the
generating model attaches to its outputs; anyone training a commercial model on the text should check those terms too. `SHA256SUMS`
lists a checksum for every data file in this release so a copy can be checked against it.

## Citation

If you use Jontro, please cite the paper:

```bibtex
{BIBTEX}
```

The entry will be updated with the arXiv identifier once the preprint is posted. `CITATION.cff` gives the same paper as its preferred
citation. To refer to this release of the data itself, you can also cite:

```bibtex
{BIBTEX_DATA}
```
"""


def build() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for src, dst in COPY.items():
        if not Path(src).exists():
            raise FileNotFoundError(src)
        (OUT / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, OUT / dst)
    stats = json.loads(Path("outputs/stats/jontro2_stats.json").read_text())
    (OUT / "README.md").write_text(card(stats).replace("{BIBTEX}", BIBTEX).replace("{BIBTEX_DATA}", BIBTEX_DATA))
    (OUT / "CITATION.cff").write_text(CFF)
    sums = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name not in ("README.md", "SHA256SUMS"))
    (OUT / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(OUT)}\n" for p in sums))
    print(f"built {OUT} ({sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) / 1e6:.1f} MB)")


def push() -> None:
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    api = HfApi()
    info = api.dataset_info(REPO_ID)
    assert info.private, "refusing to push: repo is not private"
    existing = {s.rfilename for s in info.siblings}
    refs = api.list_repo_refs(REPO_ID, repo_type="dataset")
    if ARCHIVE_TAG not in {t.name for t in refs.tags}:
        api.create_tag(REPO_ID, tag=ARCHIVE_TAG, repo_type="dataset", tag_message="Single-turn v2 release (9,158) before Jontro 2.0")
        print(f"tagged {ARCHIVE_TAG}")
    new = {str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file()}
    ops = [CommitOperationDelete(path_in_repo=f) for f in sorted(existing - new) if f != ".gitattributes"]
    ops += [CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(OUT / f)) for f in sorted(new)]
    api.create_commit(REPO_ID, ops, repo_type="dataset", commit_message="Jontro 2.0: multi-turn release (replaces the archived single-turn v2)")
    assert api.dataset_info(REPO_ID).private, "repo became public"
    print(f"pushed {len(new)} files, deleted {len(existing - new)}; private={api.dataset_info(REPO_ID).private}")


def update_card(with_data: bool = False) -> None:
    """Update README.md (and optionally the Parquet files) on the public repo; move the v2.0 tag to the new head."""
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    paths = ["README.md", "CITATION.cff", "SHA256SUMS", "baselines/multiturn_scores.json", "baselines/multiturn_results.md",
             "baselines/multiturn_by_turn.png"]
    paths += [str(p.relative_to(OUT)) for p in sorted(OUT.glob("data/*/*.parquet"))] if with_data else []
    api.create_commit(REPO_ID, [CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(OUT / f)) for f in paths],
                      repo_type="dataset", commit_message="Update dataset card" + (" and Parquet files" if with_data else ""))
    tags = {t.name for t in api.list_repo_refs(REPO_ID, repo_type="dataset").tags}
    if "v2.0" in tags:
        api.delete_tag(REPO_ID, tag="v2.0", repo_type="dataset")
    api.create_tag(REPO_ID, tag="v2.0", repo_type="dataset", tag_message="Jontro 2.0")
    print("card updated; v2.0 moved to head")


def publish() -> None:
    """Content push (needs private), tag v2.0 on the pushed commit, then flip visibility. Irreversible in practice."""
    from huggingface_hub import HfApi

    push()
    api = HfApi()
    if "v2.0" not in {t.name for t in api.list_repo_refs(REPO_ID, repo_type="dataset").tags}:
        api.create_tag(REPO_ID, tag="v2.0", repo_type="dataset", tag_message="Jontro 2.0 public release")
    api.update_repo_settings(REPO_ID, repo_type="dataset", private=False)
    print(f"{REPO_ID} is now public; tags: {[t.name for t in api.list_repo_refs(REPO_ID, repo_type='dataset').tags]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--card", action="store_true", help="update README only on the already-public repo")
    ap.add_argument("--data", action="store_true", help="with --card, also upload the Parquet files")
    args = ap.parse_args()
    build()
    if args.card:
        update_card(args.data)
    elif args.publish:
        publish()
    elif args.push:
        push()
