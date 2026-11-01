# Retention protocol: does Jontro fine-tuning preserve general behaviour?

**Frozen before any adapter was evaluated.** Modelled on the MSc-thesis utility gate (same prompts and
decoding for base and tuned arms, blinded capability packet, fixed rubric, gates stated up front).

## Arms
`base` (no adapter) versus each Jontro adapter, same base weights, greedy decoding, thinking off.

## Suites (all in `retention/data/`)
| Suite | Items | Measures |
|---|---|---|
| `truthfulqa48` | 48 A/B forced-choice, thesis packet | factual capability; strict scoring (stripped output must be exactly `A` or `B`, anything else counts wrong in the 48 denominator) |
| `general_chat.jsonl` | 100 no-tool prompts (50 Bangla/Banglish, 50 English) | coherence (LLM judge, thesis rubric, 0-100), reply-language match, degenerate-output rate |
| `tool_restraint.jsonl` | 30 unrelated/chit-chat requests under a Jontro system prompt with tools offered | spurious tool-call rate |
| drift (computed inside the generation job) | the base model's own answers to `general_chat` | KL(base \|\| tuned) per token and tuned-vs-base NLL on those answers; continuous, no gate |

## Gates (a tuned arm passes retention only if all hold)
- **G1 coherence:** mean judged coherence on `general_chat` is at most 10 points below base (thesis D32 gate).
- **G2 capability:** TruthfulQA-48 strict correct is at most 5 items below base. 48 items is a diagnostic,
  not a benchmark; report paired counts, never claim causal knowledge loss from it.
- **G3 tool restraint:** spurious tool-call rate is at most base + 5 percentage points.
- **G4 language:** reply-language match is at most 5 percentage points below base.

## If a gate fails
Do not tune on the test split. In order: (1) scale the adapter below 1.0 at inference, (2) train a
`full_mix` variant (about 20% self-distilled general data: base model answers to general prompts),
(3) lower LR / epochs. Report the failure and the trade-off either way; never relabel a failed gate.

## Judge
Non-Gemini family (Gemini generated the corpus): default `openai/gpt-4o-mini` via OpenRouter, temperature 0,
rubric copied verbatim from the thesis `e1_judge.py`. Judge errors stay in the denominator.

## Amendment 1 (2026-10-18, before any `full_mix` result existed)
`tool_restraint.jsonl` grows from 30 to **100** prompts (`tr-00`..`tr-29` unchanged; `tr-30`..`tr-99` added) because 30 prompts gave
about +/-9 points of noise. The original 30 results for `base`, the `full` checkpoints and the final `full` adapter were measured
on the 30-prompt set and are reported as such; all comparisons that involve `full_mix` use the 100-prompt set, with `base` and
`full` re-evaluated on it. G3 is unchanged (<= base + 5 points). The new prompts are authored and disjoint from the training
chit-chat and from all sourced mix prompts (near-duplicate filter in `scripts/source_mix_prompts.py`).

## Amendment 2 (2026-10-18)
The coherence judge is Gemini (`gemini-3.5-flash`, temperature 0, thinking off, rubric unchanged), chosen because Gemini credits are available. This
differs from the original plan of a non-Gemini judge. Gemini also generated the Jontro corpus, so a shared-family bias toward fluent Gemini-like
text is possible; absolute scores matter less than the base-versus-tuned gap, and the gap is what G1 tests. Judge errors stay in the denominator.
