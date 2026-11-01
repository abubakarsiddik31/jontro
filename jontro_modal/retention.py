"""Retention evaluation: generate every suite for base and each adapter, and measure drift.

Runs inside the container. See retention/PROTOCOL.md (gates frozen before any adapter was run).
One model load; adapters are attached with PEFT and the base arm uses ``disable_adapter()``, so
base and tuned arms see identical weights apart from the adapter. Greedy decoding, thinking off.
"""
from __future__ import annotations

import json
from pathlib import Path

from jontro_modal.arms import adapter_name, attach, label

DATA = Path("/retention")
OUT = Path("/out/retention")
RELEASE = Path("/data/jontro_multiturn_2000_openai.jsonl")
TRAIN_IDS = Path("/data/train_ids.jsonl")
TOOL_DOMAINS = ("healthcare", "telecom", "ecommerce")


def _read(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _tool_contexts() -> list[tuple[str, list[dict]]]:
    """(system_prompt, tools) from the first train trajectory of each domain: realistic Jontro setup."""
    train = {json.loads(line)["id"] for line in TRAIN_IDS.open()}
    seen: dict[str, tuple[str, list[dict]]] = {}
    for line in RELEASE.open():
        r = json.loads(line)
        d = r["domain"]
        if r["id"] in train and d in TOOL_DOMAINS and d not in seen:
            seen[d] = (r["messages"][0]["content"], r["tools"])
    return [seen[d] for d in TOOL_DOMAINS]


class Generator:
    def __init__(self, model, tok) -> None:
        import torch

        self.m, self.tok, self.torch = model, tok, torch
        tok.padding_side = "left"

    def generate(self, items: list[dict], max_new: int, bs: int = 16, rep_penalty: float = 1.0) -> list[dict]:
        """items: {messages, tools?}. Returns {output, finished, new_ids, prompt_ids} per item."""
        torch, tok = self.torch, self.tok
        outs: list[dict] = []
        for i in range(0, len(items), bs):
            outs.extend(self._generate_chunk(items[i:i + bs], max_new, rep_penalty))
        return outs

    def _generate_chunk(self, chunk: list[dict], max_new: int, rep_penalty: float) -> list[dict]:
        """One batch; on CUDA OOM split the batch in half (down to 1) so long prompts never kill a run."""
        torch, tok = self.torch, self.tok
        try:
            texts = [tok.apply_chat_template(c["messages"], tools=c.get("tools"), add_generation_prompt=True,
                                             tokenize=False, enable_thinking=False) for c in chunk]
            enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                gen = self.m.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                      repetition_penalty=rep_penalty,
                                      pad_token_id=tok.pad_token_id or tok.eos_token_id)
        except torch.OutOfMemoryError:
            if len(chunk) == 1:
                raise
            torch.cuda.empty_cache()
            h = len(chunk) // 2
            return self._generate_chunk(chunk[:h], max_new, rep_penalty) + self._generate_chunk(chunk[h:], max_new, rep_penalty)
        outs = []
        for row, ids in enumerate(gen[:, enc["input_ids"].shape[1]:]):
            ids = ids.tolist()
            finished = tok.eos_token_id in ids or (tok.pad_token_id in ids if tok.pad_token_id else False)
            keep = []
            for t in ids:
                if t in (tok.eos_token_id, tok.pad_token_id):
                    break
                keep.append(t)
            prompt_ids = enc["input_ids"][row][enc["attention_mask"][row].bool()].tolist()
            outs.append({"output": tok.decode(keep, skip_special_tokens=False).strip(), "finished": finished,
                         "new_ids": keep, "prompt_ids": prompt_ids})
        return outs

    def drift(self, peft_model, base_gen: list[dict]) -> dict:
        """KL(base||tuned) and NLL of base's own answers, per answer token, adapter on vs off."""
        torch = self.torch
        kls, nll_t, nll_b, n = 0.0, 0.0, 0.0, 0
        for g in base_gen:
            if not g["new_ids"]:
                continue
            ids = torch.tensor([g["prompt_ids"] + g["new_ids"]], device="cuda")
            k = len(g["new_ids"])
            tgt = ids[0, -k:]
            with torch.no_grad():
                lt = peft_model(input_ids=ids).logits[0, -k - 1:-1].float()
                with peft_model.disable_adapter():
                    lb = peft_model(input_ids=ids).logits[0, -k - 1:-1].float()
            lpt, lpb = lt.log_softmax(-1), lb.log_softmax(-1)
            kls += (lpb.exp() * (lpb - lpt)).sum(-1).sum().item()
            nll_t += -lpt.gather(1, tgt[:, None]).sum().item()
            nll_b += -lpb.gather(1, tgt[:, None]).sum().item()
            n += k
        return {"kl_base_to_tuned": kls / n, "nll_tuned": nll_t / n, "nll_base": nll_b / n, "tokens": n}


def heldout_nll(gen: "Generator", pm, rows: list[dict], max_reply: int = 256) -> dict:
    """Mean per-token NLL of human-written replies under the tuned model and under the base (adapter disabled).

    A forgetting measure that does not depend on the base model's own outputs: if tuning damaged language modelling
    in Bengali or English, the tuned NLL rises above the base NLL on text neither model trained on.
    """
    torch, tok = gen.torch, gen.tok
    out = {}
    for lang in ("bn", "en"):
        nt = nb = n = 0.0
        for r in (x for x in rows if x["lang"] == lang):
            prompt = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], add_generation_prompt=True,
                                             tokenize=True, return_dict=False, enable_thinking=False)
            reply = tok(r["reply"], add_special_tokens=False)["input_ids"][:max_reply]
            ids = torch.tensor([prompt + reply], device="cuda")
            k = len(reply)
            tgt = ids[0, -k:]
            with torch.no_grad():
                lt = pm(input_ids=ids).logits[0, -k - 1:-1].float().log_softmax(-1)
                with pm.disable_adapter():
                    lb = pm(input_ids=ids).logits[0, -k - 1:-1].float().log_softmax(-1)
            nt += -lt.gather(1, tgt[:, None]).sum().item()
            nb += -lb.gather(1, tgt[:, None]).sum().item()
            n += k
        out[f"heldout_{lang}_nll_tuned"], out[f"heldout_{lang}_nll_base"] = nt / n, nb / n
        out[f"heldout_{lang}_nll_delta"] = (nt - nb) / n
    return out


def run_retention(base_model: str, arms: list[str]) -> dict:
    """arms: run names under /out/adapters (each holds a PEFT adapter). 'base' is always evaluated."""
    import mlflow
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, device_map={"": 0})
    pm = attach(base, arms)
    pm.eval()
    gen = Generator(pm, tok)

    tqa = _read(DATA / "truthfulqa48_questions.jsonl")
    chat = _read(DATA / "general_chat.jsonl")
    trs = _read(DATA / "tool_restraint.jsonl")
    heldout = _read(DATA / "heldout_text.jsonl")
    ctx = _tool_contexts()
    suites = {
        "truthfulqa48": ([{"messages": [{"role": "user", "content": q["prompt"]}]} for q in tqa], tqa, 8),
        "general_chat": ([{"messages": [{"role": "user", "content": r["prompt"]}]} for r in chat], chat, 300),
        "tool_restraint": ([{"messages": [{"role": "system", "content": ctx[i % 3][0]},
                                          {"role": "user", "content": r["prompt"]}],
                             "tools": ctx[i % 3][1]} for i, r in enumerate(trs)], trs, 200),
    }
    mlflow.set_tracking_uri("sqlite:////out/mlflow-retention.db")
    mlflow.set_experiment("jontro-retention")
    summary: dict[str, dict] = {}
    base_chat_gen: list[dict] = []
    for raw_arm in ["base", *arms]:
        arm = raw_arm if raw_arm == "base" else raw_arm  # adapter lookup uses raw name, files use label()
        rows, results = [], {}
        ctxmgr = pm.disable_adapter() if arm == "base" else _Null()
        if arm != "base":
            pm.set_adapter(adapter_name(raw_arm))
        with ctxmgr:
            for name, (items, meta, max_new) in suites.items():
                g = gen.generate(items, max_new)
                if arm == "base" and name == "general_chat":
                    base_chat_gen = g
                for m, o in zip(meta, g):
                    row = {"arm": label(arm), "suite": name, "id": m.get("id") or m["item_id"],
                           "prompt": m.get("prompt"), "output": o["output"], "finished": o["finished"]}
                    if name == "truthfulqa48":
                        row["item_id"] = m["item_id"]
                    rows.append(row)
        if arm != "base":
            results = gen.drift(pm, base_chat_gen)
            results.update(heldout_nll(gen, pm, heldout))
        (OUT / f"{label(arm)}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
        summary[label(arm)] = results
        with mlflow.start_run(run_name=f"retention-{label(arm)}"):
            mlflow.set_tags({"experiment": "jontro-retention", "smoke": "false", "kind": "retention"})
            mlflow.log_params({"base_model": base_model, "arm": label(arm), "decoding": "greedy, thinking off"})
            mlflow.log_metrics({k: v for k, v in results.items() if isinstance(v, (int, float))})
    return summary


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def run_decode_check(base_model: str, arms: list[str], rep_penalty: float = 1.15) -> dict:
    """Re-generate the general-chat suite with a repetition penalty (no training, no drift metrics).

    Separates repetition-loop degeneration under greedy decoding from a genuine loss of Bengali ability.
    Writes /out/retention_rep/<arm>.jsonl in the same row format the Gemini judge reads.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path("/out/retention_rep")
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, device_map={"": 0})
    pm = attach(base, arms)
    pm.eval()
    gen = Generator(pm, tok)
    chat = _read(DATA / "general_chat.jsonl")
    items = [{"messages": [{"role": "user", "content": r["prompt"]}]} for r in chat]
    summary = {}
    for arm in ["base", *arms]:
        if arm != "base":
            pm.set_adapter(adapter_name(arm))
        with (pm.disable_adapter() if arm == "base" else _Null()):
            g = gen.generate(items, 300, rep_penalty=rep_penalty)
        rows = [{"arm": label(arm), "suite": "general_chat", "id": m["id"], "prompt": m["prompt"],
                 "output": o["output"], "finished": o["finished"]} for m, o in zip(chat, g)]
        (out / f"{label(arm)}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
        summary[label(arm)] = {"unfinished": sum(not r["finished"] for r in rows)}
    return summary
