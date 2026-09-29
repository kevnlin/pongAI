"""LoRA fine-tune of Qwen3-VL on the stroke strips from training/build_vlm_dataset.py.

    python -m training.train_vlm_lora --epochs 3                       # GPU, bf16, ~1 A100
    python -m training.train_vlm_lora --merge weights/vlm/qwen3vl8b_strokes/best   # write merged model for vLLM

Only the language model gets LoRA adapters; the vision encoder stays frozen. The loss covers the
JSON answer tokens only. After every epoch the adapter is scored on the val split (game_5) by greedy
generation with the same per-target accuracy as scripts/eval_vlm.py; the best epoch is kept as
<out>/best. A valmirror split (build_vlm_dataset.py --mirror) is scored too, for left-handers only. Serve it with vLLM (--enable-lora, or --merge first) and run scripts/eval_vlm.py.
Needs torch, transformers>=4.57, peft (not in requirements.txt; use a separate venv).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, get_cosine_schedule_with_warmup

from pongai.config import ROOT, WEIGHTS_DIR

TARGETS = ("hand", "technique", "lean", "feet")
LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def load_rows(data: Path, split: str) -> list[dict]:
    rows = []
    for f in sorted(data.glob(f"{split}_*.jsonl")):
        rows += [json.loads(line) for line in f.read_text().splitlines() if line]
    return rows


def messages(row: dict, with_answer: bool) -> list[dict]:
    # text before image, as pongai.coach.llm.classify_stroke sends it
    msgs = [{"role": "user", "content": [{"type": "text", "text": row["prompt"]}, {"type": "image"}]}]
    if with_answer:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": row["answer"]}]})
    return msgs


def encode(processor, row: dict, data: Path, device) -> dict:
    img = Image.open(data / row["image"]).convert("RGB")
    prompt = processor.apply_chat_template(messages(row, False), add_generation_prompt=True, tokenize=False)
    full = processor.apply_chat_template(messages(row, True), tokenize=False)
    assert full.startswith(prompt), "chat template does not extend the generation prompt"
    n_prompt = processor(text=[prompt], images=[img], return_tensors="pt")["input_ids"].shape[1]
    batch = processor(text=[full], images=[img], return_tensors="pt")
    labels = batch["input_ids"].clone()
    labels[:, :n_prompt] = -100
    batch["labels"] = labels
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def validate(model, processor, rows: list[dict], data: Path, device) -> dict:
    model.eval()
    hits = {t: 0 for t in TARGETS}
    bad_json = 0
    for row in rows:
        img = Image.open(data / row["image"]).convert("RGB")
        prompt = processor.apply_chat_template(messages(row, False), add_generation_prompt=True, tokenize=False)
        batch = processor(text=[prompt], images=[img], return_tensors="pt").to(device)
        out = model.generate(**batch, max_new_tokens=60, do_sample=False)
        text = processor.decode(out[0, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        try:
            pred = json.loads(text)
        except json.JSONDecodeError:
            pred, bad_json = {}, bad_json + 1
        gt = json.loads(row["answer"])
        for t in TARGETS:
            hits[t] += str(pred.get(t, "")).lower().strip() == gt[t]
    model.train()
    return {**{f"{t}_acc": round(hits[t] / len(rows), 4) for t in TARGETS}, "bad_json": bad_json, "n": len(rows)}


def merge(adapter: Path, model_id: str, out: Path):
    base = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    merged.save_pretrained(out)
    AutoProcessor.from_pretrained(model_id).save_pretrained(out)
    print(f"merged -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--data", default=str(ROOT / "datasets" / "vlm_strokes"))
    ap.add_argument("--out", default=str(WEIGHTS_DIR / "vlm" / "qwen3vl8b_strokes"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--merge", default=None, help="adapter dir: write a merged model to <adapter>_merged and exit")
    args = ap.parse_args()
    if args.merge:
        merge(Path(args.merge), args.model, Path(f"{args.merge.rstrip('/')}_merged"))
        return

    torch.manual_seed(args.seed)
    data, out, device = Path(args.data), Path(args.out), torch.device("cuda")
    train, val, valm = load_rows(data, "train"), load_rows(data, "val"), load_rows(data, "valmirror")
    print(f"{len(train)} train / {len(val)} val / {len(valm)} valmirror examples", flush=True)

    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16,
                                                            attn_implementation="sdpa").to(device)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps = math.ceil(len(train) / args.grad_accum) * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.03 * steps)), steps)

    history, best = [], -1.0
    model.train()
    for epoch in range(1, args.epochs + 1):
        random.Random(args.seed + epoch).shuffle(train)
        t0, running = time.time(), 0.0
        for i, row in enumerate(train, 1):
            loss = model(**encode(processor, row, data, device)).loss
            (loss / args.grad_accum).backward()
            running += loss.item()
            if i % args.grad_accum == 0 or i == len(train):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if i % 200 == 0:
                print(f"epoch {epoch} {i}/{len(train)} loss {running / 200:.4f} ({time.time() - t0:.0f}s)", flush=True)
                running = 0.0
        model.save_pretrained(out / f"epoch{epoch}")
        scores = validate(model, processor, val, data, device)
        mean = sum(scores[f"{t}_acc"] for t in TARGETS) / len(TARGETS)
        history.append({"epoch": epoch, **scores, "mean_acc": round(mean, 4)})
        if valm:  # left-handed check only; the best epoch is still chosen on the real val strokes
            history[-1]["valmirror"] = validate(model, processor, valm, data, device)
        print(f"epoch {epoch} val: {json.dumps(history[-1])}", flush=True)
        if mean > best:
            best = mean
            shutil.rmtree(out / "best", ignore_errors=True)
            shutil.copytree(out / f"epoch{epoch}", out / "best")
    (out / "train_log.json").write_text(json.dumps({"args": vars(args), "val": history}, indent=2))
    print(f"best val mean acc {best:.4f}; adapter at {out / 'best'}")


if __name__ == "__main__":
    main()
