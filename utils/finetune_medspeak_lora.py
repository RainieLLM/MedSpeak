#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MedSpeak LoRA finetune with histogram:
- Cleans dataset (rules-only system; user context; strict two-line assistant)
- Budgets KG sections (semantic/phonetic) so gold labels never get truncated
- Uses Llama 3.1 chat template
- Trains with HF Trainer (no TRL)
- Optional QLoRA 4-bit
- Prints token-length histogram and top-N longest samples for tuning
"""

import argparse, os, re, sys, torch
from typing import Dict, Any, List, Tuple
from collections import Counter
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# =========================
# Configurable system rules
# =========================
SYS_RULES = """You are MedSpeak, a medical ASR correction and QA assistant.

Output exactly two lines, no extra text.
Line 1 must start with: Corrected Text:
Line 2 must start with: Correct Option:

Rules:
- Do not repeat the question or options.
- Do not explain or add details.
- Do not use placeholders.
- The second line must be exactly in the form:
  Correct Option: <A|B|C|D>
  (where <A|B|C|D> is a single uppercase letter).
- Output nothing beyond those two lines.
""".strip()

# =========================
# CLI
# =========================
def parse_args():
    ap = argparse.ArgumentParser()
    # IO
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    # training
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_strategy", choices=["no", "steps", "epoch"], default="epoch")
    # quant
    ap.add_argument("--use_4bit", action="store_true", help="Enable QLoRA 4-bit")
    # LoRA
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.10)
    # KG budgets (tokens)
    ap.add_argument("--kg_sem_budget", type=int, default=600)
    ap.add_argument("--kg_phon_budget", type=int, default=300)
    # Histogram / inspection
    ap.add_argument("--show_hist", action="store_true", help="Print token length histogram before training")
    ap.add_argument("--top_longest", type=int, default=5, help="Show N longest samples (rendered token length)")
    return ap.parse_args()

# =========================
# Dataset cleaning
# =========================
def _clean_system(s: str) -> str:
    return SYS_RULES

def _clean_user(u: str) -> str:
    # Remove any trailing “follow the system instructions … Corrected Text …”
    u = re.split(r"\n\s*Follow the system instructions.*", u, flags=re.S)[0]
    return u.strip() + "\n"

def _normalize_assistant(a: str) -> str:
    # Force exactly two lines with exact prefixes
    lines = [ln.strip() for ln in a.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        ct_line = lines[0] if lines else ""
        co_line = "Correct Option: A"
    else:
        ct_line, co_line = lines[0], lines[1]
    ct_content = re.sub(r"^Corrected\s*Text\s*:\s*", "", ct_line, flags=re.I)
    co_content = re.sub(r"^Correct\s*Option\s*:\s*", "", co_line, flags=re.I)
    m = re.match(r"\s*([A-D])\s*$", co_content, flags=re.I)
    opt = m.group(1).upper() if m else (co_content[:1].upper() if co_content else "A")
    return f"Corrected Text: {ct_content.strip()}\nCorrect Option: {opt}\n"

def clean_messages(msgs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    for m in msgs:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            content = _clean_system(content)
        elif role == "user":
            content = _clean_user(content)
        elif role == "assistant":
            content = _normalize_assistant(content)
        out.append({"role": role, "content": content})
    return out

# =========================
# KG budgeting (user prompt)
# =========================
USER_SEP_PAT = re.compile(
    r"(?:^\[ASR_TEXT\]\s*)(?P<asr>.*?)(?:^\[OPTIONS\]\s*)(?P<opts>.*?)(?:^\[KG_SEMANTIC\]\s*)(?P<kgsem>.*?)(?:^\[KG_PHONETIC\]\s*)(?P<kgphon>.*)$",
    flags=re.S | re.M
)

def split_user_sections(user_text: str):
    m = USER_SEP_PAT.search(user_text)
    if not m:
        # If the sample doesn't follow the template, keep intact in ASR section
        return user_text.strip(), "", "", ""
    return m.group("asr").strip(), m.group("opts").strip(), m.group("kgsem").strip(), m.group("kgphon").strip()

def rebuild_user(asr: str, opts: str, kgsem: str, kgphon: str) -> str:
    parts = [
        "[ASR_TEXT]",
        asr.strip(),
        "",
        "[OPTIONS]",
        opts.strip(),
        "",
        "[KG_SEMANTIC]",
        kgsem.strip(),
        "",
        "[KG_PHONETIC]",
        kgphon.strip()
    ]
    return "\n".join(parts).strip() + "\n"

def truncate_by_token_budget(tok, text: str, max_tokens: int) -> str:
    ids = tok(text, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    clipped = tok.decode(ids[:max_tokens], skip_special_tokens=True)
    return clipped

def budget_user_kg(tok, user_text: str, kgsem_budget: int, kgphon_budget: int):
    asr, opts, kgsem, kgphon = split_user_sections(user_text)
    kgsem_small = truncate_by_token_budget(tok, kgsem, max(kgsem_budget, 0)) if kgsem else ""
    kgphon_small = truncate_by_token_budget(tok, kgphon, max(kgphon_budget, 0)) if kgphon else ""
    return rebuild_user(asr, opts, kgsem_small, kgphon_small)

def render_chat_with_budgets(tok, messages: list, max_len: int, kgsem_budget: int, kgphon_budget: int) -> str:
    msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
    uidx = next((i for i, m in enumerate(msgs) if m["role"] == "user"), None)
    if uidx is None:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

    # Try with successive budget reductions if still too long
    s_budget, p_budget = kgsem_budget, kgphon_budget
    text = ""
    for _ in range(5):
        msgs[uidx]["content"] = budget_user_kg(tok, msgs[uidx]["content"], s_budget, p_budget)
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        n_tokens = len(tok(text, add_special_tokens=False)["input_ids"])
        if n_tokens <= max_len:
            return text
        # Shrink budgets and retry
        s_budget = max(80, int(s_budget * 0.6))
        p_budget = max(40, int(p_budget * 0.6))
    return text  # last resort; tokenizer will truncate

# =========================
# Model / Tokenizer / LoRA
# =========================
def build_tokenizer(base_model: str):
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok

def build_model(args):
    if args.use_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, quantization_config=quant, device_map="auto", trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
    return model

def add_lora(model, args):
    lconf = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lconf)

# =========================
# Tokenization & histogram
# =========================
def tokenize_text(tok, text: str, max_len: int):
    return tok(
        text,
        truncation=True,
        max_length=max_len,
        padding=False,
        return_tensors=None,
    )

def tokenize_example(tok, max_len: int, kgsem_budget: int, kgphon_budget: int):
    def _fn(rec):
        text = render_chat_with_budgets(tok, rec["messages"], max_len, kgsem_budget, kgphon_budget)
        return tokenize_text(tok, text, max_len)
    return _fn

def compute_lengths(tok, ds, max_len: int, kgsem_budget: int, kgphon_budget: int) -> List[int]:
    lens = []
    for rec in ds:
        text = render_chat_with_budgets(tok, rec["messages"], max_len, kgsem_budget, kgphon_budget)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        lens.append(len(ids))
    return lens

def print_length_histogram(lengths: List[int], max_len: int, bins=(256, 512, 768, 1024, 1280, 1536, 1792, 2048)):
    cnt = Counter()
    for n in lengths:
        placed = False
        for b in bins:
            if n <= b:
                cnt[b] += 1
                placed = True
                break
        if not placed:
            cnt["over"] += 1
    total = sum(cnt.values())
    print("\n[INFO] Token length histogram (after budgeting, before truncation):")
    for b in bins:
        v = cnt.get(b, 0)
        pct = 100.0 * v / max(total, 1)
        marker = " <- many at cap" if (b == max_len and v > 0) else ""
        print(f"  <= {b:4d}: {v:6d}  ({pct:5.1f}%){marker}")
    if cnt.get("over"):
        v = cnt["over"]
        pct = 100.0 * v / max(total, 1)
        print(f"   > max: {v:6d}  ({pct:5.1f}%)  <- overflowing even after budgets")
    print()

def show_top_longest(tok, ds, lengths: List[int], top_k: int, max_len: int, kgsem_budget: int, kgphon_budget: int):
    if top_k <= 0: return
    idxs = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)[:top_k]
    print(f"[INFO] Top {top_k} longest samples (rendered length):")
    for rank, i in enumerate(idxs, 1):
        text = render_chat_with_budgets(tok, ds[i]["messages"], max_len, kgsem_budget, kgphon_budget)
        print(f"\n  #{rank}  tokens={lengths[i]}")
        # Print the first ~600 chars to avoid flooding
        preview = text[:600].replace("\n", "\\n")
        print(f"  preview: {preview}...")
    print()

# =========================
# Main
# =========================
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    # Load raw dataset
    ds = load_dataset("json", data_files=args.train_jsonl, split="train")

    # Clean messages in-memory
    def _clean_record(rec):
        if "messages" in rec:
            rec["messages"] = clean_messages(rec["messages"])
        return rec
    ds = ds.map(_clean_record, desc="Cleaning messages")

    # Tokenizer/model/LoRA
    tok = build_tokenizer(args.base_model)
    model = add_lora(build_model(args), args)

    # Optional: preview one sample (post-clean, pre-budget)
    try:
        print("===== Rendered sample (post-clean & pre-budget) =====")
        print(tok.apply_chat_template(ds[0]["messages"], tokenize=False, add_generation_prompt=False))
        print("=====================================================")
    except Exception as e:
        print(f"[WARN] Render sample failed: {e}", file=sys.stderr)

    # Histogram & top-N
    if args.show_hist or args.top_longest > 0:
        lengths = compute_lengths(tok, ds, args.max_seq_len, args.kg_sem_budget, args.kg_phon_budget)
        if args.show_hist:
            print_length_histogram(lengths, args.max_seq_len)
        if args.top_longest > 0:
            show_top_longest(tok, ds, lengths, args.top_longest, args.max_seq_len, args.kg_sem_budget, args.kg_phon_budget)

    # Tokenize with KG budgets
    keep_cols = ("messages",)
    cols_to_remove = [c for c in ds.column_names if c not in keep_cols]
    tok_ds = ds.map(
        tokenize_example(tok, args.max_seq_len, args.kg_sem_budget, args.kg_phon_budget),
        remove_columns=cols_to_remove,
        desc="Tokenizing with budgets"
    )

    # Collator (creates padded labels for causal LM)
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    # Trainer
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to=[],
        seed=args.seed,
        group_by_length=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_ds,
        data_collator=collator,
        tokenizer=tok,
    )

    trainer.train()

    # Save LoRA adapter + tokenizer
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"[DONE] LoRA adapter + tokenizer saved to {args.out_dir}")

if __name__ == "__main__":
    main()
