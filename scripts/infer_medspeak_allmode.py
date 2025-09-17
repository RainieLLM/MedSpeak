#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MedSpeak inference (all modes) — v2
- Lazy PEFT import (only if --lora_dir looks valid). If PEFT fails, continue w/o LoRA.
- Robust multi-GPU: bounded queues + blocking put(timeout=5) with backoff
- Async writer (optional) with batched fsync
- Periodic merge snapshot to <preds_out>.live; final canonical merge to preds_out
- Timeouts + small retry budget; resume; merge subcommand

Modes (via --input_mode / --use_kg):
  1) Zero-shot GT + LLM:  --base_model <base>   --input_mode text
  2) Whisper + LLM:       --base_model <base>   --input_mode audio
  3) MedSpeak (ASR+KG):   --base_model <ft>     --input_mode audio --use_kg
  4) FT + GT (+/-KG):     --base_model <ft>     --input_mode text [--use_kg]

Manifest needs:
- audio modes:  audio_path, options OR (option_a..d), answer
- text  modes:  text_gt,   options OR (option_a..d), answer
"""

import argparse, csv, json, os, re, sqlite3, sys, time, signal, threading, glob
import multiprocessing as mp
from typing import List, Dict, Any, Set, Tuple
from queue import Empty
from collections import deque

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")  # must be set before importing transformers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

# put near top, after the env vars
try:
    from transformers import LogitsProcessor, LogitsProcessorList
except Exception:
    class LogitsProcessor:
        def __call__(self, input_ids, scores):
            return scores
    class LogitsProcessorList(list):
        def __call__(self, input_ids, scores):
            for p in self:
                scores = p(input_ids, scores)
            return scores



DEFAULT_KG_SEM_BUDGET = 600
DEFAULT_KG_PHON_BUDGET = 300
DEFAULT_MAX_NEW_TOKENS_TEXT = 96

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

# ---------- utils ----------
def norm(s: str) -> str:
    return str(s).strip().lower() if s is not None else ""

def shortlist_terms(text: str, k: int = 40):
    toks = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text)
    uniq, seen = [], set()
    for t in toks:
        t = norm(t)
        if t not in seen:
            seen.add(t); uniq.append(t)
        if len(uniq) >= k:
            break
    return uniq

def query_semantic(conn, terms, limit_per=5):
    out = []
    for t in terms:
        q = "SELECT term, rel, rel_detail, related_term FROM kg WHERE term LIKE ? OR related_term LIKE ? LIMIT ?"
        for term, rel, rel_detail, related in conn.execute(q, (f"%{t}%", f"%{t}%", limit_per)):
            if rel_detail:
                out.append(f"{term} -[ {rel}:{rel_detail} ]-> {related}")
            else:
                out.append(f"{term} -[ {rel} ]-> {related}")
    return out[:100]

def load_phonetic(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            try: items.append(json.loads(s))
            except: pass
    return items

def filter_phonetic(all_items, terms, limit=80):
    tset = set(terms); out = []
    for it in all_items:
        if norm(it.get("term","")) in tset:
            cui = it.get("cui")
            out.append(f"{it['term']} ~ {it['similar']}{(' ('+cui+')') if cui else ''}")
        if len(out) >= limit: break
    return out

USER_SEP_PAT = re.compile(
    r"(?:^\[ASR_TEXT\]\s*)(?P<asr>.*?)(?:^\[OPTIONS\]\s*)(?P<opts>.*?)(?:^\[KG_SEMANTIC\]\s*)(?P<kgsem>.*?)(?:^\[KG_PHONETIC\]\s*)(?P<kgphon>.*)$",
    flags=re.S | re.M
)

def truncate_by_token_budget(tok, text: str, max_tokens: int) -> str:
    if not text or max_tokens <= 0: return ""
    ids = tok(text, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)["input_ids"]
    if len(ids) <= max_tokens: return text
    return tok.decode(ids[:max_tokens], skip_special_tokens=True)

def budget_user_kg(tok, user_prompt: str, kg_sem_budget: int, kg_phon_budget: int) -> str:
    m = USER_SEP_PAT.search(user_prompt)
    if not m: return user_prompt
    asr = m.group("asr").strip()
    opts = m.group("opts").strip()
    kgsem = m.group("kgsem").strip()
    kgphon = m.group("kgphon").strip()
    kgsem_small = truncate_by_token_budget(tok, kgsem, kg_sem_budget) if kgsem else ""
    kgphon_small = truncate_by_token_budget(tok, kgphon, kg_phon_budget) if kgphon else ""
    parts = [
        "[ASR_TEXT]", asr, "",
        "[OPTIONS]", opts, "",
        "[KG_SEMANTIC]", kgsem_small if kgsem_small else "(none)", "",
        "[KG_PHONETIC]", kgphon_small if kgphon_small else "(none)",
    ]
    return "\n".join(parts).strip() + "\n"

def render_chat(tok, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def extract_corrected_line(text: str) -> str:
    for ln in text.splitlines():
        s = ln.strip()
        if s.lower().startswith("corrected text:"):
            return "Corrected Text: " + s.split(":",1)[1].strip()
    return "Corrected Text:"

# ---------- constrained letter decode ----------
class OnlyAllowTokens(LogitsProcessor):
    def __init__(self, allowed_ids: List[int], penalty: float = 1e9):
        super().__init__()
        self.allowed = set(allowed_ids); self.penalty = penalty
    def __call__(self, input_ids, scores):
        mask = (scores * 0.0) - self.penalty
        mask[:, list(self.allowed)] = 0.0
        return scores + mask

def force_letter_generate(model, tok, prompt: str, letters=("A","B","C","D")) -> str:
    allowed = []
    for L in letters:
        for s in [L, " "+L]:
            ids = tok(s, add_special_tokens=False).input_ids
            if len(ids) == 1:
                allowed.append(ids[0])
    nl_ids = tok("\n", add_special_tokens=False).input_ids
    if len(nl_ids) == 1: allowed.append(nl_ids[0])

    input_ids = tok(prompt, return_tensors="pt").to(model.device)
    if not allowed:
        with torch.no_grad():
            out = model.generate(**input_ids, max_new_tokens=1, do_sample=False)
        gen = tok.decode(out[0][input_ids["input_ids"].shape[1]:], skip_special_tokens=True)
        m = re.search(r"[ABCD]", gen)
        return m.group(0) if m else ""

    processors = LogitsProcessorList([OnlyAllowTokens(list(set(allowed)))])
    with torch.no_grad():
        out = model.generate(
            **input_ids,
            logits_processor=processors,
            max_new_tokens=1,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            eos_token_id=tok.eos_token_id,
        )
    gen = tok.decode(out[0][input_ids["input_ids"].shape[1]:], skip_special_tokens=True)
    m = re.search(r"[ABCD]", gen)
    return m.group(0) if m else ""

# ---------- timeouts ----------
class StepTimeout(Exception): pass

def with_timeout(seconds: float, fn, *a, **kw):
    result: Dict[str, Any] = {}
    def target():
        try: result["value"] = fn(*a, **kw)
        except Exception as e: result["error"] = e
    th = threading.Thread(target=target, daemon=True)
    th.start(); th.join(seconds)
    if th.is_alive(): raise StepTimeout(f"step exceeded {seconds}s")
    if "error" in result: raise result["error"]
    return result.get("value")

# ---------- worker ----------
def worker_main(task_q: mp.Queue, result_q: mp.Queue, args, gpu_id: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import yaml

    print(f"[INIT][GPU {gpu_id}] Worker starting. Loading ASR & model...", flush=True)

    # ASR
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.asr_backend == "faster-whisper":
        from faster_whisper import WhisperModel
        compute = "float16" if device == "cuda" else "int8"
        asr_tuple: Tuple[str, Any] = ("faster", WhisperModel(args.asr_model, device=device, compute_type=compute))
    else:
        import whisper
        kw = {}
        if args.whisper_cache: kw["download_root"] = args.whisper_cache
        asr_tuple = ("openai", whisper.load_model(args.asr_model, device=device, **kw))

    def transcribe(asr_t, audio_path: str) -> str:
        b, engine = asr_t
        if b == "faster":
            segs, _ = engine.transcribe(audio_path, language="en")
            return "".join(s.text for s in segs).strip()
        else:
            res = engine.transcribe(audio_path, language="en")
            return res["text"].strip()

    # Tokenizer & model
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"

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
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

    # Optional LoRA (lazy import; non-fatal on failure)
    def looks_like_lora_dir(p: str) -> bool:
        if not p or not os.path.isdir(p): return False
        has_conf = os.path.exists(os.path.join(p, "adapter_config.json"))
        has_wt = any(os.path.exists(os.path.join(p, fn)) for fn in ("adapter_model.safetensors","adapter_model.bin"))
        return has_conf and has_wt

    lora_info = "No LoRA (full model only)"
    if args.lora_dir and looks_like_lora_dir(args.lora_dir):
        try:
            from peft import PeftModel  # lazy import
            model = PeftModel.from_pretrained(model, args.lora_dir, local_files_only=os.path.isdir(args.lora_dir))
            lora_info = "LoRA applied"
        except Exception as e:
            print(f"[WARN][GPU {gpu_id}] Failed to load LoRA '{args.lora_dir}': {type(e).__name__}: {e}. "
                  f"Continuing without LoRA.", flush=True)

    model.eval()
    print(f"[INIT][GPU {gpu_id}] Model ready. {lora_info}. ASR={args.asr_backend}:{args.asr_model}", flush=True)

    # KG & prompt
    conn = sqlite3.connect(args.kg_sql)
    phon_all = load_phonetic(args.kg_phon)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    system_prompt = cfg.get("system_prompt", SYS_RULES)

    # Shard file
    os.makedirs(args.shard_out_dir, exist_ok=True)
    shard_path = os.path.join(args.shard_out_dir, f"shard_gpu{gpu_id}_pid{os.getpid()}.jsonl")
    print(f"[INIT][GPU {gpu_id}] Shard path: {os.path.abspath(shard_path)}", flush=True)

    processed = 0
    t0 = time.time()
    last_hb = time.time()

    def transcribe_with_timeout(audio_path: str) -> str:
        return with_timeout(args.asr_timeout, transcribe, asr_tuple, audio_path)

    def generate_with_timeout(chat_prefix: str):
        inp = tok(chat_prefix, return_tensors="pt").to(model.device)
        def _gen():
            with torch.no_grad():
                return model.generate(
                    **inp,
                    max_new_tokens=args.max_new_tokens_text,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    eos_token_id=tok.eos_token_id,
                    max_time=args.gen_max_time,
                )
        return with_timeout(max(1.0, args.gen_max_time + 2.0), _gen)

    def send_result(payload):
        sent, attempts = False, 0
        while not sent:
            try:
                result_q.put(payload, block=True, timeout=5.0)
                sent = True
            except Exception:
                attempts += 1
                if attempts % 6 == 0 and payload is not None:
                    print(f"[GPU {gpu_id}] waiting to send idx={payload[0]} (result_q full)...", flush=True)
                time.sleep(min(0.5 * attempts, 3.0))

    INPUT_MODE = args.input_mode   # "audio" | "text"
    USE_KG    = bool(args.use_kg)
    prompt_template = cfg["prompt_template"]

    while True:
        try:
            item = task_q.get()
        except (EOFError, KeyboardInterrupt):
            send_result(None); break
        if item is None:
            send_result(None); break

        idx, row = item
        attempt = 0

        while True:
            try:
                now = time.time()
                if now - last_hb > 30:
                    print(f"[HEARTBEAT][GPU {gpu_id}] alive; processed={processed}", flush=True)
                    last_hb = now

                # Options
                if "options" in row and row["options"]:
                    parts = [p.strip() for p in row["options"].split("||")]
                else:
                    parts = [
                        f"Option A: {row['option_a']}",
                        f"Option B: {row['option_b']}",
                        f"Option C: {row['option_c']}",
                        f"Option D: {row['option_d']}",
                    ]
                if len(parts) != 4:
                    raise ValueError("Row options malformed (need 4 parts)")
                optA = parts[0].split(":",1)[-1].strip()
                optB = parts[1].split(":",1)[-1].strip()
                optC = parts[2].split(":",1)[-1].strip()
                optD = parts[3].split(":",1)[-1].strip()

                # Input text
                if INPUT_MODE == "text":
                    asr_text = (row.get("text_gt") or "").strip()
                    if not asr_text:
                        raise ValueError("input_mode=text but 'text_gt' is empty")
                else:
                    asr_text = transcribe_with_timeout(row["audio_path"])
                print(f"[GPU {gpu_id}] idx={idx} INPUT: {asr_text[:140]}...", flush=True)

                # KG optional
                if USE_KG:
                    terms = shortlist_terms(asr_text + "\n" + "\n".join([optA,optB,optC,optD]))
                    sem = query_semantic(conn, terms)
                    phon = filter_phonetic(phon_all, terms)
                    kg_sem_str = "; ".join(sem) if sem else "(none)"
                    kg_phon_str = "; ".join(phon) if phon else "(none)"
                else:
                    kg_sem_str = "(none)"; kg_phon_str = "(none)"

                user_prompt_raw = prompt_template.format(
                    asr_text=asr_text,
                    opt_a=optA, opt_b=optB, opt_c=optC, opt_d=optD,
                    kg_sem=kg_sem_str, kg_phon=kg_phon_str,
                )
                user_prompt = budget_user_kg(tok, user_prompt_raw, args.kg_sem_budget, args.kg_phon_budget)
                chat_prefix = render_chat(tok, system_prompt, user_prompt)

                # Generate corrected line
                out = generate_with_timeout(chat_prefix)
                text_full = tok.decode(out[0], skip_special_tokens=True)
                corrected_line = extract_corrected_line(text_full)
                corrected_payload = corrected_line.replace("Corrected Text:", "", 1).strip()
                corrected_line = f"Corrected Text: {corrected_payload}"

                # Force option letter
                prompt_for_letter = chat_prefix + corrected_line + "\n" + "Correct Option: "
                letter = force_letter_generate(model, tok, prompt_for_letter).strip()
                if letter not in {"A","B","C","D"}: letter = ""

                rec = {
                    "idx": idx,
                    "answer": (row.get("answer") or row.get("answer_letter","")).strip().upper(),
                    "pred_option": letter,
                    "corrected_text": corrected_payload,
                    "text_gt": row.get("text_gt",""),
                    "dataset": row.get("dataset",""),
                    "subject": row.get("subject",""),
                    "uid": row.get("uid",""),
                    "audio_path": row.get("audio_path",""),
                    "mode": f"{INPUT_MODE}{'+KG' if USE_KG else ''}",
                }

                # write shard
                with open(shard_path, "a", encoding="utf-8") as sf:
                    sf.write(json.dumps(rec) + "\n"); sf.flush(); os.fsync(sf.fileno())
                if args.echo_json:
                    print(f"[ECHO][GPU {gpu_id}] {json.dumps(rec)}", flush=True)

                # send to main
                send_result((idx, rec, None))
                processed += 1
                if processed % 10 == 0:
                    rate = processed / max(1e-6, (time.time() - t0))
                    print(f"[GPU {gpu_id}] progress: {processed} samples, ~{rate:.2f} it/s", flush=True)
                break
            except StepTimeout as e:
                attempt += 1
                print(f"[GPU {gpu_id}] TIMEOUT idx={idx} attempt={attempt}: {e}", file=sys.stderr, flush=True)
                if attempt > args.retries:
                    err_rec = {"idx": idx, "error": f"Timeout: {e}", "audio_path": row.get("audio_path","")}
                    with open(shard_path, "a", encoding="utf-8") as sf:
                        sf.write(json.dumps(err_rec) + "\n"); sf.flush(); os.fsync(sf.fileno())
                    send_result((idx, None, f"Timeout: {e}"))
                    break
                time.sleep(min(2.0 * attempt, 8.0))
            except Exception as e:
                print(f"[GPU {gpu_id}] ERROR idx={idx}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                err_rec = {"idx": idx, "error": f"{type(e).__name__}: {e}", "audio_path": row.get("audio_path","")}
                with open(shard_path, "a", encoding="utf-8") as sf:
                    sf.write(json.dumps(err_rec) + "\n"); sf.flush(); os.fsync(sf.fileno())
                send_result((idx, None, f"{type(e).__name__}: {e}"))
                break

# ---------- merge helpers ----------
def read_jsonl(fp: str):
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            try: yield json.loads(s)
            except: continue

def merge_shards(shard_dir: str, final_path: str, include_existing_final: bool = True) -> int:
    recs = []
    if include_existing_final and os.path.exists(final_path):
        recs.extend(list(read_jsonl(final_path)))
    for fp in glob.glob(os.path.join(shard_dir, "*.jsonl")):
        recs.extend(list(read_jsonl(fp)))
    with_idx = [r for r in recs if isinstance(r, dict) and "idx" in r]
    with_idx.sort(key=lambda r: r.get("idx", 10**12))
    out_tmp = final_path + ".tmp"
    seen = set()
    with open(out_tmp, "w", encoding="utf-8") as out:
        for r in with_idx:
            i = int(r.get("idx"))
            if i in seen: continue
            seen.add(i)
            out.write(json.dumps(r) + "\n")
        out.flush(); os.fsync(out.fileno())
    os.replace(out_tmp, final_path)
    print(f"[MERGE] -> {os.path.abspath(final_path)} (unique idx={len(seen)})", flush=True)
    return len(seen)

def load_done_idx_from_path(path: str) -> Set[int]:
    done = set()
    if not os.path.exists(path): return done
    for rec in read_jsonl(path):
        if isinstance(rec, dict) and "idx" in rec:
            done.add(int(rec["idx"]))
    return done

def load_done_idx_from_shards(shard_dir: str) -> Set[int]:
    done = set()
    for fp in glob.glob(os.path.join(shard_dir, "*.jsonl")):
        for rec in read_jsonl(fp):
            if isinstance(rec, dict) and "idx" in rec:
                done.add(int(rec["idx"]))
    return done

# ---------- orchestrator & CLI ----------
def run_infer(args):
    # Output dirs
    if args.preds_out:
        out_dir = os.path.dirname(args.preds_out)
        if out_dir: os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.shard_out_dir, exist_ok=True)

    # touch/clear preds_out unless resume
    if not args.resume or not os.path.exists(args.preds_out):
        with open(args.preds_out, "w", encoding="utf-8") as f:
            f.write(""); f.flush(); os.fsync(f.fileno())

    print(f"[INFO] CWD: {os.getcwd()}")
    print(f"[INFO] Writing predictions to: {os.path.abspath(args.preds_out)}")
    print(f"[INFO] Shard directory: {os.path.abspath(args.shard_out_dir)}")

    # Load manifest
    with open(args.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    N = len(rows)
    print(f"[INFO] Loaded {N} rows from {args.manifest}")
    if N == 0:
        print("[FATAL] Manifest has zero rows.", file=sys.stderr); sys.exit(2)

    # Resume set
    done_idx: Set[int] = set()
    if args.resume:
        done_idx |= load_done_idx_from_path(args.preds_out)
        done_idx |= load_done_idx_from_shards(args.shard_out_dir)
        if args.resume_from:
            done_idx |= load_done_idx_from_path(args.resume_from)
        print(f"[RESUME] Skipping {len(done_idx)} already-done indices")

    # Tasks
    tasks = [(i, rows[i]) for i in range(N) if i not in done_idx]
    print(f"[INFO] Enqueuing {len(tasks)} tasks (skipped {N - len(tasks)})")

    # Queues
    ctx = mp.get_context("spawn")
    task_q: mp.Queue = ctx.Queue(maxsize=args.task_q_max)
    result_q: mp.Queue = ctx.Queue(maxsize=args.result_q_max)

    # Spawn workers
    gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()] or ["0"]
    total_workers = len(gpu_list) * max(1, args.workers_per_gpu)
    print(f"[INFO] Starting {total_workers} workers across GPUs={gpu_list}")

    procs = []
    for g in gpu_list:
        for _ in range(args.workers_per_gpu):
            p = ctx.Process(target=worker_main, args=(task_q, result_q, args, g), daemon=False)
            p.start(); procs.append(p)

    # Signals
    stop_event = threading.Event()
    def _handle_sig(signum, frame):
        print(f"[SIGNAL] Received {signum}. Stopping after drain.", flush=True)
        stop_event.set()
    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    # Enqueue tasks
    for item in tasks: task_q.put(item)
    for _ in procs: task_q.put(None)

    # Periodic merger thread
    merger_thread = None
    merger_stop = threading.Event()
    merge_target_for_periodic = (args.preds_out + ".live") if args.merge_to_live else args.preds_out

    def merger_loop():
        while not merger_stop.wait(timeout=max(1, args.periodic_merge_secs)):
            try:
                cnt = merge_shards(
                    args.shard_out_dir,
                    merge_target_for_periodic,
                    include_existing_final=not args.merge_to_live
                )
                print(f"[HEARTBEAT][MERGE] merged_count={cnt}", flush=True)
            except Exception as e:
                print(f"[MERGE] Error: {e}", file=sys.stderr, flush=True)

    if args.periodic_merge_secs and args.periodic_merge_secs > 0:
        merger_thread = threading.Thread(target=merger_loop, daemon=True)
        merger_thread.start()
        print(f"[INFO] Periodic merge every {args.periodic_merge_secs}s → {merge_target_for_periodic}")

    # Async writer
    write_buf = deque()
    writer_stop = threading.Event()
    written_lines = 0

    def writer_loop():
        nonlocal written_lines
        with open(args.preds_out, "a", encoding="utf-8") as out:
            while not writer_stop.is_set() or write_buf:
                if not write_buf:
                    time.sleep(0.01); continue
                line = write_buf.popleft()
                out.write(line); written_lines += 1
                if args.fsync_every > 0 and (written_lines % args.fsync_every == 0):
                    out.flush(); os.fsync(out.fileno())

    writer_thread = None
    out_file = None
    if args.async_writer:
        writer_thread = threading.Thread(target=writer_loop, daemon=True)
        writer_thread.start()
        print(f"[INFO] Async writer enabled (fsync_every={args.fsync_every})")
    else:
        out_file = open(args.preds_out, "a", encoding="utf-8")

    # Collector
    pending = len(tasks); errors = 0; written = 0; last_hb = time.time()
    def write_line(line: str):
        nonlocal written
        if args.async_writer:
            write_buf.append(line)
        else:
            out_file.write(line); out_file.flush()
            if args.fsync_every <= 1: os.fsync(out_file.fileno())
        written += 1

    while pending and not stop_event.is_set():
        if time.time() - last_hb > 30:
            print(f"[HEARTBEAT][MAIN] pending={pending} written={written} errors={errors}", flush=True)
            last_hb = time.time()
        try:
            msg = result_q.get(timeout=10)
        except Empty:
            continue
        if msg is None:
            continue
        idx, rec, err = msg; pending -= 1
        if err:
            errors += 1
            rec = rec or {"idx": idx, "error": err}
        write_line(json.dumps(rec) + "\n")
        if args.echo_json:
            print(f"[ECHO][MAIN] {json.dumps(rec)}", flush=True)

    for p in procs: p.join(timeout=5)

    if args.async_writer:
        writer_stop.set()
        if writer_thread: writer_thread.join(timeout=5)
    else:
        if out_file: out_file.close()

    if merger_thread:
        merger_stop.set()
        merger_thread.join(timeout=5)

    # Final canonical merge
    try:
        total_unique = merge_shards(args.shard_out_dir, args.preds_out, include_existing_final=True)
        print(f"[DONE] Final merge complete. Unique idx={total_unique}")
    except Exception as e:
        print(f"[DONE] Final merge FAILED: {e}", file=sys.stderr)

    print(f"[STATS] tasks={len(tasks)} written={written} errors={errors}")

def build_parser():
    p = argparse.ArgumentParser(description="MedSpeak robust multi-GPU inference (all modes, v2)")
    sub = p.add_subparsers(dest="cmd", required=False)

    infer = sub.add_parser("infer", help="Run inference (default)")
    infer.add_argument("--manifest", required=True)
    infer.add_argument("--kg_sql", required=True)
    infer.add_argument("--kg_phon", required=True)
    infer.add_argument("--base_model", required=True)
    infer.add_argument("--lora_dir", default="")
    infer.add_argument("--use_4bit", action="store_true")

    infer.add_argument("--asr_backend", choices=["whisper","faster-whisper"], default="whisper")
    infer.add_argument("--asr_model", default="small")
    infer.add_argument("--whisper_cache", default="")
    infer.add_argument("--config", default="config_v2.yaml")

    infer.add_argument("--preds_out", required=True)
    infer.add_argument("--shard_out_dir", required=True)
    infer.add_argument("--echo_json", action="store_true")

    infer.add_argument("--kg_sem_budget", type=int, default=DEFAULT_KG_SEM_BUDGET)
    infer.add_argument("--kg_phon_budget", type=int, default=DEFAULT_KG_PHON_BUDGET)
    infer.add_argument("--max_new_tokens_text", type=int, default=DEFAULT_MAX_NEW_TOKENS_TEXT)

    infer.add_argument("--gpus", default="0")
    infer.add_argument("--workers_per_gpu", type=int, default=1)

    infer.add_argument("--periodic_merge_secs", type=int, default=30)
    infer.add_argument("--resume", action="store_true")
    infer.add_argument("--resume_from", default="")

    infer.add_argument("--gen_max_time", type=float, default=45.0)
    infer.add_argument("--asr_timeout", type=float, default=60.0)
    infer.add_argument("--retries", type=int, default=1)

    infer.add_argument("--merge_to_live", action="store_true")
    infer.add_argument("--task_q_max", type=int, default=8192)
    infer.add_argument("--result_q_max", type=int, default=8192)

    infer.add_argument("--async_writer", action="store_true")
    infer.add_argument("--fsync_every", type=int, default=10)

    infer.add_argument("--input_mode", choices=["audio","text"], default="audio")
    infer.add_argument("--use_kg", action="store_true")

    m = sub.add_parser("merge", help="Merge shards into a final preds file")
    m.add_argument("--shard_dir", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--include_existing_final", action="store_true")

    return p

def main():
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "merge":
        merge_shards(args.shard_dir, args.out, include_existing_final=args.include_existing_final)
        return
    run_infer(args)

if __name__ == "__main__":
    main()
