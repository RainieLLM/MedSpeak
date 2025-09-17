#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read *existing* CSVs (MMLU/MedQA/MedMCQA), synthesize WAVs (TTS if available,
else silence), and optionally run Whisper ASR, then emit/append a unified
`manifest.csv` with columns:
  audio_path, text_gt, question, option_a..d, answer, dataset, subject, uid, asr_text

Expected CSV formats:
- mmlu_qa.csv: id,question,optionA,optionB,optionC,optionD,answer_letter
  (id looks like "anatomy_0"; subject is parsed from the id prefix before '_')
- medqa_qa.csv: id,question,optionA,optionB,optionC,optionD,answer_letter
- medmcqa_qa.csv: id,question,optionA,optionB,optionC,optionD,answer_letter

Usage examples:
  # build silence WAVs, no ASR (fastest, no native deps)
  python scripts/csv_to_audio_manifest.py \
    --mmlu_csv data/csv_files/mmlu_qa.csv \
    --medqa_csv data/csv_files/medqa_qa.csv \
    --medmcqa_csv data/csv_files/medmcqa_qa.csv \
    --tts silence \
    --manifest data/qa/manifest.csv

  # add Whisper ASR later (CPU-safe)
  python scripts/csv_to_audio_manifest.py \
    --mmlu_csv data/csv_files/mmlu_qa.csv \
    --medqa_csv data/csv_files/medqa_qa.csv \
    --medmcqa_csv data/csv_files/medmcqa_qa.csv \
    --tts auto \
    --transcribe_whisper small \
    --manifest data/qa/manifest.csv

Requirements (minimal):
- numpy
- optional: pyttsx3 (for spoken TTS), openai-whisper, torchaudio, ffmpeg (for ASR)
"""

# ---- Stability caps (avoid OpenMP/BLAS fights) ----
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional
import wave
import numpy as np

# Optional dependencies
try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

SR = 16000

# --------- helpers ---------

def ensure_dirs():
    for p in ["data/audio/benchmarks", "data/qa", "artifacts"]:
        os.makedirs(p, exist_ok=True)

def clean(s: str) -> str:
    return " ".join(str(s or "").split()).strip()

def to_letter(x) -> Optional[str]:
    if isinstance(x, int) and 0 <= x <= 3:
        return "ABCD"[x]
    xx = str(x).strip().upper()
    return xx[:1] if xx and xx[0] in "ABCD" else None

# ------ WAV writing (stdlib only) ------

def write_wav_pcm16(path: str, samples: np.ndarray, sr: int = SR):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    y = np.asarray(samples).squeeze()
    if y.dtype != np.int16:
        y = np.clip(y, -1.0, 1.0)
        y = (y * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(y.tobytes())

def save_silence_wav(path: str, seconds: float = 3.0):
    n = int(SR * max(0.5, seconds))
    write_wav_pcm16(path, np.zeros(n, dtype=np.int16), SR)


def tts_to_wav(text: str, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not TTS_AVAILABLE:
        save_silence_wav(path, seconds=max(2.0, 0.18*len(text.split())))
        return "silence"
    try:
        engine = pyttsx3.init()
        rate = engine.getProperty("rate")
        engine.setProperty("rate", int(rate * 0.9))
        engine.save_to_file(text, path)
        engine.runAndWait()
        return "tts"
    except Exception:
        save_silence_wav(path, seconds=max(2.0, 0.18*len(text.split())))
        return "silence"

# ------ manifest utils ------

def append_manifest(rows: List[Dict[str, str]], manifest_csv: str):
    cols = [
        "audio_path","text_gt","question","option_a","option_b","option_c","option_d",
        "answer","dataset","subject","uid","asr_text"
    ]
    exists = os.path.exists(manifest_csv)
    os.makedirs(os.path.dirname(manifest_csv), exist_ok=True)
    with open(manifest_csv, "a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        if not exists:
            wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"[MANIFEST] Appended {len(rows)} rows -> {manifest_csv}")


def row_to_manifest(audio_path, question, opts, ans, ds, subj, uid, text_gt, asr_text=""):
    return {
        "audio_path": audio_path,
        "text_gt": text_gt,
        "question": question,
        "option_a": opts[0],
        "option_b": opts[1],
        "option_c": opts[2],
        "option_d": opts[3],
        "answer": ans,
        "dataset": ds,
        "subject": subj,
        "uid": uid,
        "asr_text": asr_text,
    }

# ------ CSV readers ------

def read_generic_csv(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
    return rows

# ------ build items from each dataset ------

def items_from_mmlu(csv_path: str) -> List[Dict[str, str]]:
    items = []
    for r in read_generic_csv(csv_path):
        rid = str(r.get("id", "")).strip()
        subj = rid.split("_", 1)[0] if "_" in rid else "general"
        q = clean(r.get("question", ""))
        opts = [clean(r.get("optionA", "")), clean(r.get("optionB", "")), clean(r.get("optionC", "")), clean(r.get("optionD", ""))]
        ans = to_letter(r.get("answer_letter", "")) or "A"
        items.append({
            "dataset": "MMLU",
            "subject": subj,
            "uid": f"mmlu::{rid}",
            "question": q,
            "option_a": opts[0], "option_b": opts[1], "option_c": opts[2], "option_d": opts[3],
            "answer": ans,
            "text_gt": q,
        })
    print(f"[MMLU] Loaded {len(items)} from {csv_path}")
    return items


def items_from_medqa(csv_path: str) -> List[Dict[str, str]]:
    items = []
    for r in read_generic_csv(csv_path):
        rid = str(r.get("id", "")).strip()
        q = clean(r.get("question", ""))
        opts = [clean(r.get("optionA", "")), clean(r.get("optionB", "")), clean(r.get("optionC", "")), clean(r.get("optionD", ""))]
        ans = to_letter(r.get("answer_letter", "")) or "A"
        items.append({
            "dataset": "MedQA",
            "subject": "general",
            "uid": f"medqa::{rid}",
            "question": q,
            "option_a": opts[0], "option_b": opts[1], "option_c": opts[2], "option_d": opts[3],
            "answer": ans,
            "text_gt": q,
        })
    print(f"[MedQA] Loaded {len(items)} from {csv_path}")
    return items


def items_from_medmcqa(csv_path: str) -> List[Dict[str, str]]:
    items = []
    for r in read_generic_csv(csv_path):
        rid = str(r.get("id", "")).strip()
        q = clean(r.get("question", ""))
        opts = [clean(r.get("optionA", "")), clean(r.get("optionB", "")), clean(r.get("optionC", "")), clean(r.get("optionD", ""))]
        ans = to_letter(r.get("answer_letter", "")) or "A"
        items.append({
            "dataset": "MedMCQA",
            "subject": "general",
            "uid": f"medmcqa::{rid}",
            "question": q,
            "option_a": opts[0], "option_b": opts[1], "option_c": opts[2], "option_d": opts[3],
            "answer": ans,
            "text_gt": q,
        })
    print(f"[MedMCQA] Loaded {len(items)} from {csv_path}")
    return items

# ------ synthesis + (optional) ASR ------

def synthesize_audio_for_items(items: List[Dict[str, str]], audio_root: str, tts_mode: str) -> List[Dict[str, str]]:
    manifest_rows = []
    for i, it in enumerate(items):
        try:
            txt = it.get("text_gt", it["question"]) or ""
            subj = it["subject"]
            dsname = it["dataset"]
            subdir = os.path.join(audio_root, dsname, subj.replace(" ", "_"))
            os.makedirs(subdir, exist_ok=True)
            fname = f"{dsname.lower()}_{subj.replace(' ','_')}_{i:07d}.wav"
            apath = os.path.join(subdir, fname)

            if tts_mode == "tts":
                mode = tts_to_wav(txt, apath)
            else:
                save_silence_wav(apath, seconds=max(2.0, 0.18 * len(txt.split())))
                mode = "silence"

            if (i + 1) % 100 == 0:
                print(f"[AUDIO] {i+1}/{len(items)} synthesized (mode={mode})")

            manifest_rows.append(row_to_manifest(
                audio_path=apath,
                question=it["question"],
                opts=[it["option_a"], it["option_b"], it["option_c"], it["option_d"]],
                ans=it["answer"],
                ds=dsname,
                subj=subj,
                uid=it["uid"],
                text_gt=txt,
                asr_text="",
            ))
        except Exception as e:
            print(f"[AUDIO][WARN] failed at idx={i}: {e}", file=sys.stderr)
            continue
    return manifest_rows


def run_whisper_on_manifest(manifest_csv: str, model_size: str = "small"):
    if not WHISPER_AVAILABLE:
        print("[ASR] openai-whisper not installed; skipping ASR.", file=sys.stderr)
        return
    print(f"[ASR] Loading Whisper model: {model_size}")
    asr = whisper.load_model(model_size)

    with open(manifest_csv, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
        fieldnames = rdr.fieldnames

    for idx, r in enumerate(rows):
        if r.get("asr_text"):
            continue
        apath = r["audio_path"]
        try:
            result = asr.transcribe(apath, task="transcribe", fp16=False)
            r["asr_text"] = clean(result.get("text", ""))
        except Exception as e:
            print(f"[ASR][WARN] {apath}: {e}", file=sys.stderr)
            r["asr_text"] = ""
        if (idx + 1) % 50 == 0:
            print(f"[ASR] {idx+1}/{len(rows)} done...")

    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"[ASR] Updated manifest with ASR text -> {manifest_csv}")

# ------ main ------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mmlu_csv", type=str, required=False, default="")
    p.add_argument("--medqa_csv", type=str, required=False, default="")
    p.add_argument("--medmcqa_csv", type=str, required=False, default="")
    p.add_argument("--tts", choices=["auto","silence","tts"], default="auto")
    p.add_argument("--transcribe_whisper", type=str, default="none")
    p.add_argument("--manifest", type=str, default="data/qa/manifest.csv")
    p.add_argument("--limit", type=int, default=0, help="limit total items (0 = all)")
    args = p.parse_args()

    ensure_dirs()

    # Decide TTS mode
    tts_mode = "tts" if (args.tts == "tts" or (args.tts == "auto" and TTS_AVAILABLE)) else "silence"
    print(f"[INFO] TTS mode: {tts_mode} (pyttsx3 {'available' if TTS_AVAILABLE else 'NOT available'})")
    print(f"[INFO] Whisper ASR: {args.transcribe_whisper} (installed={WHISPER_AVAILABLE})")

    items: List[Dict[str, str]] = []
    if args.mmlu_csv and os.path.exists(args.mmlu_csv):
        items += items_from_mmlu(args.mmlu_csv)
    if args.medqa_csv and os.path.exists(args.medqa_csv):
        items += items_from_medqa(args.medqa_csv)
    if args.medmcqa_csv and os.path.exists(args.medmcqa_csv):
        items += items_from_medmcqa(args.medmcqa_csv)

    if args.limit > 0:
        items = items[:args.limit]
    print(f"[INFO] Total items: {len(items)}")

    manifest_rows = synthesize_audio_for_items(items, "data/audio/benchmarks", tts_mode)
    append_manifest(manifest_rows, args.manifest)

    if args.transcribe_whisper.lower() != "none":
        run_whisper_on_manifest(args.manifest, model_size=args.transcribe_whisper)

    print("\n[DONE]")
    print(f"  Manifest: {args.manifest}")
    print("  Note: If you used silence WAVs, ASR will transcribe silence unless you add noise or TTS.")

if __name__ == "__main__":
    main()
