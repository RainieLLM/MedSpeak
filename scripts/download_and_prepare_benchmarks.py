
import argparse
import os
import json
from typing import List, Dict

import numpy as np
import soundfile as sf
from datasets import load_dataset

try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

SR = 16000

def ensure_dirs():
    for p in [
        "data/audio/benchmarks",
        "data/qa",
        "data/csv_files/mmlu",
        "data/csv_files/medmcqa",
        "data/csv_files/medqa",
    ]:
        os.makedirs(p, exist_ok=True)

def save_silence_wav(path: str, seconds: float = 3.0):
    n = int(SR * max(0.5, seconds))
    audio = np.zeros((n,), dtype=np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, audio, SR, subtype="PCM_16")

def tts_to_wav(text: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not TTS_AVAILABLE:
        seconds = max(2.0, 0.2 * len(text.split()))
        save_silence_wav(path, seconds)
        return "silence"
    engine = pyttsx3.init()
    rate = engine.getProperty("rate")
    engine.setProperty("rate", int(rate * 0.9))
    try:
        engine.save_to_file(text, path)
        engine.runAndWait()
    except Exception:
        seconds = max(2.0, 0.2 * len(text.split()))
        save_silence_wav(path, seconds)
        return "silence"
    # ensure 16k mono
    try:
        audio, sr = sf.read(path, always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            import numpy as np
            x_old = np.linspace(0, 1, num=len(audio), endpoint=False)
            x_new = np.linspace(0, 1, num=int(len(audio) * SR / sr), endpoint=False)
            audio = np.interp(x_new, x_old, audio).astype(np.float32)
        sf.write(path, audio, SR, subtype="PCM_16")
    except Exception:
        save_silence_wav(path, seconds=3.0)
        return "silence"
    return "tts"

def clean(s: str) -> str:
    return " ".join(str(s).split()).strip()

def write_csv(rows: List[Dict[str, str]], out_csv: str):
    import csv
    if not rows:
        cols = ["dataset","subject","uid","question","option_a","option_b","option_c","option_d","answer"]
    else:
        cols = list(rows[0].keys())
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"Wrote CSV: {out_csv} (rows={len(rows)})")

def row_to_manifest(rec: Dict[str, str]) -> Dict[str, str]:
    return {
        "audio_path": rec["audio_path"],
        "text_gt": rec["text_gt"],
        "question": rec["question"],
        "option_a": rec.get("option_a",""),
        "option_b": rec.get("option_b",""),
        "option_c": rec.get("option_c",""),
        "option_d": rec.get("option_d",""),
        "answer": rec["answer"],
        "dataset": rec.get("dataset",""),
        "subject": rec.get("subject",""),
        "uid": rec.get("uid",""),
    }

MMLU_SUBJECTS_DEFAULT = ["clinical_knowledge","anatomy","professional_medicine","medical_genetics","virology","college_biology","nutrition"]

def load_mmlu(subjects: List[str], split: str, limit: int):
    items = []
    for subj in subjects:
        try:
            ds = load_dataset("cais/mmlu", subj, split=split)
        except Exception:
            ds = load_dataset("hendrycks_test", subj, split=split)
        for idx, ex in enumerate(ds):
            q = clean(ex.get("question",""))
            choices = ex.get("choices") or [ex.get("A",""), ex.get("B",""), ex.get("C",""), ex.get("D","")]
            ans = ex.get("answer") or ex.get("target")
            if isinstance(ans, int):
                letter = "ABCD"[ans]
            else:
                letter = str(ans).strip().upper()[:1]
            opts = [clean(choices[i]) if i < len(choices) else "" for i in range(4)]
            uid = f"mmlu::{subj}::{idx}"
            items.append({
                "dataset": "MMLU",
                "subject": subj,
                "uid": uid,
                "question": q,
                "option_a": opts[0], "option_b": opts[1], "option_c": opts[2], "option_d": opts[3],
                "answer": letter
            })
            if limit and len(items) >= limit:
                return items
    return items

def load_medmcqa(split: str, limit: int):
    ds = load_dataset("medmcqa", split=split)
    items = []
    for idx, ex in enumerate(ds):
        q = clean(ex.get("question",""))
        opts = [clean(ex.get("opa","")), clean(ex.get("opb","")), clean(ex.get("opc","")), clean(ex.get("opd",""))]
        letter = clean(ex.get("cop","")).upper()[:1]
        uid = f"medmcqa::{split}::{idx}"
        items.append({
            "dataset": "MedMCQA",
            "subject": "general",
            "uid": uid,
            "question": q,
            "option_a": opts[0], "option_b": opts[1], "option_c": opts[2], "option_d": opts[3],
            "answer": letter
        })
        if limit and len(items) >= limit:
            break
    return items

def try_load_medqa(limit: int):
    trials = [
        ("bigbio/med_qa", {"name": "med_qa_en", "split": "validation"}),
        ("openlifescienceai/medical_meadow_medqa_4options", {"split": "validation"}),
    ]
    items = []
    for name, kwargs in trials:
        try:
            ds = load_dataset(name, **kwargs)
            d = next(iter(ds.values())) if isinstance(ds, dict) else ds
            for idx, ex in enumerate(d):
                q = clean(ex.get("question", ex.get("prompt","")))
                choices = []
                for key in ["A","B","C","D","option_a","option_b","option_c","option_d","opa","opb","opc","opd"]:
                    if key in ex:
                        choices.append(clean(ex[key]))
                if len(choices) < 4 and "choices" in ex:
                    ch = ex["choices"]
                    for i in range(min(4, len(ch))):
                        choices.append(clean(ch[i]))
                choices = (choices + ["","","",""])[:4]
                ans = ex.get("answer", ex.get("label", ex.get("cop","")))
                if isinstance(ans, int):
                    letter = "ABCD"[ans]
                else:
                    letter = str(ans).strip().upper()[:1]
                uid = f"medqa::validation::{idx}"
                items.append({
                    "dataset": "MedQA",
                    "subject": "general",
                    "uid": uid,
                    "question": q,
                    "option_a": choices[0], "option_b": choices[1], "option_c": choices[2], "option_d": choices[3],
                    "answer": letter
                })
                if limit and len(items) >= limit:
                    return items
            return items
        except Exception:
            continue
    return items

def synthesize(items, audio_root, tts_mode):
    rows = []
    for i, it in enumerate(items):
        q = it["question"]
        subj = it["subject"]
        dsname = it["dataset"]
        slug_subj = subj.replace(" ", "_")
        subdir = os.path.join(audio_root, dsname, slug_subj)
        filename = f"{dsname.lower()}_{slug_subj}_{i:06d}.wav"
        audio_path = os.path.join(subdir, filename)
        if tts_mode == "tts":
            tts_to_wav(q, audio_path)
        else:
            seconds = max(2.0, 0.18*len(q.split()))
            save_silence_wav(audio_path, seconds)
        rec = row_to_manifest({
            "audio_path": audio_path,
            "text_gt": q,
            "question": q,
            "option_a": it["option_a"],
            "option_b": it["option_b"],
            "option_c": it["option_c"],
            "option_d": it["option_d"],
            "answer": it["answer"],
            "dataset": dsname,
            "subject": subj,
            "uid": it["uid"],
        })
        rows.append(rec)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmlu_subjects", type=str, default=",".join(MMLU_SUBJECTS_DEFAULT))
    ap.add_argument("--mmlu_split", type=str, default="test")
    ap.add_argument("--mmlu_limit", type=int, default=200)
    ap.add_argument("--medmcqa_split", type=str, default="validation")
    ap.add_argument("--medmcqa_limit", type=int, default=500)
    ap.add_argument("--medqa_limit", type=int, default=500)
    ap.add_argument("--tts", choices=["auto","silence","tts"], default="auto")
    ap.add_argument("--out_manifest", type=str, default="data/qa/manifest.csv")
    args = ap.parse_args()

    ensure_dirs()
    tts_mode = "tts" if (args.tts == "tts" or (args.tts == "auto" and TTS_AVAILABLE)) else "silence"

    subjects = [s.strip() for s in args.mmlu_subjects.split(",") if s.strip()]
    print(f"Fetching MMLU subjects: {subjects}")
    mmlu_items = load_mmlu(subjects, split=args.mmlu_split, limit=args.mmlu_limit)
    write_csv(mmlu_items, "data/csv_files/mmlu/mmlu.csv")

    print("Fetching MedMCQA...")
    medmcqa_items = load_medmcqa(split=args.medmcqa_split, limit=args.medmcqa_limit)
    write_csv(medmcqa_items, "data/csv_files/medmcqa/medmcqa.csv")

    print("Fetching MedQA (best-effort)...")
    medqa_items = try_load_medqa(limit=args.medqa_limit)
    write_csv(medqa_items, "data/csv_files/medqa/medqa.csv")

    all_rows = []
    all_rows += synthesize(mmlu_items, "data/audio/benchmarks", tts_mode)
    all_rows += synthesize(medmcqa_items, "data/audio/benchmarks", tts_mode)
    all_rows += synthesize(medqa_items, "data/audio/benchmarks", tts_mode)

    import csv
    cols = ["audio_path","text_gt","question","option_a","option_b","option_c","option_d","answer","dataset","subject","uid"]
    os.makedirs(os.path.dirname(args.out_manifest), exist_ok=True)
    with open(args.out_manifest, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in all_rows:
            wr.writerow(r)
    print(f"Wrote manifest: {args.out_manifest} (rows={len(all_rows)})")

    print("Done. Mode:", tts_mode.upper())
    if tts_mode != "tts":
        print("Note: Using silent WAV placeholders (valid 16kHz mono). Install 'pyttsx3' to synthesize voiced audio.")

if __name__ == "__main__":
    main()
