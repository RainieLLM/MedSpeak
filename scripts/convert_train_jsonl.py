#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix context in chat-style JSONL for MedSpeak:
- Replace every system message with a short, strict, placeholder-free prompt.
- Remove any trailing "Follow the system..." block and any "Corrected Text:" / "Correct Option:" tails from USER messages.
- Leave ASSISTANT labels and all metadata untouched.

Usage:
  python fix_train_jsonl_context.py --in datasets/qa/train.jsonl --out datasets/qa/train_context_fixed.jsonl
  # or overwrite in place:
  python fix_train_jsonl_context.py --in datasets/qa/train.jsonl --inplace
"""

import argparse, json, os, re, sys, tempfile, shutil

NEW_SYSTEM = (
    "You are MedSpeak, a medical ASR correction and QA assistant.\n\n"
    "Output exactly two lines, no extra text.\n"
    "Line 1 must start with: Corrected Text:\n"
    "Line 2 must start with: Correct Option:\n\n"
    "Rules:\n"
    "- Do not repeat the question or options.\n"
    "- Do not explain or add details.\n"
    "- Do not use placeholders.\n"
    "- The second line must be exactly in the form: Correct Option: <A|B|C|D> (where <A|B|C|D> is a single uppercase letter)."
    "- Output nothing beyond those two lines."
)

# Patterns to strip any prompt/placeholder tail from USER messages
RE_TAIL_STARTS = re.compile(
    r"(?mi)^\s*(follow the system instructions.*|corrected\s*text\s*:|correct\s*option\s*:).*"
)

def clean_user_content(text: str) -> str:
    """
    Remove any trailing block that begins with:
    - 'Follow the system instructions...' (any case)
    - 'Corrected Text:' lines included in the USER message
    - 'Correct Option:' lines included in the USER message
    Everything from that first matched line to end-of-text is removed.
    """
    if not isinstance(text, str):
        return text
    m = RE_TAIL_STARTS.search(text)
    return text[:m.start()].rstrip() if m else text

def fix_record(line: str) -> str:
    obj = json.loads(line)

    # 1) Replace system content(s)
    msgs = obj.get("messages", [])
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "system":
            m["content"] = NEW_SYSTEM

    # 2) Clean user tails
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "user":
            m["content"] = clean_user_content(m.get("content", ""))

    obj["messages"] = msgs
    return json.dumps(obj, ensure_ascii=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input JSONL path")
    ap.add_argument("--out", dest="out", help="Output JSONL path (default: *_context_fixed.jsonl)")
    ap.add_argument("--inplace", action="store_true", help="Overwrite input in place")
    args = ap.parse_args()

    inp = args.inp
    if not os.path.isfile(inp):
        print(f"Input not found: {inp}", file=sys.stderr)
        sys.exit(1)

    if args.inplace and args.out:
        print("Use either --inplace or --out, not both.", file=sys.stderr)
        sys.exit(2)

    if args.inplace:
        dir_ = os.path.dirname(inp) or "."
        outp = os.path.join(dir_, ".__tmp_ctx_fixed.jsonl")
    else:
        root, ext = os.path.splitext(inp)
        outp = args.out or f"{root}_context_fixed{ext or '.jsonl'}"

    total, wrote = 0, 0
    with open(inp, "r", encoding="utf-8") as f_in, open(outp, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            total += 1
            try:
                fixed = fix_record(line)
            except Exception as e:
                print(f"[WARN] Skipping malformed line {total}: {e}", file=sys.stderr)
                continue
            f_out.write(fixed + "\n")
            wrote += 1

    if args.inplace:
        shutil.move(outp, inp)
        print(f"Patched {wrote}/{total} examples in place -> {inp}")
    else:
        print(f"Patched {wrote}/{total} examples -> {outp}")

if __name__ == "__main__":
    main()
