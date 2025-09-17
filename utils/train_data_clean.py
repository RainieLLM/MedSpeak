#!/usr/bin/env python3
import json, re, argparse

SYS_RULES = """You are MedSpeak, a medical ASR correction and QA assistant.

Output exactly two lines, no extra text.
Line 1 must start with: Corrected Text:
Line 2 must start with: Correct Option:

Rules:
- Do not repeat the question or options.
- Do not explain or add details.
- Do not use placeholders.
- Output nothing beyond those two lines.
""".strip()

def clean_system(s):
    return SYS_RULES

def clean_user(u):
    # Remove any trailing “follow the system … Corrected Text …” appendix if present
    u = re.split(r"\n\s*Follow the system instructions.*", u, flags=re.S)[0]
    return u.strip() + "\n"

def normalize_assistant(a):
    lines = [ln.strip() for ln in a.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return a.strip() + "\n"  # leave as-is if malformed
    ct, co = lines[0], lines[1]

    # Force exact prefixes
    ct_content = re.sub(r"^Corrected\s*Text\s*:\s*", "", ct, flags=re.I)
    co_content = re.sub(r"^Correct\s*Option\s*:\s*", "", co, flags=re.I)

    # Uppercase option single letter
    m = re.match(r"\s*([A-D])\s*$", co_content, flags=re.I)
    opt = m.group(1).upper() if m else co_content[:1].upper()

    return f"Corrected Text: {ct_content.strip()}\nCorrect Option: {opt}\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    with open(args.in_jsonl, "r", encoding="utf-8") as fin, open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for ln in fin:
            if not ln.strip(): continue
            rec = json.loads(ln)
            msgs = rec.get("messages", [])
            for m in msgs:
                if m["role"] == "system":
                    m["content"] = clean_system(m["content"])
                elif m["role"] == "user":
                    m["content"] = clean_user(m["content"])
                elif m["role"] == "assistant":
                    m["content"] = normalize_assistant(m["content"])
            rec["messages"] = msgs
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
