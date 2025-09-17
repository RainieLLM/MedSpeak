#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate accuracy & WER (punctuation-insensitive) directly from JSONL
where each record already contains:
  - answer (GT A/B/C/D)
  - pred_option (predicted A/B/C/D)
  - text_gt (reference text)
  - corrected_text (predicted text)
  - dataset / subject (optional, for grouping)

Usage:
python scripts/eval_acc_wer.py \
  --preds runs/whisper_ftllm_noKG.jsonl \
  --out results/whisper_ftllm_noKG_eval.json
"""

import argparse, json, re, csv
from collections import defaultdict

PUNC_TO_IGNORE = set(".,?!:;\"'()[]{}`“”’/\\|-_+*=#@^&~")
WS_RE = re.compile(r"\s+")

def normalize(s: str) -> str:
    if not s: return ""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in PUNC_TO_IGNORE)
    return WS_RE.sub(" ", s).strip()

def wer(ref: str, hyp: str) -> float:
    r, h = normalize(ref).split(), normalize(hyp).split()
    R, H = len(r), len(h)
    if R == 0:
        return 0.0 if H == 0 else 1.0
    dp = [[0]*(H+1) for _ in range(R+1)]
    for i in range(R+1): dp[i][0] = i
    for j in range(H+1): dp[0][j] = j
    for i in range(1, R+1):
        for j in range(1, H+1):
            cost = 0 if r[i-1] == h[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )
    return dp[R][H] / max(1, R)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="JSONL file with answer, pred_option, text_gt, corrected_text")
    ap.add_argument("--out", required=True, help="Output JSON summary")
    ap.add_argument("--details_csv", default="", help="Optional CSV with per-sample metrics")
    ap.add_argument("--group_by", nargs="*", default=["dataset", "subject"], help="Group columns if present")
    args = ap.parse_args()

    total, correct, wers = 0, 0, []
    groups = defaultdict(lambda: {"n": 0, "correct": 0, "wers": []})
    details = []

    with open(args.preds, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)

            total += 1
            ans_gt = (rec.get("answer") or "").strip().upper()
            ans_pred = (rec.get("pred_option") or "").strip().upper()
            is_correct = int(ans_pred in {"A","B","C","D"} and ans_pred == ans_gt)
            correct += is_correct

            ref = rec.get("text_gt", "")
            hyp = rec.get("corrected_text", "")
            sample_wer = wer(ref, hyp) if ref else None
            if sample_wer is not None: wers.append(sample_wer)

            for col in args.group_by:
                key = rec.get(col, "").strip()
                if key:
                    g = groups[key]
                    g["n"] += 1
                    g["correct"] += is_correct
                    if sample_wer is not None:
                        g["wers"].append(sample_wer)

            details.append({
                "idx": rec.get("idx"),
                "answer_gt": ans_gt,
                "pred_option": ans_pred,
                "correct": is_correct,
                "wer": sample_wer if sample_wer is not None else "",
                **{col: rec.get(col,"") for col in args.group_by}
            })

    acc = correct / max(1, total)
    mean_wer = sum(wers)/len(wers) if wers else None
    summary = {
        "total": total,
        "overall": {"accuracy": acc, "mean_wer": mean_wer},
        "groups": {
            k: {
                "n": v["n"],
                "accuracy": v["correct"]/v["n"] if v["n"] else None,
                "mean_wer": sum(v["wers"])/len(v["wers"]) if v["wers"] else None
            }
            for k,v in groups.items()
        }
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.details_csv:
        with open(args.details_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["idx","answer_gt","pred_option","correct","wer"] + args.group_by
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(details)

    print(f"[EVAL] total={total} acc={acc:.4f} mean_wer={(mean_wer if mean_wer is not None else 'NA')}")
    print(f"[EVAL] summary -> {args.out}")
    if args.details_csv: print(f"[EVAL] details -> {args.details_csv}")

if __name__ == "__main__":
    main()
