# MedSpeak — Knowledge Graph–Aided ASR Error Correction for Spoken Medical QA  

**Llama-3.1-8B-Instruct (Full Fine-Tuning) + Whisper Small + Medical KG (Semantic + Phonetic)**

<p align="center">
  <img src="/assets/medspeak_fig1.pdf" alt="MedSpeak Overview" width="900"/>
</p>


## 📄 Paper

**MedSpeak: A Knowledge Graph-Aided ASR Error Correction Framework for Spoken Medical QA**

Spoken medical question answering (SQA) pipelines typically rely on ASR transcripts. However, **medical terminology** is frequently misrecognized, which propagates errors into downstream QA. **MedSpeak** addresses this by injecting **medical knowledge graph evidence**—including both **semantic relations** and **phonetic similarity**—into an LLM that **jointly performs transcript correction and MCQ answering**.

**Core idea:** retrieve KG evidence conditioned on noisy ASR + answer options, then prompt a (fine-tuned) LLM to generate:

- **Corrected transcript**, and
- **Selected answer option** (A/B/C/D)

---

## 🧠 System Overview

End-to-end pipeline:

- Build **Knowledge Graph** (SQLite + phonetic JSONL) from CSVs
- Build **SFT JSONL** with KG snippets (semantic + phonetic evidence)
- **Full fine-tune** Llama-3.1-8B-Instruct (budgeted KG context)
- Inference: **Whisper Small ASR → KG retrieval → LLM joint correction + MCQ answer**
- Compare multiple inference modes:
  - Whisper → **Non-Fine-Tuned LLM**
  - Whisper → **Fine-Tuned LLM**
  - **GT text** → Non-Fine-Tuned / Fine-Tuned LLM
  - (Optional) enable/disable KG for ablations
- Evaluation: **WER** + **QA accuracy**

---

## 🖼️ Add the Model Figure

Place the overview figure at:

```
assets/medspeak_overview.png
```

(You can export Fig.1 from the paper PDF and save it to that path.)

---

## 🧱 Repo Structure (Important Scripts)

- `scripts/download_and_prepare_benchmarks.py`  
  Download public benchmarks, optionally synthesize audio (TTS), build `manifest.csv`.
- `utils/csv_to_audio_manifest.py`  
  Build `manifest.csv` from your CSVs; run Whisper transcription; generate audio if needed.
- `scripts/prepare_kg.py`  
  Build semantic KG (SQLite) + phonetic KG (JSONL).
- `scripts/build_training_jsonl.py`  
  Build SFT JSONL with KG evidence snippets.
- `scripts/convert_train_jsonl.py`  
  Optional: adjust system context to reduce repeated outputs.
- `scripts/full_finetuning.py`  
  Full fine-tuning for Llama-3.1-8B-Instruct with KG budgets.
- `scripts/infer_medspeak_allmodes.py`  
  Multi-GPU server inference + shard merge for all evaluation modes.

---

## ⚙️ Quickstart

### 1) Environment

```bash
conda create -n medspeak python=3.10 -y
conda activate medspeak
pip install -r requirements.txt
```

### 2) Fetch public benchmarks & create WAVs + manifest

> This script downloads benchmarks from the internet and can create WAVs + a unified manifest.

```bash
python scripts/download_and_prepare_benchmarks.py \
  --mmlu_limit 0 \
  --medmcqa_limit 0 \
  --medqa_limit 0 \
  --tts auto \
  --out_manifest data/qa/manifest.csv
```

### 3) (Optional) Prepare `manifest.csv` from local CSVs

```bash
python utils/csv_to_audio_manifest.py \
  --mmlu_csv data/csv_files/mmlu_qa.csv \
  --medqa_csv data/csv_files/medqa_qa.csv \
  --medmcqa_csv data/csv_files/medmcqa_qa.csv \
  --tts auto \
  --transcribe_whisper small \
  --manifest data/qa/manifest.csv
```

---

## 🧠 Knowledge Graph Preparation (Semantic + Phonetic)

> Replace the CSV paths with your own files if needed.

```bash
python scripts/prepare_kg.py \
  --phonetic_csv data/kg_csv/KG-phonetic.csv \
  --rel_csv data/kg_csv/KG-RELATIONSHIP.csv \
  --rel_csv2 data/kg_csv/SELECT_DISTINCT_t1_Term_AS_Term_Name__r.csv \
  --kg_big_csv data/kg_csv/kg.csv \
  --out_sqlite artifacts/kg_semantic.sqlite \
  --out_phonetic artifacts/kg_phonetic.jsonl
```

---

## 🏗️ Build Training JSONL (SFT with KG Evidence)

```bash
python scripts/build_training_jsonl.py \
  --manifest data/qa/manifest.csv \
  --kg_sql artifacts/kg_semantic.sqlite \
  --kg_phon artifacts/kg_phonetic.jsonl \
  --out_jsonl data/qa/train.jsonl
```

## 🔥 Full Fine-Tuning (Llama-3.1-8B-Instruct)

> Full fine-tuning is GPU-intensive. Adjust epochs/batch/seq_len as needed.

```bash
export TRANSFORMERS_NO_TORCHVISION=1

CUDA_VISIBLE_DEVICES=1,2,4,5 \
python scripts/full_finetuning.py \
  --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --train_jsonl data/qa/train.jsonl \
  --out_dir outputs/fullft-medspeak_hist_server_version \
  --epochs 10 \
  --batch_size 4 \
  --grad_accum 8 \
  --lr 5e-5 \
  --max_seq_len 2048 \
  --kg_sem_budget 600 \
  --kg_phon_budget 300 \
  --show_hist \
  --top_longest 5 \
  --full_finetune
```

---

## 🚀 Inference (Server / All Modes)

This unified entrypoint supports multiple evaluation modes and shard-based multi-GPU inference.

### Mode 1) Zero-shot: GT text + Base LLM (no KG)

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
python scripts/infer_medspeak_allmodes.py infer 
```

### Mode 2) Whisper + Base LLM (audio, no KG)

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
python scripts/infer_medspeak_allmodes.py infer 
```

### Mode 3) MedSpeak: Whisper + KG + Fine-Tuned LLM (audio + KG)

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
python scripts/infer_medspeak_allmodes.py infer 
```

### Mode 4) Fine-Tuned LLM + GT text (optional KG)

#### 4A) No KG

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
python scripts/infer_medspeak_allmodes.py infer 
```

#### 4B) With KG

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
python scripts/infer_medspeak_allmodes.py infer 
```

### (Optional) Manual shard merge anytime

```bash
python scripts/infer_medspeak_allmodes.py merge \
  --shard_dir runs/shards_medspeak_full \
  --out runs/medspeak_full.jsonl \
  --include_existing_final
```

---

## 📊 Evaluation

We report:

- **WER (Word Error Rate)** for ASR correction quality
- **QA Accuracy** for multiple-choice answering correctness

---

## 📌 Notes

- `meta-llama/Meta-Llama-3.1-8B-Instruct` may require access approval (HuggingFace / model license).
- Full fine-tuning is compute heavy; for quick tests reduce:
  - `--epochs`, `--batch_size`, `--max_seq_len`, or use fewer GPUs.