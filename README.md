# MedSpeak — Knowledge‑Enhanced ASR Error Correction + QA (Llama‑3.1‑8B + Full FineTuning)

End‑to‑end pipeline:
- Build Knowledge Graph (SQLite + phonetic JSONL) from CSVs
- Build SFT JSONL with KG snippets
- Fine‑tune Llama‑3.1‑8B‑Instrucy
- Inference: Whisper Small ASR → KG retrieval → LLM joint correction + MCQ answer
- Inference: Whisper Small ASR -> Non-Fine-tuned LLM, Fine-Tuned LLM and GT-> NFT LLM and FT LLM
- Evaluation: WER + QA accuracy
- Compare them

### Quickstart
```bash
conda create -n medspeak python=3.10 -y
conda activate medspeak
pip install -r requirements.txt

# Fetch public benchmarks & create WAVs + manifest by downloading it from the internet
python scripts/download_and_prepare_benchmarks.py --mmlu_limit 0 --medmcqa_limit 0 --medqa_limit 0 --tts auto --out_manifest data/qa/manifest.csv

#Preapre manifest.jsonl from the csv present in the folder
python utils/csv_to_audio_manifest.py \
    --mmlu_csv data/csv_files/mmlu_qa.csv \
    --medqa_csv data/csv_files/medqa_qa.csv \
    --medmcqa_csv data/csv_files/medmcqa_qa.csv \
    --tts auto \
    --transcribe_whisper small \
    --manifest data/qa/manifest.csv

# Prepare KG (replace CSV paths with your files)
python scripts/prepare_kg.py   --phonetic_csv data/kg_csv/KG-phonetic.csv   --rel_csv data/kg_csv/KG-RELATIONSHIP.csv   --rel_csv2 data/kg_csv/SELECT_DISTINCT_t1_Term_AS_Term_Name__r.csv   --kg_big_csv data/kg_csv/kg.csv   --out_sqlite artifacts/kg_semantic.sqlite   --out_phonetic artifacts/kg_phonetic.jsonl

# Build training JSONL
python scripts/build_training_jsonl.py --manifest data/qa/manifest.csv --kg_sql artifacts/kg_semantic.sqlite --kg_phon artifacts/kg_phonetic.jsonl --out_jsonl data/qa/train.jsonl




# Change the system comtext because the Finetuned LLM produced repeated output
python scripts/convert_train_jsonl.py --in data/qa/train.jsonl --out data/qa/fixv2.jsonl


#Full-Finetune with budgeted KG-context

export TRANSFORMERS_NO_TORCHVISION=1
CUDA_VISIBLE_DEVICES=1,2,4,5 \
python scripts/full_finetuning.py \
  --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --train_jsonl data/qa/train.jsonl \
  --out_dir outputs/fullft-medspeak_hist_server_version \
  --epochs 10 --batch_size 4 --grad_accum 8 --lr 5e-5 \
  --max_seq_len 2048 \
  --kg_sem_budget 600 --kg_phon_budget 300 \
  --show_hist --top_longest 5 \
  --full_finetune




# for server infer for all modes:
1) Zero-shot GT text + LLM (base, no KG)
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
python scripts/infer_medspeak_allmodes.py infer \
  --manifest data/qa/manifest.csv \
  --kg_sql artifacts/kg_semantic.sqlite \
  --kg_phon artifacts/kg_phonetic.jsonl \
  --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --config config_v2.yaml \
  --preds_out runs/zeroshot_gttext_llm.jsonl \
  --shard_out_dir runs/shards_zeroshot_gttext_llm \
  --gpus 1,2,3,4,5,6,7 --workers_per_gpu 1 \
  --periodic_merge_secs 30 --merge_to_live \
  --async_writer --fsync_every 10 \
  --task_q_max 8192 --result_q_max 8192 \
  --gen_max_time 45 --asr_timeout 60 --retries 1 \
  --input_mode text \
  --resume --echo_json

2) Whisper + LLM (audio, no KG)
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
python scripts/infer_medspeak_allmodes.py infer \
  --manifest data/qa/manifest.csv \
  --kg_sql artifacts/kg_semantic.sqlite \
  --kg_phon artifacts/kg_phonetic.jsonl \
  --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --config config_v2.yaml \
  --preds_out runs/whisper_llm.jsonl \
  --shard_out_dir runs/shards_whisper_llm \
  --gpus 1,2,3,4,5,6,7 --workers_per_gpu 1 \
  --periodic_merge_secs 30 --merge_to_live \
  --async_writer --fsync_every 10 \
  --task_q_max 8192 --result_q_max 8192 \
  --gen_max_time 45 --asr_timeout 60 --retries 1 \
  --input_mode audio \
  --resume --echo_json

3) MedSpeak (audio + KG; use your finetuned model)
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
python scripts/infer_medspeak_allmodes.py infer \
  --manifest data/qa/manifest.csv \
  --kg_sql artifacts/kg_semantic.sqlite \
  --kg_phon artifacts/kg_phonetic.jsonl \
  --base_model outputs/fullft-medspeak_hist_server_version \
  --config config_v2.yaml \
  --preds_out runs/medspeak_full.jsonl \
  --shard_out_dir runs/shards_medspeak_full \
  --gpus 1,2,3,4,5,6,7 --workers_per_gpu 1 \
  --periodic_merge_secs 30 --merge_to_live \
  --async_writer --fsync_every 10 \
  --task_q_max 8192 --result_q_max 8192 \
  --gen_max_time 45 --asr_timeout 60 --retries 1 \
  --input_mode audio --use_kg \
  --resume --echo_json

4) FT + GT (finetuned + ground-truth text; optional KG)

No KG:

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
python scripts/infer_medspeak_allmodes.py infer \
  --manifest data/qa/manifest.csv \
  --kg_sql artifacts/kg_semantic.sqlite \
  --kg_phon artifacts/kg_phonetic.jsonl \
  --base_model outputs/fullft-medspeak_hist_server_version \
  --config config_v2.yaml \
  --preds_out runs/ft_gttext_llm.jsonl \
  --shard_out_dir runs/shards_ft_gttext_llm \
  --gpus 1,2,3,4,5,6,7 --workers_per_gpu 1 \
  --periodic_merge_secs 30 --merge_to_live \
  --async_writer --fsync_every 10 \
  --task_q_max 8192 --result_q_max 8192 \
  --gen_max_time 45 --asr_timeout 60 --retries 1 \
  --input_mode text \
  --resume --echo_json


With KG:

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
python scripts/infer_medspeak_allmodes.py infer \
  --manifest data/qa/manifest.csv \
  --kg_sql artifacts/kg_semantic.sqlite \
  --kg_phon artifacts/kg_phonetic.jsonl \
  --base_model outputs/fullft-medspeak_hist_server_version \
  --config config_v2.yaml \
  --preds_out runs/ft_gttext_llm_withkg.jsonl \
  --shard_out_dir runs/shards_ft_gttext_llm_withkg \
  --gpus 1,2,3,4,5,6,7 --workers_per_gpu 1 \
  --periodic_merge_secs 30 --merge_to_live \
  --async_writer --fsync_every 10 \
  --task_q_max 8192 --result_q_max 8192 \
  --gen_max_time 45 --asr_timeout 60 --retries 1 \
  --input_mode text --use_kg \
  --resume --echo_json

(Optional) Manual shard merge anytime
python scripts/infer_medspeak_allmodes.py merge \
  --shard_dir runs/shards_medspeak_full \
  --out runs/medspeak_full.jsonl \
  --include_existing_final
```
