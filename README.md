# MedSpeak — Knowledge‑Enhanced ASR Error Correction + QA (Llama‑3.1‑8B + LoRA)

End‑to‑end pipeline:
- Build Knowledge Graph (SQLite + phonetic JSONL) from CSVs
- Build SFT JSONL with KG snippets
- Fine‑tune Llama‑3.1‑8B‑Instruct with LoRA
- Inference: Whisper Small ASR → KG retrieval → LLM joint correction + MCQ answer
- Evaluation: WER + QA accuracy (grouped PDFs)
- Benchmarks downloader: MMLU / MedMCQA / MedQA to `data/csv_files/` + WAVs

### Quickstart
```bash
conda create -n medspeak python=3.10 -y
conda activate medspeak
pip install -r requirements.txt

# Fetch public benchmarks & create WAVs + manifest
python scripts/download_and_prepare_benchmarks.py --mmlu_limit 50 --medmcqa_limit 50 --medqa_limit 50 --tts auto --out_manifest data/qa/manifest.csv

# Prepare KG (replace CSV paths with your files)
python scripts/prepare_kg.py   --phonetic_csv data/kg_csv/KG-phonetic.csv   --rel_csv data/kg_csv/KG-RELATIONSHIP.csv   --rel_csv2 data/kg_csv/SELECT_DISTINCT_t1_Term_AS_Term_Name__r.csv   --kg_big_csv data/kg_csv/kg.csv   --out_sqlite artifacts/kg_semantic.sqlite   --out_phonetic artifacts/kg_phonetic.jsonl

# Build training JSONL
python scripts/build_training_jsonl.py --manifest data/qa/manifest.csv --kg_sql artifacts/kg_semantic.sqlite --kg_phon artifacts/kg_phonetic.jsonl --out_jsonl data/qa/train.jsonl

# Fine‑tune LoRA
python scripts/finetune_lora.py --base_model meta-llama/Meta-Llama-3.1-8B-Instruct --train_jsonl data/qa/train.jsonl --out_dir outputs/lora-medspeak --epochs 1 --batch_size 2 --grad_accum 8 --lr 1e-4 --use_4bit

# Batch inference (ASR→KG→LLM) to preds.jsonl
python scripts/batch_infer_groups.py --manifest data/qa/manifest.csv --kg_sql artifacts/kg_semantic.sqlite --kg_phon artifacts/kg_phonetic.jsonl --base_model meta-llama/Meta-Llama-3.1-8B-Instruct --lora_dir outputs/lora-medspeak --asr_model small --preds_out data/qa/preds.jsonl

# Evaluate
python scripts/compute_wer.py --pred_jsonl data/qa/preds.jsonl --ref_field text_gt --hyp_field corrected_text
python scripts/evaluate_qa.py --pred_jsonl data/qa/preds.jsonl
python scripts/evaluate_per_group_pdf.py --pred_jsonl data/qa/preds.jsonl --out_pdf outputs/accuracy_by_group.pdf

# Or run 3-mode evaluation orchestrator (GT‑zeroshot, ASR+LoRA, LLM‑only)
python scripts/eval_orchestrator.py --manifest data/qa/manifest.csv --kg_sql artifacts/kg_semantic.sqlite --kg_phon artifacts/kg_phonetic.jsonl --base_model meta-llama/Meta-Llama-3.1-8B-Instruct --lora_dir outputs/lora-medspeak --asr_model small --out_dir outputs/eval_runs
```

## [Just Got Accepted In ICASSP 2026](https://2026.ieeeicassp.org/)
