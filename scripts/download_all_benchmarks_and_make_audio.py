#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_all_benchmarks_and_make_audio.py

What it does:
1) Downloads benchmarks to CSV:
   - MMLU medical subjects (per-subject CSV + combined CSV)
   - MedQA (combined CSV)
   - MedMCQA (combined CSV)
2) Synthesizes 16 kHz mono WAV audio for every question:
   - TTS via pyttsx3 if available; otherwise valid silence WAVs
3) (Optional) Runs Whisper Small ASR to produce transcripts and appends 'asr_text' to the manifest
4) Writes/extends data/qa/manifest.csv with columns:
   audio_path, text_gt, question, option_a..d, answer, dataset, subject, uid, asr_text

Usage (full run, with Whisper ASR):
  python scripts/download_all_benchmarks_and_make_audio.py \
    --mmlu_subjects clinical_knowledge,anatomy,college_medicine,college_biology,medical_genetics,professional_medicine \
    --mmlu_split test \
    --tts auto \
    --include_medqa \
    --include_medmcqa \
    --transcribe_whisper small \
    --manifest data/qa/manifest.csv

If you want to skip ASR to go faster:
  --transcribe_whisper none

You can re-run; it will append new rows. Delete manifest to start fresh.

Requirements:
- datasets, soundfile, numpy, (optional) pyttsx3, (optional) openai-whisper, torchaudio, ffmpeg installed
"""

import argparse
import os
import csv
import time
import sys
from typing import List, Dict, Tuple, Optional

import numpy as np
import soundfile as sf
from datasets import load_dataset

# ---------- Optional imports ----------
try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

try:
    import whisper  # openai-whisper
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

SR = 16000

# ------------------- Utils -------------------

def ensure_dirs():
    for p in [
        "data/audio/benchmarks",
        "data/qa",
        "data/csv_files/mmlu",
        "data/csv_files/medqa",
        "data/csv_files/medmcqa",
    ]:
        os.makedirs(p, exist_ok=True)

def clean(s: str) -> str:
    return " ".join(str(s or "").split()).strip()

def is_letter(x: str) -> bool:
    return str(x).strip().upper()[:1] in {"A","B","C","D"}

def to_letter(x) -> Optional[str]:
    if isinstance(x, int):
        if 0 <= x <= 3:
            return "ABCD"[x]
    xx = str(x).strip().upper()
    return xx[:1] if (xx and xx[0] in "ABCD") else None

def save_silence_wav(path: str, seconds: float = 3.0):
    n = int(SR * max(0.5, seconds))
    audio = np.zeros((n,), dtype=np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, audio, SR, subtype="PCM_16")

def tts_to_wav(text: str, path: str) -> str:
    """
    Returns 'tts' or 'silence', but always writes a valid 16k mono wav.
    """
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
        # make sure it's 16k mono
        try:
            y, sr = sf.read(path, always_2d=False)
            if isinstance(y, np.ndarray) and y.ndim > 1:
                y = y.mean(axis=1)
            if sr != SR:
                x_old = np.linspace(0, 1, num=len(y), endpoint=False)
                x_new = np.linspace(0, 1, num=int(len(y) * SR / sr), endpoint=False)
                y = np.interp(x_new, x_old, y).astype(np.float32)
            sf.write(path, y, SR, subtype="PCM_16")
        except Exception:
            save_silence_wav(path, seconds=3.0)
            return "silence"
        return "tts"
    except Exception:
        save_silence_wav(path, seconds=max(2.0, 0.18*len(text.split())))
        return "silence"

def write_csv_rows(rows: List[Dict[str, str]], out_csv: str, header: List[str]):
    if not rows:
        # still create an empty file with header
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=header)
            wr.writeheader()
        print(f"[CSV] Wrote {out_csv} (rows=0)")
        return
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=header)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"[CSV] Wrote {out_csv} (rows={len(rows)})")

def append_manifest(rows: List[Dict[str, str]], manifest_csv: str):
    cols = ["audio_path","text_gt","question","option_a","option_b","option_c","option_d","answer","dataset","subject","uid","asr_text"]
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

# ------------------- MMLU -------------------

MMLU_DEFAULT_SUBJECTS = [
    "clinical_knowledge",
    "anatomy",
    "college_medicine",
    "college_biology",
    "medical_genetics",
    "professional_medicine",
]

def fetch_mmlu_subject(subj: str, split: str) -> List[Dict[str, str]]:
    """
    Returns rows with keys: dataset, subject, uid, question, option_a..d, answer, text_gt
    """
    # Try CAIS/MMLU first, then hendrycks_test fallback
    ds = None
    tried = []
    for repo in [("cais/mmlu", {"name": subj, "split": split}),
                 ("hendrycks_test", {"name": subj, "split": split})]:
        try:
            ds = load_dataset(repo[0], **repo[1])
            tried.append(repo[0])
            break
        except Exception:
            tried.append(repo[0])
            continue
    if ds is None:
        print(f"[MMLU][WARN] Could not load subject {subj} from: {tried}", file=sys.stderr)
        return []

    rows = []
    data = ds  # Dataset object (split-loaded)
    for idx, ex in enumerate(data):
        q = clean(ex.get("question",""))
        # choices can be a list or A/B/C/D keys
        ch = []
        if "choices" in ex and isinstance(ex["choices"], (list, tuple)):
            ch = [clean(x) for x in ex["choices"][:4]]
        else:
            ch = [clean(ex.get("A","")), clean(ex.get("B","")), clean(ex.get("C","")), clean(ex.get("D",""))]
        ch = (ch + ["","","",""])[:4]

        ans = ex.get("answer", ex.get("target"))
        letter = to_letter(ans)
        if letter is None:
            # skip invalid row
            continue

        rows.append({
            "dataset": "MMLU",
            "subject": subj,
            "uid": f"mmlu::{subj}::{idx}",
            "question": q,
            "option_a": ch[0], "option_b": ch[1], "option_c": ch[2], "option_d": ch[3],
            "answer": letter,
            "text_gt": q,  # for our use, the clean target transcript is the question text
        })
    return rows

def save_mmlu_csvs(all_by_subject: Dict[str, List[Dict[str, str]]]):
    header = ["dataset","subject","uid","question","option_a","option_b","option_c","option_d","answer","text_gt"]
    # per subject
    for subj, rows in all_by_subject.items():
        out_csv = os.path.join("data/csv_files/mmlu", f"{subj}.csv")
        write_csv_rows(rows, out_csv, header)
    # combined
    combined = [r for _, rows in all_by_subject.items() for r in rows]
    out_all = "data/csv_files/mmlu/mmlu.csv"
    write_csv_rows(combined, out_all, header)

# ------------------- MedQA -------------------

def fetch_medqa_all() -> List[Dict[str, str]]:
    """
    Try several MedQA mirrors. Returns combined rows or empty list.
    """
    trials = [
        ("bigbio/med_qa", {"name": "med_qa_en", "split": "validation"}),
        ("bigbio/med_qa", {"name": "med_qa_en", "split": "test"}),
        ("openlifescienceai/medical_meadow_medqa_4options", {"split": "validation"}),
        ("openlifescienceai/medical_meadow_medqa_4options", {"split": "test"}),
    ]
    rows: List[Dict[str, str]] = []
    for repo, kwargs in trials:
        try:
            ds = load_dataset(repo, **kwargs)
            data = next(iter(ds.values())) if isinstance(ds, dict) else ds
            for i, ex in enumerate(data):
                q = clean(ex.get("question", ex.get("prompt","")))
                choices = []
                # prefer structured keys
                for k in ["option_a","option_b","option_c","option_d","A","B","C","D","opa","opb","opc","opd"]:
                    if k in ex: choices.append(clean(ex[k]))
                if not choices and "choices" in ex:
                    choices = [clean(c) for c in ex["choices"][:4]]
                choices = (choices + ["","","",""])[:4]

                ans = ex.get("answer", ex.get("label", ex.get("cop","")))
                letter = to_letter(ans)
                if letter is None:
                    continue

                rows.append({
                    "dataset": "MedQA",
                    "subject": "general",
                    "uid": f"medqa::{kwargs.get('split','unk')}::{i}",
                    "question": q,
                    "option_a": choices[0], "option_b": choices[1], "option_c": choices[2], "option_d": choices[3],
                    "answer": letter,
                    "text_gt": q,
                })
        except Exception as e:
            print(f"[MedQA][WARN] {repo} {kwargs} failed: {e}", file=sys.stderr)
            continue
    return rows

def save_medqa_csv(rows: List[Dict[str, str]]):
    header = ["dataset","subject","uid","question","option_a","option_b","option_c","option_d","answer","text_gt"]
    out_csv = "data/csv_files/medqa/medqa.csv"
    write_csv_rows(rows, out_csv, header)

# ------------------- MedMCQA -------------------

def fetch_medmcqa_all() -> List[Dict[str, str]]:
    """
    Try several MedMCQA mirrors. Returns combined rows or empty list.
    """
    trials = [
        ("openlifescienceai/medical_meadow_medmcqa_4options", {"split": "validation"}),
        ("openlifescienceai/medical_meadow_medmcqa_4options", {"split": "test"}),
        ("medmcqa", {"split": "validation"}),  # might not exist in all hubs; best-effort
        ("medmcqa", {"split": "test"}),
    ]
    rows: List[Dict[str, str]] = []
    for repo, kwargs in trials:
        try:
            ds = load_dataset(repo, **kwargs)
            data = next(iter(ds.values())) if isinstance(ds, dict) else ds
            for i, ex in enumerate(data):
                q = clean(ex.get("question", ex.get("prompt","")))
                choices = []
                for k in ["option_a","option_b","option_c","option_d","A","B","C","D","opa","opb","opc","opd"]:
                    if k in ex: choices.append(clean(ex[k]))
                if not choices and "choices" in ex:
                    choices = [clean(c) for c in ex["choices"][:4]]
                choices = (choices + ["","","",""])[:4]

                ans = ex.get("answer", ex.get("label", ex.get("cop","")))
                letter = to_letter(ans)
                if letter is None:
                    continue

                rows.append({
                    "dataset": "MedMCQA",
                    "subject": "general",
                    "uid": f"medmcqa::{kwargs.get('split','unk')}::{i}",
                    "question": q,
                    "option_a": choices[0], "option_b": choices[1], "option_c": choices[2], "option_d": choices[3],
                    "answer": letter,
                    "text_gt": q,
                })
        except Exception as e:
            print(f"[MedMCQA][WARN] {repo} {kwargs} failed: {e}", file=sys.stderr)
            continue
    return rows

def save_medmcqa_csv(rows: List[Dict[str, str]]):
    header = ["dataset","subject","uid","question","option_a","option_b","option_c","option_d","answer","text_gt"]
    out_csv = "data/csv_files/medmcqa/medmcqa.csv"
    write_csv_rows(rows, out_csv, header)

# ------------------- Audio + Manifest + ASR -------------------

def synthesize_audio_for_items(items: List[Dict[str, str]], audio_root: str, tts_mode: str) -> List[Dict[str, str]]:
    manifest_rows = []
    for i, it in enumerate(items):
        q = it["question"]
        txt = it.get("text_gt", q)
        subj = it["subject"]
        dsname = it["dataset"]
        subdir = os.path.join(audio_root, dsname, subj.replace(" ", "_"))
        os.makedirs(subdir, exist_ok=True)
        fname = f"{dsname.lower()}_{subj.replace(' ','_')}_{i:07d}.wav"
        apath = os.path.join(subdir, fname)

        if tts_mode == "tts":
            tts_to_wav(txt, apath)
        else:
            save_silence_wav(apath, seconds=max(2.0, 0.18 * len(txt.split())))

        manifest_rows.append(row_to_manifest(
            audio_path=apath,
            question=it["question"],
            opts=[it["option_a"], it["option_b"], it["option_c"], it["option_d"]],
            ans=it["answer"],
            ds=dsname,
            subj=subj,
            uid=it["uid"],
            text_gt=txt,
            asr_text="",  # will be filled later if ASR is enabled
        ))
    return manifest_rows

def run_whisper_on_manifest(manifest_csv: str, model_size: str = "small", batch: int = 16):
    """
    Appends/updates 'asr_text' for each row using Whisper.
    If whisper is not available, warns and returns.
    """
    if not WHISPER_AVAILABLE:
        print("[ASR] openai-whisper not installed; skipping ASR.", file=sys.stderr)
        return

    print(f"[ASR] Loading Whisper model: {model_size}")
    asr = whisper.load_model(model_size)

    # Load manifest rows
    with open(manifest_csv, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
        fieldnames = rdr.fieldnames

    # Transcribe
    for idx, r in enumerate(rows):
        apath = r["audio_path"]
        # Skip if already present
        if r.get("asr_text"):
            continue
        try:
            # whisper expects a file path; it loads audio internally
            result = asr.transcribe(apath, task="transcribe", fp16=True)
            r["asr_text"] = clean(result.get("text", ""))
        except Exception as e:
            print(f"[ASR][WARN] failed on {apath}: {e}", file=sys.stderr)
            r["asr_text"] = ""

        if (idx + 1) % 50 == 0:
            print(f"[ASR] {idx+1}/{len(rows)} done...")

    # Write back manifest with updated asr_text
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"[ASR] Updated manifest with ASR text -> {manifest_csv}")

# ------------------- Main -------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmlu_subjects", type=str,
                    default=",".join(MMLU_DEFAULT_SUBJECTS),
                    help="Comma-separated MMLU subjects")
    ap.add_argument("--mmlu_split", type=str, default="test")
    ap.add_argument("--include_medqa", action="store_true")
    ap.add_argument("--include_medmcqa", action="store_true")
    ap.add_argument("--tts", choices=["auto","silence","tts"], default="auto")
    ap.add_argument("--transcribe_whisper", type=str, default="none",
                    help="one of: none | tiny | base | small | medium | large (Whisper sizes)")
    ap.add_argument("--manifest", type=str, default="data/qa/manifest.csv")
    args = ap.parse_args()

    ensure_dirs()

    # Decide TTS mode
    tts_mode = "tts" if (args.tts == "tts" or (args.tts == "auto" and TTS_AVAILABLE)) else "silence"
    print(f"[INFO] TTS mode: {tts_mode} (pyttsx3 {'available' if TTS_AVAILABLE else 'NOT available'})")
    print(f"[INFO] Whisper ASR: {args.transcribe_whisper} (installed={WHISPER_AVAILABLE})")

    # ----------------- MMLU -----------------
    subjects = [s.strip() for s in args.mmlu_subjects.split(",") if s.strip()]
    all_by_subject: Dict[str, List[Dict[str, str]]] = {}
    for subj in subjects:
        print(f"[MMLU] Fetching subject: {subj} (split={args.mmlu_split})")
        rows = fetch_mmlu_subject(subj, split=args.mmlu_split)
        all_by_subject[subj] = rows
    save_mmlu_csvs(all_by_subject)

    # ----------------- MedQA -----------------
    medqa_rows: List[Dict[str, str]] = []
    if args.include_medqa:
        print("[MedQA] Fetching...")
        medqa_rows = fetch_medqa_all()
        save_medqa_csv(medqa_rows)

    # ----------------- MedMCQA -----------------
    medmcqa_rows: List[Dict[str, str]] = []
    if args.include_medmcqa:
        print("[MedMCQA] Fetching...")
        medmcqa_rows = fetch_medmcqa_all()
        save_medmcqa_csv(medmcqa_rows)

    # ----------------- Audio + Manifest -----------------
    all_items = [r for _, rows in all_by_subject.items() for r in rows] + medqa_rows + medmcqa_rows
    print(f"[AUDIO] Synthesizing WAVs for {len(all_items)} items...")
    mani_rows = synthesize_audio_for_items(all_items, "data/audio/benchmarks", tts_mode=tts_mode)
    append_manifest(mani_rows, args.manifest)

    # ----------------- Whisper ASR (optional) -----------------
    if args.transcribe_whisper.lower() != "none":
        run_whisper_on_manifest(args.manifest, model_size=args.transcribe_whisper)

    print("\n[DONE]")
    print(f"  Manifest: {args.manifest}")
    print(f"  CSV roots: data/csv_files/mmlu/, data/csv_files/medqa/, data/csv_files/medmcqa/")
    if tts_mode != "tts":
        print("  Note: Used silence WAV placeholders. Install 'pyttsx3' and rerun with --tts tts for spoken audio.")
    if args.transcribe_whisper.lower() != "none" and not WHISPER_AVAILABLE:
        print("  Note: Whisper not installed. pip install openai-whisper torchaudio ffmpeg-python")
    print("  You can now build training JSONL and fine-tune.")
    print("    e.g., python scripts/build_training_jsonl.py --manifest data/qa/manifest.csv --kg_sql artifacts/kg_semantic.sqlite --kg_phon artifacts/kg_phonetic.jsonl --out_jsonl data/qa/train.jsonl --asr_noise none")

if __name__ == "__main__":
    main()
