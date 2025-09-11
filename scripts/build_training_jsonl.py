#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_training_jsonl.py

Create SFT training JSONL for MedSpeak:
- Input: manifest CSV with columns:
    audio_path, text_gt, question, option_a, option_b, option_c, option_d, answer
    (optional but supported: dataset, subject, uid)
- KG inputs:
    - SQLite semantic graph with table:
        kg(term, term_cui, rel, rel_detail, related_term, related_cui)
    - JSONL phonetic pairs with objects:
        {"term": "...", "similar": "...", "cui": "..."}
- Output JSONL with fields:
    {
      "messages": [ {role, content}... ],
      "assistant_target": "...",
      "answer": "A|B|C|D",
      "corrected_text_ref": "<text_gt>",
      "text_gt": "<text_gt>",
      "audio_path": "...",
      "dataset": "...",
      "subject": "...",
      "uid": "..."
    }

Usage:
  python scripts/build_training_jsonl.py \
    --manifest data/qa/manifest.csv \
    --kg_sql artifacts/kg_semantic.sqlite \
    --kg_phon artifacts/kg_phonetic.jsonl \
    --out_jsonl data/qa/train.jsonl \
    --asr_noise light

Notes:
- Robust to header case/whitespace variants.
- Semantic query uses simple LIKE across shortlist terms (k<=40).
- Phonetic filtering matches shortlist terms (lowercased).
- Optional light noise injection for ASR_TEXT.
"""

import argparse
import csv
import json
import os
import re
import sqlite3
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

# -------------------- Utilities --------------------

def norm(s: str) -> str:
    return " ".join(str(s or "").strip().split()).lower()

def normalize_letter(x: str) -> Optional[str]:
    if x is None:
        return None
    x = str(x).strip().upper()
    if not x:
        return None
    c = x[0]
    return c if c in {"A","B","C","D"} else None

def safe_get(row: Dict[str,str], *keys, default="") -> str:
    for k in keys:
        if k in row:
            return row[k]
        # case/spacing tolerant
        for kk in row.keys():
            if norm(kk) == norm(k):
                return row[kk]
    return default

def extract_terms(question: str, options: List[str], k: int = 40) -> List[str]:
    """
    Build a shortlist of candidate terms from question + options.
    Tokenize on non-letters; keep words with length >= 3.
    Deduplicate but preserve order. Return up to k terms (lowercased).
    """
    text = " ".join([question] + options)
    # keep alnum and spaces
    toks = re.findall(r"[A-Za-z][A-Za-z\-']{1,}", text)
    # Also capture multi-word candidates around ' of ' & ' for ' constructs
    # (very light heuristic)
    mw = re.findall(r"([A-Za-z][A-Za-z\-']{1,}\s+(?:of|for)\s+[A-Za-z][A-Za-z\-']{1,})", text, re.IGNORECASE)
    cand = toks + mw
    out = []
    seen = set()
    for t in cand:
        t2 = norm(t)
        if len(t2) >= 3 and t2 not in seen:
            seen.add(t2)
            out.append(t2)
        if len(out) >= k:
            break
    return out

def light_noise(s: str) -> str:
    """
    Tiny synthetic ASR noise: drop punctuation, occasional char subs, lowercasing.
    Keeps it readable but slightly noisy.
    """
    x = s
    # drop punctuation (basic)
    x = re.sub(r"[.,;:!?()]", "", x)
    # simple homophone-ish tweaks
    subs = [
        (r"\bph\b", "f"),
        (r"\bmedic(al)?\b", "medik"),
        (r"\bhemo", "heemo"),
        (r"\bcardia", "kardia"),
    ]
    for pat, rep in subs:
        x = re.sub(pat, rep, x, flags=re.IGNORECASE)
    # random-ish lowercasing without RNG for determinism: alternate chars
    chars = list(x)
    for i in range(0, len(chars), 7):
        chars[i] = chars[i].lower()
    return "".join(chars)

# -------------------- KG access --------------------

def iter_phonetic_pairs(phon_jsonl_path: str) -> Iterable[Tuple[str, str, str]]:
    """
    Stream phonetic JSONL; yield (term, similar, cui) in lowercase.
    """
    with open(phon_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            term = norm(obj.get("term", ""))
            sim = norm(obj.get("similar", ""))
            cui = norm(obj.get("cui", ""))
            if term and sim:
                yield (term, sim, cui)

def collect_phonetic_snippets(phon_jsonl_path: str, shortlist: List[str], max_pairs: int = 80) -> str:
    """
    Filter phonetic pairs where 'term' is in shortlist. Build compact string:
      "<term> ~ <similar> (CUI); ..."
    """
    wanted = set(shortlist)
    out = []
    count = 0
    for term, sim, cui in iter_phonetic_pairs(phon_jsonl_path):
        if term in wanted:
            seg = f"{term} ~ {sim}" + (f" ({cui})" if cui else "")
            out.append(seg)
            count += 1
            if count >= max_pairs:
                break
    return "; ".join(out) if out else ""

def collect_semantic_snippets(sqlite_path: str, shortlist: List[str], max_edges: int = 120) -> str:
    """
    Query SQLite table 'kg' for LIKE matches against terms in shortlist.
    Returns compact edges:
       "<term> -[ REL:detail ]-> <related_term>; ..."
    """
    if not os.path.exists(sqlite_path):
        return ""
    con = sqlite3.connect(sqlite_path)
    con.row_factory = sqlite3.Row
    out = []
    try:
        cur = con.cursor()
        grabbed = 0
        # Prefer exact term matches before broad LIKE
        for t in shortlist:
            # exact-ish (case-insensitive)
            cur.execute(
                """
                SELECT term, rel, rel_detail, related_term
                FROM kg
                WHERE LOWER(term) = ? OR LOWER(related_term) = ?
                LIMIT 50
                """,
                (t, t),
            )
            for r in cur.fetchall():
                term = norm(r["term"])
                rel = norm(r["rel"])
                detail = norm(r["rel_detail"])
                rel_txt = f"{rel}" + (f":{detail}" if detail else "")
                rt = norm(r["related_term"])
                out.append(f"{term} -[ {rel_txt} ]-> {rt}")
                grabbed += 1
                if grabbed >= max_edges:
                    return "; ".join(out)

        # fallback broad LIKE
        for t in shortlist:
            like = f"%{t}%"
            cur.execute(
                """
                SELECT term, rel, rel_detail, related_term
                FROM kg
                WHERE LOWER(term) LIKE ? OR LOWER(related_term) LIKE ?
                LIMIT 50
                """,
                (like, like),
            )
            for r in cur.fetchall():
                term = norm(r["term"])
                rel = norm(r["rel"])
                detail = norm(r["rel_detail"])
                rel_txt = f"{rel}" + (f":{detail}" if detail else "")
                rt = norm(r["related_term"])
                out.append(f"{term} -[ {rel_txt} ]-> {rt}")
                grabbed += 1
                if grabbed >= max_edges:
                    return "; ".join(out)

    finally:
        con.close()

    return "; ".join(out) if out else ""

# -------------------- Builder --------------------

SYSTEM_PROMPT = (
    "You are MedSpeak, a medical ASR error correction and QA assistant.\n"
    "You will receive:\n"
    "- ASR noisy text\n"
    "- Multiple-choice options (A..D)\n"
    "- Knowledge snippets: semantic relations and phonetic pairs\n"
    "Your task:\n"
    "1) Correct the ASR text if needed (medical terminology matters).\n"
    "2) Choose the correct option (A/B/C/D) using the knowledge and context.\n"
    "Produce outputs in the exact two-line format with no extra text:\n"
    "Corrected Text: <final corrected transcript>\n"
    "Correct Option: <A|B|C|D>\n"
)

def make_user_prompt(asr_text: str,
                     options: List[str],
                     kg_sem: str,
                     kg_phon: str) -> str:
    parts = []
    parts.append("[ASR_TEXT]")
    parts.append(asr_text.strip())
    parts.append("\n[OPTIONS]")
    parts.append(f"Option A: {options[0]}")
    parts.append(f"Option B: {options[1]}")
    parts.append(f"Option C: {options[2]}")
    parts.append(f"Option D: {options[3]}")
    parts.append("\n[KG_SEMANTIC]")
    parts.append(kg_sem if kg_sem else "(none)")
    parts.append("\n[KG_PHONETIC]")
    parts.append(kg_phon if kg_phon else "(none)")
    parts.append("\nFollow the system instructions strictly and output exactly:")
    parts.append("Corrected Text: <...>")
    parts.append("Correct Option: <A|B|C|D>")
    return "\n".join(parts)

def build_record(row: Dict[str,str],
                 kg_sql: str,
                 kg_phon: str,
                 asr_noise: str = "none") -> Optional[Dict]:
    """
    Create one JSONL record from one manifest row.
    """
    # normalize/collect fields
    text_gt   = safe_get(row, "text_gt")
    question  = safe_get(row, "question")
    option_a  = safe_get(row, "option_a")
    option_b  = safe_get(row, "option_b")
    option_c  = safe_get(row, "option_c")
    option_d  = safe_get(row, "option_d")
    gold      = normalize_letter(safe_get(row, "answer"))
    if gold is None:
        return None

    dataset   = safe_get(row, "dataset")
    subject   = safe_get(row, "subject")
    uid       = safe_get(row, "uid")
    audio     = safe_get(row, "audio_path")

    # shortlist terms
    shortlist = extract_terms(question, [option_a, option_b, option_c, option_d], k=40)

    # KG snippets
    kg_sem = collect_semantic_snippets(kg_sql, shortlist, max_edges=120) if kg_sql else ""
    kg_phn = collect_phonetic_snippets(kg_phon, shortlist, max_pairs=80) if kg_phon else ""

    # ASR text (noisy or clean)
    asr_text = text_gt.strip()
    if asr_noise == "light":
        asr_text = light_noise(asr_text)

    # build assistant target (gold)
    assistant_target = f"Corrected Text: {text_gt.strip()}\nCorrect Option: {gold}"

    # user prompt
    user_prompt = make_user_prompt(
        asr_text=asr_text,
        options=[option_a, option_b, option_c, option_d],
        kg_sem=kg_sem,
        kg_phon=kg_phn
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
        {"role": "assistant", "content": assistant_target},
    ]

    rec = {
        "messages": messages,
        "assistant_target": assistant_target,
        "answer": gold,
        "corrected_text_ref": text_gt.strip(),
        "text_gt": text_gt.strip(),
        "audio_path": audio,
        "dataset": dataset,
        "subject": subject,
        "uid": uid,
    }
    return rec

# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Path to data/qa/manifest.csv")
    ap.add_argument("--kg_sql", required=True, help="Path to artifacts/kg_semantic.sqlite")
    ap.add_argument("--kg_phon", required=True, help="Path to artifacts/kg_phonetic.jsonl")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL path")
    ap.add_argument("--asr_noise", choices=["none","light"], default="none",
                    help="If 'light', inject small noise into ASR_TEXT.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    # Read manifest
    rows: List[Dict[str,str]] = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)

    total = 0
    kept = 0
    skipped_bad_label = 0

    with open(args.out_jsonl, "w", encoding="utf-8") as out:
        for r in rows:
            total += 1
            rec = build_record(r, args.kg_sql, args.kg_phon, asr_noise=args.asr_noise)
            if rec is None:
                skipped_bad_label += 1
                continue
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[BUILD] Manifest rows: {total}")
    print(f"[BUILD] Kept: {kept}")
    print(f"[BUILD] Skipped (invalid answer not in A-D): {skipped_bad_label}")
    print(f"[BUILD] Wrote: {args.out_jsonl}")

if __name__ == "__main__":
    main()
