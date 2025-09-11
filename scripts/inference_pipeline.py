
import argparse
import json
import re
import sqlite3

import torch
import whisper
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from yaml import safe_load

def norm(s: str) -> str:
    return str(s).strip().lower() if s is not None else ""

def shortlist_terms(text: str, k: int = 40):
    toks = re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", text)
    uniq, seen = [], set()
    for t in toks:
        t = norm(t)
        if t not in seen:
            seen.add(t); uniq.append(t)
        if len(uniq) >= k:
            break
    return uniq

def query_semantic(conn, terms, limit_per=5):
    out = []
    for t in terms:
        q = f"SELECT term, rel, rel_detail, related_term FROM kg WHERE term LIKE ? OR related_term LIKE ? LIMIT {limit_per}"
        for row in conn.execute(q, (f"%{t}%", f"%{t}%")):
            term, rel, rel_detail, related = row
            if rel_detail:
                out.append(f"{term} -[ {rel}:{rel_detail} ]-> {related}")
            else:
                out.append(f"{term} -[ {rel} ]-> {related}")
    return out[:100]

def load_phonetic(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except Exception:
                pass
    return items

def filter_phonetic(all_items, terms, limit=80):
    tset = set(terms)
    out = []
    for it in all_items:
        if it.get("term") in tset:
            cui = it.get("cui")
            out.append(f"{it['term']} ~ {it['similar']}{(' ('+cui+')') if cui else ''}")
        if len(out) >= limit:
            break
    return out

def greedy_generate(model, tok, prompt, max_new_tokens=256):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            eos_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0], skip_special_tokens=True)

def extract_outputs(text: str):
    m1 = re.search(r"Corrected Text:\s*(.*)", text)
    m2 = re.search(r"Correct Option:\s*([ABCD])", text, re.IGNORECASE)
    corrected = m1.group(1).strip() if m1 else ""
    opt = m2.group(1).upper() if m2 else ""
    return corrected, opt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--options", required=True, help="Use ' || ' between A..D")
    ap.add_argument("--kg_sql", required=True)
    ap.add_argument("--kg_phon", required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--lora_dir", required=True)
    ap.add_argument("--asr_model", default="small")
    ap.add_argument("--print_context", action="store_true")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    asr = whisper.load_model(args.asr_model)
    asr_res = asr.transcribe(args.audio, language="en")
    asr_text = asr_res["text"].strip()

    parts = [p.strip() for p in args.options.split("||")]
    if len(parts) != 4:
        raise ValueError("--options must contain 4 parts separated by '||'")
    optA = parts[0].split(":", 1)[-1].strip()
    optB = parts[1].split(":", 1)[-1].strip()
    optC = parts[2].split(":", 1)[-1].strip()
    optD = parts[3].split(":", 1)[-1].strip()

    with open(args.config, "r") as f:
        cfg = safe_load(f)

    conn = sqlite3.connect(args.kg_sql)
    terms = shortlist_terms(asr_text + "\n" + "\n".join([optA,optB,optC,optD]))
    sem = query_semantic(conn, terms)
    phon_all = load_phonetic(args.kg_phon)
    phon = filter_phonetic(phon_all, terms)

    user_prompt = cfg["prompt_template"].format(
        asr_text=asr_text, opt_a=optA, opt_b=optB, opt_c=optC, opt_d=optD,
        kg_sem="; ".join(sem) if sem else "(none)",
        kg_phon="; ".join(phon) if phon else "(none)",
    )
    full_prompt = f"<|system|>\n{cfg['system_prompt']}\n<|user|>\n{user_prompt}\n<|assistant|>\n"

    if args.print_context:
        print("===== CONTEXT =====")
        print(user_prompt)
        print("===================")

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.lora_dir)

    out_text = greedy_generate(model, tok, full_prompt, max_new_tokens=args.max_new_tokens)

    corrected, pred_opt = extract_outputs(out_text)

    print(f"Corrected Text: {corrected}")
    print(f"Correct Option: {pred_opt}")

if __name__ == "__main__":
    main()
