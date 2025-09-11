
import argparse, json, re, sqlite3, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from yaml import safe_load

def norm(s): return str(s).strip().lower() if s is not None else ""

def shortlist(text, k=40):
    import re
    toks = re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", text or "")
    out, seen = [], set()
    for t in toks:
        t = norm(t)
        if t not in seen:
            seen.add(t); out.append(t)
        if len(out) >= k: break
    return out

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
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try: items.append(json.loads(line))
                except: pass
    except: pass
    return items

def filter_phonetic(all_items, terms, limit=80):
    tset = set(terms)
    out = []
    for it in all_items:
        if it.get("term") in tset:
            cui = it.get("cui")
            out.append(f"{it['term']} ~ {it['similar']}{(' ('+cui+')') if cui else ''}")
            if len(out) >= limit: break
    return out

def greedy(model, tok, prompt, max_new=256):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids, max_new_tokens=max_new, do_sample=False, temperature=0.0, top_p=1.0,
            eos_token_id=tok.eos_token_id
        )
    return tok.decode(out[0], skip_special_tokens=True)

def extract(text: str):
    import re
    m1 = re.search(r"Corrected Text:\s*(.*)", text)
    m2 = re.search(r"Correct Option:\s*([ABCD])", text, re.IGNORECASE)
    return (m1.group(1).strip() if m1 else ""), (m2.group(1).upper() if m2 else "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_text", required=True, help="Text to place in [ASR_TEXT]")
    ap.add_argument("--options", required=True, help="Option A..D as 'Option A: .. || Option B: .. || Option C: .. || Option D: ..'")
    ap.add_argument("--kg_sql", required=True)
    ap.add_argument("--kg_phon", required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--lora_dir", default=None, help="If provided, load adapter; else pure base model")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--print_context", action="store_true")
    ap.add_argument("--no_kg", action="store_true", help="Skip KG lookups")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = safe_load(f)

    parts = [p.strip() for p in args.options.split("||")]
    if len(parts) != 4: raise SystemExit("--options must have 4 parts separated by '||'")
    optA = parts[0].split(":",1)[-1].strip()
    optB = parts[1].split(":",1)[-1].strip()
    optC = parts[2].split(":",1)[-1].strip()
    optD = parts[3].split(":",1)[-1].strip()

    if args.no_kg:
        sem = "(none)"; phon = "(none)"
    else:
        conn = sqlite3.connect(args.kg_sql)
        terms = shortlist(args.input_text + "\n" + "\n".join([optA,optB,optC,optD]))
        sem = "; ".join(query_semantic(conn, terms)) or "(none)"
        phon_all = load_phonetic(args.kg_phon)
        phon = "; ".join(filter_phonetic(phon_all, terms)) or "(none)"

    user_prompt = cfg["prompt_template"].format(
        asr_text=args.input_text, opt_a=optA, opt_b=optB, opt_c=optC, opt_d=optD,
        kg_sem=sem, kg_phon=phon
    )
    full = f"<|system|>\n{cfg['system_prompt']}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
    if args.print_context:
        print("===== CONTEXT =====")
        print(user_prompt)
        print("===================")

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto", trust_remote_code=True
    )
    if args.lora_dir:
        model = PeftModel.from_pretrained(model, args.lora_dir)

    out_text = greedy(model, tok, full, max_new=args.max_new_tokens)
    corr, opt = extract(out_text)
    print(f"Corrected Text: {corr}")
    print(f"Correct Option: {opt}")

if __name__ == "__main__":
    main()
