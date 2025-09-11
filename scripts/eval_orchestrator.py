
import argparse, csv, os, sys, json, subprocess

def run(cmd):
    print(">>", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        raise SystemExit(f"Command failed: {' '.join(cmd)}")
    return res.stdout

def parse_stdout(stdout: str):
    corr = ""; opt = ""
    for ln in stdout.splitlines():
        if ln.startswith("Corrected Text:"):
            corr = ln.split(":",1)[1].strip()
        if ln.startswith("Correct Option:"):
            opt = ln.split(":",1)[1].strip().upper()
    return corr, opt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/qa/manifest.csv")
    ap.add_argument("--kg_sql", default="artifacts/kg_semantic.sqlite")
    ap.add_argument("--kg_phon", default="artifacts/kg_phonetic.jsonl")
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--lora_dir", default="outputs/lora-medspeak")
    ap.add_argument("--asr_model", default="small")
    ap.add_argument("--out_dir", default="outputs/eval_runs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    preds_gt = os.path.join(args.out_dir, "preds_gt_zeroshot.jsonl")
    preds_asr = os.path.join(args.out_dir, "preds_asr_lora.jsonl")
    preds_llm = os.path.join(args.out_dir, "preds_llm_only.jsonl")

    # Mode A: Zero-shot on GT text (no LoRA, no KG)
    with open(args.manifest, newline="", encoding="utf-8") as f, open(preds_gt, "w", encoding="utf-8") as out:
        rd = csv.DictReader(f)
        for row in rd:
            options = f"Option A: {row['option_a']} || Option B: {row['option_b']} || Option C: {row['option_c']} || Option D: {row['option_d']}"
            cmd = [
                sys.executable, "scripts/infer_text_prompt.py",
                "--input_text", row["text_gt"],
                "--options", options,
                "--kg_sql", args.kg_sql,
                "--kg_phon", args.kg_phon,
                "--base_model", args.base_model,
                "--no_kg"
            ]
            stdout = run(cmd)
            corr, opt = parse_stdout(stdout)
            rec = {
                "answer": row.get("answer","").upper().strip(),
                "pred_option": opt,
                "corrected_text": corr,
                "text_gt": row.get("text_gt",""),
                "dataset": row.get("dataset",""),
                "subject": row.get("subject",""),
                "uid": row.get("uid",""),
                "mode": "gt_zeroshot"
            }
            out.write(json.dumps(rec)+"\n")

    # Mode B: ASR + LLM with LoRA (full pipeline)
    cmd = [
        sys.executable, "scripts/batch_infer_groups.py",
        "--manifest", args.manifest,
        "--kg_sql", args.kg_sql,
        "--kg_phon", args.kg_phon,
        "--base_model", args.base_model,
        "--lora_dir", args.lora_dir,
        "--asr_model", args.asr_model,
        "--preds_out", preds_asr
    ]
    run(cmd)

    # Mode C: Just LLM on options only (empty ASR text; base model)
    with open(args.manifest, newline="", encoding="utf-8") as f, open(preds_llm, "w", encoding="utf-8") as out:
        rd = csv.DictReader(f)
        for row in rd:
            input_text = ""
            options = f"Option A: {row['option_a']} || Option B: {row['option_b']} || Option C: {row['option_c']} || Option D: {row['option_d']}"
            cmd = [
                sys.executable, "scripts/infer_text_prompt.py",
                "--input_text", input_text,
                "--options", options,
                "--kg_sql", args.kg_sql,
                "--kg_phon", args.kg_phon,
                "--base_model", args.base_model,
                "--no_kg"
            ]
            stdout = run(cmd)
            corr, opt = parse_stdout(stdout)
            rec = {
                "answer": row.get("answer","").upper().strip(),
                "pred_option": opt,
                "corrected_text": corr,
                "text_gt": row.get("text_gt",""),
                "dataset": row.get("dataset",""),
                "subject": row.get("subject",""),
                "uid": row.get("uid",""),
                "mode": "llm_only"
            }
            out.write(json.dumps(rec)+"\n")

    print("Wrote:")
    print(" ", preds_gt)
    print(" ", preds_asr)
    print(" ", preds_llm)

    # Per-group PDFs
    for p in [preds_gt, preds_asr, preds_llm]:
        out_pdf = os.path.join(args.out_dir, os.path.splitext(os.path.basename(p))[0] + "_acc.pdf")
        cmd = [sys.executable, "scripts/evaluate_per_group_pdf.py", "--pred_jsonl", p, "--out_pdf", out_pdf]
        run(cmd)
    print("Per-group accuracy PDFs saved in:", args.out_dir)

if __name__ == "__main__":
    main()
