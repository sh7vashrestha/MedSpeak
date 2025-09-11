
import csv, json, subprocess, argparse, os, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/qa/manifest.csv")
    ap.add_argument("--kg_sql", default="artifacts/kg_semantic.sqlite")
    ap.add_argument("--kg_phon", default="artifacts/kg_phonetic.jsonl")
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--lora_dir", default="outputs/lora-medspeak")
    ap.add_argument("--asr_model", default="small")
    ap.add_argument("--preds_out", default="data/qa/preds.jsonl")
    ap.add_argument("--print_context", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.preds_out), exist_ok=True)

    with open(args.manifest, newline="", encoding="utf-8") as f, open(args.preds_out, "w", encoding="utf-8") as out:
        rd = csv.DictReader(f)
        for row in rd:
            options = f"Option A: {row['option_a']} || Option B: {row['option_b']} || Option C: {row['option_c']} || Option D: {row['option_d']}"
            cmd = [
                sys.executable, "scripts/inference_pipeline.py",
                "--audio", row["audio_path"],
                "--options", options,
                "--kg_sql", args.kg_sql,
                "--kg_phon", args.kg_phon,
                "--base_model", args.base_model,
                "--lora_dir", args.lora_dir,
                "--asr_model", args.asr_model,
            ]
            if args.print_context:
                cmd.append("--print_context")
            res = subprocess.run(cmd, capture_output=True, text=True)
            stdout = res.stdout
            corrected, pred = "", ""
            for ln in stdout.splitlines():
                if ln.startswith("Corrected Text:"):
                    corrected = ln.split(":",1)[1].strip()
                if ln.startswith("Correct Option:"):
                    pred = ln.split(":",1)[1].strip().upper()
            rec = {
                "answer": row.get("answer","").strip().upper(),
                "pred_option": pred,
                "corrected_text": corrected,
                "text_gt": row.get("text_gt",""),
                "dataset": row.get("dataset",""),
                "subject": row.get("subject",""),
                "uid": row.get("uid",""),
                "audio_path": row.get("audio_path",""),
            }
            out.write(json.dumps(rec) + "\n")
            print(f"Wrote prediction -> {row.get('dataset','')}/{row.get('subject','')}: {row.get('uid','')}  pred={pred}")
    print(f"All done. Predictions at {args.preds_out}")

if __name__ == "__main__":
    main()
