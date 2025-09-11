
import argparse
import json
from jiwer import wer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", required=True)
    ap.add_argument("--ref_field", required=True)
    ap.add_argument("--hyp_field", required=True)
    args = ap.parse_args()

    refs, hyps = [], []
    with open(args.pred_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            refs.append(str(r.get(args.ref_field, "")))
            hyps.append(str(r.get(args.hyp_field, "")))

    score = wer(refs, hyps)
    print(f"WER: {score:.4f}")

if __name__ == "__main__":
    main()
