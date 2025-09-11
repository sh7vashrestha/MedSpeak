
import argparse
import json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", required=True)
    args = ap.parse_args()

    total = 0
    correct = 0
    counts = {"A":0, "B":0, "C":0, "D":0}

    with open(args.pred_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            gold = str(r.get("answer", "")).strip().upper()
            pred = str(r.get("pred_option", "")).strip().upper()
            if gold in counts:
                counts[gold] += 1
            if gold and pred:
                total += 1
                if gold == pred:
                    correct += 1

    acc = correct / total if total else 0.0
    print(f"Total: {total}  Correct: {correct}  Accuracy: {acc:.4f}")
    print("Counts (gold):", counts)

if __name__ == "__main__":
    main()
