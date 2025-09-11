
import argparse, json, os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", default="data/qa/preds.jsonl")
    ap.add_argument("--out_pdf", default="outputs/accuracy_by_group.pdf")
    ap.add_argument("--group_cols", default="dataset,subject")
    args = ap.parse_args()

    rows = []
    with open(args.pred_jsonl, "r", encoding="utf-8") as f:
        for ln in f:
            rows.append(json.loads(ln))
    if not rows:
        print("No rows found in preds.jsonl"); return

    df = pd.DataFrame(rows)
    df["answer"] = df["answer"].astype(str).str.upper().str.strip()
    df["pred_option"] = df["pred_option"].astype(str).str.upper().str.strip()

    groups = [c.strip() for c in args.group_cols.split(",") if c.strip()]
    for c in groups:
        if c not in df.columns:
            df[c] = ""

    grp = df.groupby(groups).apply(lambda g: (g["pred_option"] == g["answer"]).mean()).reset_index(name="accuracy")
    grp["n"] = df.groupby(groups).size().values
    grp = grp.sort_values(groups)

    os.makedirs(os.path.dirname(args.out_pdf), exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(8, len(grp)*0.3), 0.6*len(grp)+1))
    ax.axis('off')
    header = groups + ["N", "Accuracy"]
    table_data = [header]
    for _, r in grp.iterrows():
        row = [str(r[g]) for g in groups] + [int(r["n"]), f"{r['accuracy']:.3f}"]
        table_data.append(row)
    table = ax.table(cellText=table_data, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)
    fig.tight_layout()
    fig.savefig(args.out_pdf, bbox_inches='tight')
    print(f"Saved per-group accuracy PDF to {args.out_pdf}")

if __name__ == "__main__":
    main()
