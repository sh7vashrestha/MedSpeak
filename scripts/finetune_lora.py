import argparse, os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel
from trl import SFTTrainer
from datasets import load_dataset

def format_chat(rec):
    """Flatten 'messages' into one training string."""
    msgs = rec["messages"]
    out = ""
    for m in msgs:
        out += f"<|{m['role']}|>\n{m['content'].strip()}\n"
    return [out]   # must return list[str]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--use_4bit", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Tokenizer (still needed for saving, not passed to trainer)
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Model
    if args.use_4bit:
        qconf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, quantization_config=qconf,
            device_map="auto", trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto", trust_remote_code=True
        )

    # LoRA
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # Dataset
    ds = load_dataset("json", data_files=args.train_jsonl, split="train")

    # Trainer (no tokenizer arg)
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        formatting_func=format_chat,
        args=TrainingArguments(
            output_dir=args.out_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            logging_steps=10,
            save_strategy="epoch",
            bf16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            report_to=[],
        ),
    )

    trainer.train()

    # Save LoRA adapter + tokenizer
    if isinstance(model, PeftModel):
        model.save_pretrained(args.out_dir)
    else:
        model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"[DONE] LoRA adapter + tokenizer saved to {args.out_dir}")

if __name__ == "__main__":
    main()
