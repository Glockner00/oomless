import argparse
import tempfile

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

torch.cuda.set_per_process_memory_fraction(0.5)

parser = argparse.ArgumentParser(description="Fine-tune GPT-2 small on WikiText-2.")
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--max-length", type=int, default=128, help="tokens per text")
parser.add_argument("--steps", type=int, default=300)
parser.add_argument(
    "--bug",
    choices=["keep-outputs"],
    help="keep-outputs: keep every evaluation output on the GPU to compute the accuracy",
)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2")

data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1").filter(
    lambda row: row["text"].strip()
)
data = data.map(
    lambda rows: tokenizer(
        rows["text"], truncation=True, max_length=args.max_length, padding="max_length"
    ),
    batched=True,
    remove_columns=["text"],
)


def accuracy(pred: EvalPrediction) -> dict[str, float]:
    preds, labels = pred.predictions, pred.label_ids
    if preds.ndim == 3:
        preds = preds.argmax(-1)
    preds, labels = preds[:, :-1], labels[:, 1:]
    mask = labels != -100
    return {"accuracy": float((preds[mask] == labels[mask]).mean())}


trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=tempfile.mkdtemp(),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        max_steps=args.steps,
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
    ),
    train_dataset=data["train"],
    eval_dataset=data["validation"].select(range(500)),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    compute_metrics=accuracy,
    preprocess_logits_for_metrics=None
    if args.bug == "keep-outputs"
    else lambda logits, _labels: logits.argmax(-1),
)
trainer.train()
print(f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
