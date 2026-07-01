import pandas as pd
from datasets import Dataset
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import torch
import evaluate
import numpy as np
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

df = pd.read_csv(r"C:\Langauge Translator Project\English_tamil.csv")
df = df[['English', 'Tamil']].dropna()
df.columns = ['en', 'ta']

train_size = int(0.95 * len(df))
train_df = df[:train_size]
val_df = df[train_size:]

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

model_name = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def preprocess_function(examples):
    inputs = examples['en']   # English input ✔
    targets = examples['ta']  # Tamil output ✔

    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    labels = tokenizer(text_target=targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=['en', 'ta'])
val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=['en', 'ta'])

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

bleu_metric = evaluate.load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = bleu_metric.compute(
        predictions=decoded_preds,
        references=[[label] for label in decoded_labels]
    )
    return {"bleu": round(result["score"], 2)}

training_args = Seq2SeqTrainingArguments(
    output_dir="en_ta_model",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    logging_steps=100,
    predict_with_generate=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

import os
last_checkpoint = None
if os.path.isdir("model"):
    checkpoints = [
        os.path.join("model", d)
        for d in os.listdir("model")
        if d.startswith("checkpoint")
    ]
    if checkpoints:
        last_checkpoint = max(checkpoints, key=os.path.getmtime)
        print(f"Resuming from: {last_checkpoint}")
    else:
        print("No checkpoint found — starting fresh.")

print("Starting training...")
trainer.train(resume_from_checkpoint=last_checkpoint)


trainer.save_model("model")
tokenizer.save_pretrained("model")
print("Training complete! Model saved to model/")