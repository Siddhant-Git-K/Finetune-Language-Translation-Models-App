
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

df = pd.read_csv("C:\Langauge Translator Project\En-Hi language Translator\Cleaned_Dataset_Final.csv")
df = df[['Hindi', 'English']].dropna()
df.columns = ['hi', 'en']

train_size = int(0.95 * len(df))
train_df = df[:train_size]
val_df = df[train_size:]

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

model_name = "Helsinki-NLP/opus-mt-hi-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def preprocess_function(examples):
    inputs = examples['hi']
    targets = examples['en']
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding=False)
    labels = tokenizer(text_target=targets, max_length=128, truncation=True, padding=False)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=['en', 'hi'])
val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=['en', 'hi'])

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
    output_dir="my_en_translator",
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    logging_steps=100,
    predict_with_generate=True,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
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
if os.path.isdir("my_en_translator"):
    checkpoints = [
        os.path.join("my_en_translator", d)
        for d in os.listdir("my_en_translator")
        if d.startswith("checkpoint")
    ]
    if checkpoints:
        last_checkpoint = max(checkpoints, key=os.path.getmtime)
        print(f"Resuming from: {last_checkpoint}")
    else:
        print("No checkpoint found — starting fresh.")

print("Starting training..")
trainer.train(resume_from_checkpoint=last_checkpoint)


trainer.save_model("my_en_translator")
tokenizer.save_pretrained("my_en_translator")
print("Training complete! Model saved to my_en_translator/")
