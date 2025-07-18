# Kapampangan-to-English MarianMT Training Pipeline

## 1. Configuration and Imports
import pandas as pd
from datasets import Dataset
from transformers import (
    MarianTokenizer,
    MarianMTModel,
)
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.data.data_collator import DataCollatorForSeq2Seq
import torch
import evaluate

CSV_PATH = "data/kapampangan_english.csv"
MODEL_NAME = "Helsinki-NLP/opus-mt-en-ROMANCE"
MODEL_DIR = "./kapampangan_mt_model"

## 2. Load and Clean CSV
df = pd.read_csv(CSV_PATH)
df = df.rename(columns={"kapampangan": "src_text", "english": "tgt_text"})
df = df.dropna(subset=["src_text", "tgt_text"])

## 3. Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df[["src_text", "tgt_text"]])
dataset = dataset.train_test_split(test_size=0.2, seed=42)

## 4. Load Tokenizer and Base Model
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)

## 5. Tokenization Function
def preprocess(example):
    model_inputs = tokenizer(
        example["src_text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["tgt_text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess, batched=True)

## 6. Define Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=MODEL_DIR,
    learning_rate=5e-6,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=30,
    weight_decay=0.01,
    predict_with_generate=True,
    save_total_limit=2,
    logging_dir="./logs",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
)

## 7. Trainer Setup
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)

## 8. Train the Model
trainer.train()

## 9. Save Final Model
trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print(f"✅ Model saved to: {MODEL_DIR}")

## 10. Define Translation Function
def kapampangan_translate(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

## 11. Evaluate BLEU Score
print("\n--- Evaluating BLEU Score ---")
bleu = evaluate.load("bleu")

preds = [kapampangan_translate(x) for x in df["src_text"]]
refs = [[x] for x in df["tgt_text"]]

bleu_score = bleu.compute(predictions=preds, references=refs)
print(" BLEU Score:", bleu_score)

## 12. Manual Translation Test
print("\n--- Manual Test ---")
sample_texts = [
    "Ali ku balu",
    "Anya ka?",
    "Masanting ya ing panaun ngeni",
    "E ku makanyan",
]

for i, kap_text in enumerate(sample_texts):
    translated = kapampangan_translate(kap_text)
    print(f"[{i+1}] Kapampangan: {kap_text}")
    print(f"    \u27a4 English: {translated}")
