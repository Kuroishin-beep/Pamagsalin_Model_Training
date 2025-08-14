# === Kapampangan-to-English NLLB-200 Training Pipeline (with <kap> tag) ===

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
import torch
import evaluate

# === 1. Config ===
CSV_PATH = "data/kapampangan_english.csv"
MODEL_NAME = "facebook/nllb-200-distilled-600M"
MODEL_DIR = "./kapampangan_mt_nllb"

SPECIAL_SRC_TOKEN = "<kap>"   # Custom Kapampangan language marker
TGT_LANG = "eng_Latn"         # English (Latin script)

# === 2. Load CSV ===
df = pd.read_csv(CSV_PATH)
df = df.rename(columns={"kapampangan": "src_text", "english": "tgt_text"})
df = df.dropna(subset=["src_text", "tgt_text"])

# === 3. Convert to HF Dataset ===
dataset = Dataset.from_pandas(df[["src_text", "tgt_text"]])
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# === 4. Load Tokenizer & Model ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  # No src_lang yet
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Add <kap> as a special token
tokenizer.add_special_tokens({'additional_special_tokens': [SPECIAL_SRC_TOKEN]})
model.resize_token_embeddings(len(tokenizer))

# Force English output
model.config.forced_bos_token_id = tokenizer.lang_code_to_id[TGT_LANG]

# === 5. Preprocess ===
def preprocess(examples):
    # Prepend <kap> to source text
    src_texts = [f"{SPECIAL_SRC_TOKEN} {text}" for text in examples["src_text"]]

    # Tokenize with modern text_target API
    model_inputs = tokenizer(
        src_texts,
        truncation=True,
        padding="max_length",
        max_length=128,
        text_target=examples["tgt_text"]
    )

    # Replace PAD token IDs in labels with -100
    model_inputs["labels"] = [
        [(t if t != tokenizer.pad_token_id else -100) for t in label]
        for label in model_inputs["labels"]
    ]
    return model_inputs

tokenized_dataset = dataset.map(preprocess, batched=True)

# === 6. Training Args ===
training_args = Seq2SeqTrainingArguments(
    output_dir=MODEL_DIR,
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=15,
    weight_decay=0.01,
    predict_with_generate=True,
    save_total_limit=2,
    logging_dir="./logs",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
)

# === 7. Trainer ===
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)

# === 8. Train ===
trainer.train()

# === 9. Save Model & Tokenizer ===
trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print(f"✅ Model saved to: {MODEL_DIR}")

# === 10. Translation (Batched) ===
def batch_translate(texts, batch_size=8):
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for i in range(0, len(texts), batch_size):
        src_texts = [f"{SPECIAL_SRC_TOKEN} {t}" for t in texts[i:i+batch_size]]
        inputs = tokenizer(src_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs)
        results.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return results

# === 11. Evaluate BLEU ===
print("\n--- Evaluating BLEU Score ---")
bleu = evaluate.load("bleu")

test_df = dataset["test"].to_pandas()
preds = batch_translate(test_df["src_text"].tolist())
refs = [[x] for x in test_df["tgt_text"].tolist()]

bleu_score = bleu.compute(predictions=preds, references=refs)
print(" BLEU Score:", bleu_score)

# === 12. Manual Test ===
print("\n--- Manual Test ---")
sample_texts = [
    "Ali ku balu",
    "Anya ka?",
    "Masanting ya ing panaun ngeni",
    "E ku makanyan",
]

for i, kap_text in enumerate(sample_texts):
    translated = batch_translate([kap_text])[0]
    print(f"[{i+1}] Kapampangan: {kap_text}")
    print(f"    ➤ English: {translated}")
