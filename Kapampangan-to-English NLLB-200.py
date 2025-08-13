# === Kapampangan-to-English NLLB-200 Training Pipeline (with <kap> tag) ===

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.data.data_collator import DataCollatorForSeq2Seq
import torch
import evaluate

# === 1. Config ===
CSV_PATH = "data/kapampangan_english.csv"
MODEL_NAME = "facebook/nllb-200-distilled-600M"  # NLLB model
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
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang="eng_Latn")  # temp placeholder
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Add <kap> as a special token
tokenizer.add_special_tokens({'additional_special_tokens': [SPECIAL_SRC_TOKEN]})
model.resize_token_embeddings(len(tokenizer))

# === 5. Preprocess ===
def preprocess(examples):
    # Prepend <kap> tag to source text
    src_texts = [f"{SPECIAL_SRC_TOKEN} {text}" for text in examples["src_text"]]
    
    # Tokenize source
    model_inputs = tokenizer(
        src_texts,
        truncation=True,
        padding="max_length",
        max_length=128
    )

    # Tokenize target (English)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["tgt_text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    # Replace pad token IDs in labels with -100
    labels_input_ids = [
        [(t if t != tokenizer.pad_token_id else -100) for t in label]
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels_input_ids

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

# === 9. Save ===
trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print(f"✅ Model saved to: {MODEL_DIR}")

# === 10. Translation Function ===
def kapampangan_translate(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    src_text = f"{SPECIAL_SRC_TOKEN} {text}"
    inputs = tokenizer(src_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(TGT_LANG)  # Force English output
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# === 11. Evaluate BLEU ===
print("\n--- Evaluating BLEU Score ---")
bleu = evaluate.load("bleu")

preds = [kapampangan_translate(x) for x in df["src_text"]]
refs = [[x] for x in df["tgt_text"]]

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
    translated = kapampangan_translate(kap_text)
    print(f"[{i+1}] Kapampangan: {kap_text}")
    print(f"    ➤ English: {translated}")
