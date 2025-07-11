from datasets import load_dataset
from transformers import MarianTokenizer, MarianMTModel, Seq2SeqTrainer, Seq2SeqTrainingArguments

# Load your dataset
dataset = load_dataset("csv", data_files={"train": "kapampangan_english.csv"})["train"]

# Load pre-trained Tagalog-English MarianMT as base (you can fine-tune from this)
model_name = "Helsinki-NLP/opus-mt-tl-en"  # Replace if you want to use a different one
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Tokenize the dataset
def preprocess(example):
    inputs = tokenizer(example["kapampangan"], max_length=128, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example["english"], max_length=128, truncation=True, padding="max_length")
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess, batched=True)

# Set training parameters
training_args = Seq2SeqTrainingArguments(
    output_dir="./kapampangan-translation-model",
    per_device_train_batch_size=8,
    num_train_epochs=10,
    save_total_limit=2,
    save_steps=500,
    evaluation_strategy="no",
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Start training
trainer.train()

# Save the trained model
trainer.save_model("./kapampangan-translation-model")
tokenizer.save_pretrained("./kapampangan-translation-model")

print("\n✅ Translation model saved to ./kapampangan-translation-model")
