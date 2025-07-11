import os 
import torch
from datasets import load_dataset, Audio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer

print("Loading dataset...")
#Step 1: Load and format the datasets
data_files = {"train": "data/validated_audio/metadata.csv"}
dataset = load_dataset("csv", data_files=data_files)
dataset = dataset["train"]
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

print("Loading processor and model...")
#Step 2 load processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-S3")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-S3")

print("Preprocessing dataset...")
#Step 3 Preprocessing
def prepare_batch(batch):
    try:
        audio = batch["audio"]["array"]
        # Process audio with processor
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        
        # Process transcription
        with processor.as_target_processor():
            labels = processor(batch["transcription"]).input_ids
        
        batch["input_values"] = inputs.input_values[0]
        batch["labels"] = torch.tensor(labels)
        return batch
    except Exception as e:
        print(f"Error processing batch: {e}")
        return None

# Filter out None values from processing
processed_dataset = dataset.map(prepare_batch, remove_columns=dataset.column_names)
processed_dataset = processed_dataset.filter(lambda x: x is not None)

print("Setting up training arguments...")
#Step 4 training Arguments
training_args = TrainingArguments(
    output_dir="./wav2vec2-kapampangan",
    per_device_train_batch_size=1,
    num_train_epochs=10,
    logging_steps=10,
    save_steps=500,
    save_total_limit=None,
    fp16=torch.cuda.is_available(),
    learning_rate=3e-4,
    warmup_steps=500,
    weight_decay=0.005,
)

print("Setting up trainer...")
#STEP 5 Defining the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset
)

print("Starting training...")
#Step 6 Train the Model
trainer.train()

print("Saving model...")
model.save_pretrained("./wav2vec2-kapampangan-model")
processor.save_pretrained("./wav2vec2-kapampangan-model")

print("\n✅ Training complete. Model saved to ./wav2vec2-kapampangan-model")