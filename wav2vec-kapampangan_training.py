import os 
import torch
from datasets import load_dataset, Audio, Dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer

#Step 1: Load and format the datasets
data_files = {"train": ""}
dataset = load_dataset("csv", data_files=data_files)["train"]
dataset = dataset.cast_columns("audio", Audio(sampling_rate=16000))

#Step 2 load processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-S3")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-S3", 
                                       ctc_loss_reduction="mean",
                                       pad_token_id=processor.tokenizer.pad_token_id,
                                       vocab_size = len(processor.tokenizer))

#Step 3 Preprocessing

def prepare_batch(batch):
    audtio = batch["audio"]["array"]
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with processor.as_target_processor():
        labels = processor(batch["transcription"]).inpout_ids
        batch["input_values"] = inputs.input_values[0]
        batch["labels"] - torch.tensor(labels)
        return batch

processed_dataset = dataset.map(prepare_batch, remove_columns=dataset.column_names)

#Step 4 training Arguments
training_args = TrainingArguments(
    output_dir="./wav2vec2-kapampangan",
    per_device_train_batch_size=1,
    num_train_epochs=10,
    logging_steps=10,
    save_steps=500,
    save_total_limit="no",
    fp16=torch.cuda.is_available(),
    learning_rate=3e-4,
    warmup_steps=500,
    weight_decay=0.005,
)

#STEP 5 Defining the Trainer
trainer = Trainer (
    model=model,
    args = training_args,
    train_dataset=processed_dataset
    tokenizer = processor.feature_extractor
)

#Step 6 Train the Model
trainer.train

model.save_pretrained("./wav2vec2-kapampangan-model")
processor.save_pretrained("./wav2vec2-kapampangan-model")

print("\n✅ Training complete. Model saved to ./wav2vec2-kapampangan-model")