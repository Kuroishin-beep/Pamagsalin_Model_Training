import os
import re
import json
import torch
import torchaudio
import evaluate
import pandas as pd
from datasets import Dataset, Audio
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC
)
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer

# --- Configuration ---
# IMPORTANT: Update these paths
VALIDATED_DATA_FOLDER = 'data/validated_audio' # The folder created by the validation script
MODEL_OUTPUT_DIR = './kapampangan_wav2vec2_model' # Directory to save the trained model
BASE_MODEL = "facebook/wav2vec2-large-xlsr-53" 

# --- 1. Load the Dataset ---
def load_custom_dataset(data_folder):
    """Loads the dataset from the metadata.csv file."""
    metadata_path = os.path.join(data_folder, "metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"metadata.csv not found in {data_folder}. "
            "Please ensure you have run the validation script first."
        )
    dataset_df = pd.read_csv(metadata_path)
    # Convert DataFrame to Hugging Face Dataset object
    custom_dataset = Dataset.from_pandas(dataset_df)
    return custom_dataset

# --- 2. Create Vocabulary ---
def create_vocabulary(data):
    """
    Extracts all unique characters from the transcription column
    and creates a vocabulary file.
    """
    # Regex to extract characters, handling potential variations
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'

    def extract_all_chars(batch):
        all_text = " ".join(batch["transcription"])
        # Normalize and remove special characters
        all_text = re.sub(chars_to_ignore_regex, '', all_text).lower()
        # Create a set of unique characters
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    # Extract vocabulary from the dataset
    vocab_result = data.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=data.column_names
    )

    # Combine all unique characters from all batches
    vocab_list = list(set(vocab_result["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}

    # Add a padding token, which is crucial for CTC loss
    vocab_dict["|"] = vocab_dict.pop(" ")
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    
    # Save the vocabulary as a json file
    vocab_path = os.path.join(MODEL_OUTPUT_DIR, 'vocab.json')
    if not os.path.exists(MODEL_OUTPUT_DIR):
        os.makedirs(MODEL_OUTPUT_DIR)
    with open(vocab_path, 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)
    
    print(f"Vocabulary created and saved to {vocab_path}")
    return vocab_path

# --- 3. Preprocess the Data ---
def preprocess_data(dataset, processor):
    """
    Prepares the dataset for training:
    1. Loads and resamples audio.
    2. Tokenizes transcriptions.
    """
    import librosa
    import soundfile as sf

    def prepare_dataset(batch):
        try:
            # Load audio file directly using librosa
            audio_path = batch["file_path"]
            audio_array, sr = librosa.load(audio_path, sr=16000)
            
            # Process audio to get the input_values
            batch["input_values"] = processor(audio_array, sampling_rate=16000).input_values[0]
            batch["input_length"] = len(batch["input_values"])
            
            # Process text to get the labels
            with processor.as_target_processor():
                batch["labels"] = processor(batch["transcription"]).input_ids
            return batch
        except Exception as e:
            print(f"Error processing {batch['file_path']}: {e}")
            return None

    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)
    # Filter out None values from processing errors
    dataset = dataset.filter(lambda x: x is not None)
    print("Dataset successfully preprocessed.")
    return dataset

# --- 4. Define Metrics and Data Collator ---
class DataCollatorCTCWithPadding:
    """
    Data collator that dynamically pads the inputs and labels for CTC.
    """
    def __init__(self, processor):
        self.processor = processor
        self.padding = "longest"

    def __call__(self, features):
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

def compute_metrics(pred):
    """Computes Word Error Rate (WER) for evaluation."""
    pred_logits = pred.predictions
    pred_ids = torch.argmax(torch.from_numpy(pred_logits), dim=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# --- Main Training Execution ---
if __name__ == '__main__':
    # Step 1: Load data
    print("--- Step 1: Loading Dataset ---")
    raw_dataset = load_custom_dataset(VALIDATED_DATA_FOLDER)
    
    # --- Realistic train/test split for a real scenario ---
    from datasets import DatasetDict
    RANDOM_SEED = 42
    SPLIT_RATIO = 0.1  # 10% for evaluation

    if len(raw_dataset) > 1:
        dataset_split = raw_dataset.train_test_split(
            test_size=SPLIT_RATIO,
            shuffle=True,
            seed=RANDOM_SEED
        )
        train_dataset = dataset_split['train']
        eval_dataset = dataset_split['test']
        print(f"Dataset split into {len(train_dataset)} training samples and {len(eval_dataset)} evaluation samples.")
    else:
        train_dataset = raw_dataset
        eval_dataset = raw_dataset
        print("Warning: Dataset is too small for a split. Evaluating on the training set.")


    # Step 2: Create vocabulary
    print("\n--- Step 2: Creating Vocabulary ---")
    vocab_path = create_vocabulary(train_dataset)

    # Step 3: Setup Processor (Tokenizer + Feature Extractor)
    print("\n--- Step 3: Setting up Processor ---")
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        "./", # Use local directory
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
        vocab_file=vocab_path
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    # Save processor for later use
    processor.save_pretrained(MODEL_OUTPUT_DIR)
    print("Processor created and saved.")

    # Step 4: Preprocess the dataset
    print("\n--- Step 4: Preprocessing Data ---")
    processed_train_dataset = preprocess_data(train_dataset, processor)
    processed_eval_dataset = preprocess_data(eval_dataset, processor)
    
    # Step 5: Setup Trainer
    print("\n--- Step 5: Setting up Model and Trainer ---")
    data_collator = DataCollatorCTCWithPadding(processor=processor)
    wer_metric = evaluate.load("wer")

    model = Wav2Vec2ForCTC.from_pretrained(
        BASE_MODEL,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    # Freeze the feature extraction layers, as they are already well-trained
    model.freeze_feature_encoder()

    # Define training arguments
    # Adjust mo  based on your GPU capacity
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        group_by_length=True,
        per_device_train_batch_size=4, # Lower if you get memory errors
        per_device_eval_batch_size=4,
        num_train_epochs=15, # Increase for better results on a larger dataset
        fp16=True, # Use mixed precision for faster training if you have a compatible GPU
        gradient_checkpointing=True,
        save_steps=500,
        eval_steps=500,
        logging_steps=50,
        learning_rate=3e-4,
        warmup_steps=500,
        save_total_limit=2,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=processed_train_dataset,
        eval_dataset=processed_eval_dataset,
        tokenizer=processor.feature_extractor,
    )

    # Step 6: Train the model
    print("\n--- Step 6: Starting Training ---")
    print("This may take a significant amount of time depending on your hardware and dataset size.")
    trainer.train()

    # Step 7: Save the final model
    print("\n--- Step 7: Saving Final Model ---")
    trainer.save_model(MODEL_OUTPUT_DIR)
    print(f"Training complete! Your fine-tuned model is saved in: {MODEL_OUTPUT_DIR}")

