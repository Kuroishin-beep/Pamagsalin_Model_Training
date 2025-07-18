%%writefile kapampangan_wav2vec2_training.ipynb
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Kapampangan Speech Recognition with Wav2Vec2\n",
    "## Complete Training Pipeline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Setup and Dependencies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install torch torchaudio transformers datasets evaluate pandas librosa soundfile"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import json\n",
    "import re\n",
    "import pandas as pd\n",
    "import torch\n",
    "import torchaudio\n",
    "import librosa\n",
    "import soundfile as sf\n",
    "from datasets import Dataset, DatasetDict\n",
    "from transformers import (\n",
    "    Wav2Vec2CTCTokenizer,\n",
    "    Wav2Vec2FeatureExtractor,\n",
    "    Wav2Vec2Processor,\n",
    "    Wav2Vec2ForCTC,\n",
    "    TrainingArguments,\n",
    "    Trainer\n",
    ")\n",
    "import evaluate\n",
    "from tqdm import tqdm"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Configuration"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# @markdown ### Dataset Configuration\n",
    "NUM_SPEAKERS = 40\n",
    "NUM_CATEGORIES = 14\n",
    "SENTENCES_PER_CATEGORY = 20\n",
    "TEST_SPLIT_RATIO = 0.2\n",
    "RANDOM_SEED = 42\n",
    "\n",
    "# @markdown ### Model Configuration\n",
    "BASE_MODEL = \"facebook/wav2vec2-large-xlsr-53\"\n",
    "MODEL_DIR = \"./kapampangan_wav2vec2_model\"\n",
    "AUDIO_FOLDER = \"data/kapampangan_audio\"\n",
    "METADATA_FILE = \"data/kapampangan_metadata.json\"\n",
    "\n",
    "# Create directories if they don't exist\n",
    "os.makedirs(AUDIO_FOLDER, exist_ok=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Dataset Preparation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# @markdown ### Create Metadata File\n",
    "\n",
    "def generate_metadata():\n",
    "    \"\"\"Generates metadata JSON file for the Kapampangan dataset.\"\"\"\n",
    "    metadata = []\n",
    "    \n",
    "    for speaker in range(1, NUM_SPEAKERS + 1):\n",
    "        for category in range(1, NUM_CATEGORIES + 1):\n",
    "            for sentence_num in range(1, SENTENCES_PER_CATEGORY + 1):\n",
    "                file_name = f\"spk_{speaker:02d}_cat_{category:02d}_sent_{sentence_num:02d}.wav\"\n",
    "                \n",
    "                # Example transcription format - replace with actual Kapampangan sentences\n",
    "                transcription = f\"Example Kapampangan sentence {sentence_num} from speaker {speaker} in category {category}\"\n",
    "                \n",
    "                metadata.append({\n",
    "                    \"file_path\": os.path.join(AUDIO_FOLDER, file_name),\n",
    "                    \"transcription\": transcription,\n",
    "                    \"speaker_id\": speaker,\n",
    "                    \"category_id\": category,\n",
    "                    \"sentence_id\": sentence_num\n",
    "                })\n",
    "    \n",
    "    with open(METADATA_FILE, 'w', encoding='utf-8') as f:\n",
    "        json.dump(metadata, f, indent=4, ensure_ascii=False)\n",
    "    \n",
    "    return len(metadata)\n",
    "\n",
    "total_samples = generate_metadata()\n",
    "print(f\"Generated metadata for {total_samples} audio samples\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Training Pipeline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# @markdown ### Load Dataset\n",
    "def load_dataset(metadata_file):\n",
    "    \"\"\"Loads dataset from metadata JSON file.\"\"\"\n",
    "    with open(metadata_file, 'r', encoding='utf-8') as f:\n",
    "        data = json.load(f)\n",
    "    \n",
    "    df = pd.DataFrame(data)\n",
    "    dataset = Dataset.from_pandas(df)\n",
    "    \n",
    "    # Train-test split\n",
    "    dataset = dataset.train_test_split(\n",
    "        test_size=TEST_SPLIT_RATIO,\n",
    "        shuffle=True,\n",
    "        seed=RANDOM_SEED\n",
    "    )\n",
    "    \n",
    "    return dataset\n",
    "\n",
    "dataset = load_dataset(METADATA_FILE)\n",
    "print(f\"Dataset loaded with {len(dataset['train'])} training and {len(dataset['test'])} test samples\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# @markdown ### Create Vocabulary\n",
    "def create_vocab(dataset):\n",
    "    \"\"\"Extracts vocabulary from transcriptions.\"\"\"\n",
    "    chars_to_ignore_regex = r\"[\\,\\?\\.\\!\\-\\;\\:\\\"\\'\\%\\[\\]\\–\\—\\”\\„\\“]\u200b\"\n",
    "    \n",
    "    def extract_all_chars(batch):\n",
    "        all_text = \" \".join(batch[\"transcription\"])\n",
    "        all_text = re.sub(chars_to_ignore_regex, '', all_text).lower()\n",
    "        vocab = list(set(all_text))\n",
    "        return {\"vocab\": [vocab], \"all_text\": [all_text]}\n",
    "    \n",
    "    vocab_result = dataset[\"train\"].map(\n",
    "        extract_all_chars,\n",
    "        batched=True,\n",
    "        batch_size=-1,\n",
    "        keep_in_memory=True,\n",
    "        remove_columns=dataset[\"train\"].column_names\n",
    "    )\n",
    "    \n",
    "    vocab_list = list(set(vocab_result[\"vocab\"][0]))\n",
    "    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}\n",
    "    \n",
    "    # Special tokens\n",
    "    vocab_dict[\"|\"] = vocab_dict.pop(\" \") if \" \" in vocab_dict else len(vocab_dict)\n",
    "    vocab_dict[\"[UNK]\"] = len(vocab_dict)\n",
    "    vocab_dict[\"[PAD]\"] = len(vocab_dict)\n",
    "    \n",
    "    # Save vocabulary\n",
    "    vocab_path = os.path.join(MODEL_DIR, \"vocab.json\")\n",
    "    os.makedirs(MODEL_DIR, exist_ok=True)\n",
    "    \n",
    "    with open(vocab_path, \"w\") as vocab_file:\n",
    "        json.dump(vocab_dict, vocab_file)\n",
    "    \n",
    "    print(f\"Vocabulary created with {len(vocab_dict)} items\")\n",
    "    return vocab_path\n",
    "\n",
    "vocab_path = create_vocab(dataset)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# @markdown ### Initialize Processor\n",
    "tokenizer = Wav2Vec2CTCTokenizer(\n",
    "    vocab_file=vocab_path,\n",
    "    unk_token=\"[UNK]\",\n",
    "    pad_token=\"[PAD]\",\n",
    "    word_delimiter_token=\"|\"\n",
    ")\n",
    "\n",
    "feature_extractor = Wav2Vec2FeatureExtractor(\n",
    "    feature_size=1,\n",
    "    sampling_rate=16000,\n",
    "    padding_value=0.0,\n",
    "    do_normalize=True,\n",
    "    return_attention_mask=False\n",
    ")\n",
    "\n",
    "processor = Wav2Vec2Processor(\n",
    "    feature_extractor=feature_extractor,\n",
    "    tokenizer=tokenizer\n",
    ")\n",
    "\n",
    "# Save processor\n",
    "processor.save_pretrained(MODEL_DIR)\n",
    "print(\"Processor initialized and saved\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# @markdown ### Data Preprocessing\n",
    "def prepare_dataset(batch):\n",
    "    \"\"\"Processes audio files and transcriptions.\"\"\"\n",
    "    try:\n",
    "        # Load audio\n",
    "        audio_array, sampling_rate = torchaudio.load(batch[\"file_path\"])\n",
    "        \n",
    "        # Resample if needed\n",
    "        if sampling_rate != 16000:\n",
    "            resampler = torchaudio.transforms.Resample(\n",
    "                orig_freq=sampling_rate,\n",
    "                new_freq=16000\n",
    "            )\n",
    "            audio_array = resampler(audio_array)\n",
    "        \n",
    "        # Squeeze and convert to numpy\n",
    "        audio_array = audio_array.squeeze().numpy()\n",
    "        \n",
    "        # Process audio\n",
    "        batch[\"input_values\"] = processor(\n",
    "            audio_array, \n",
    "            sampling_rate=16000\n",
    "        ).input_values[0]\n",
    "        \n",
    "        # Process text\n",
    "        with processor.as_target_processor():\n",
    "            batch[\"labels\"] = processor(batch[\"transcription\"]).input_ids\n",
    "        \n",
    "        batch[\"input_length\"] = len(batch[\"input_values\"])\n",
    "        \n",
    "    except Exception as e:\n",
    "        print(f\"Error processing {batch.get('file_path', 'unknown')}: {str(e)}\")\n",
    "        return None\n",
    "    \n",
    "    return batch\n",
    "\n",
    "# Process datasets\n",
    "def preprocess_datasets(dataset):\n",
    "    \"\"\"Applies preprocessing to train and test sets.\"\"\"\n",
    "    processed_train = []\n",
    "    processed_test = []\n",
    "    \n",
    "    # Process train set\n",
    "    for item in tqdm(dataset[\"train\"], desc=\"Processing training set\"):\n",
    "        processed_item = prepare_dataset(item)\n",
    "        if processed_item is not None:\n",
    "            processed_train.append(processed_item)\n",
    "    \n",
    "    # Process test set\n",
    "    for item in tqdm(dataset[\"test\"], desc=\"Processing test set\"):\n",
    "        processed_item = prepare_dataset(item)\n",
    "        if processed_item is not None:\n",
    "            processed_test.append(processed_item)\n",
    "    \n",
    "    # Create new datasets\n",
    "    train_dataset = Dataset.from_list(processed_train)\n",
    "    test_dataset = Dataset.from_list(processed_test)\n",
    "    \n",
    "    return DatasetDict({\n",
    "        \"train\": train_dataset,\n",
    "        \"test\": test_dataset\n",
    "    })\n",
    "\n",
    "processed_dataset = preprocess_datasets(dataset)\n",
    "print(f\"Preprocessing complete. Train: {len(processed_dataset['train'])}, Test: {len(processed_dataset['test'])}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# @markdown ### Training Setup\n",
    "# Custom Data Collator\n",
    "class DataCollatorCTCWithPadding:\n",
    "    \"\"\"Data collator for dynamic padding of inputs and labels.\"\"\"\n",
    "    def __init__(self, processor):\n",
    "        self.processor = processor\n",
    "    \n",
    "    def __call__(self, features):\n",
    "        input_features = [{\"input_values\": feature[\"input_values\"]} for feature in features]\n",
    "        label_features = [{\"input_ids\": feature[\"labels\"]} for feature in features]\n",
    "        \n",
    "        batch = self.processor.pad(\n",
    "            input_features,\n",
    "            padding=\"longest\",\n",
    "            return_tensors=\"pt\"\n",
    "        )\n",
    "        \n",
    "        with self.processor.as_target_processor():\n",
    "            labels_batch = self.processor.pad(\n",
    "                label_features,\n",
    "                padding=\"longest\",\n",
    "                return_tensors=\"pt\"\n",
    "            )\n",
    "        \n",
    "        # Replace padding with -100 for loss computation\n",
    "        labels = labels_batch[\"input_ids\"].masked_fill(\n",
    "            labels_batch.attention_mask.ne(1), -100\n",
    "        )\n",
    "        \n",
    "        batch[\"labels\"] = labels\n",
    "        return batch\n",
    "\n",
    "# Initialize data collator\n",
    "data_collator = DataCollatorCTCWithPadding(processor=processor)\n",
    "\n",
    "# Load evaluation metric\n",
    "wer_metric = evaluate.load(\"wer\")\n",
    "\n",
    "# Compute metrics function\n",
    "def compute_metrics(pred):\n",
    "    \"\"\"Computes Word Error Rate (WER).\"\"\"\n",
    "    pred_logits = pred.predictions\n",
    "    pred_ids = torch.argmax(torch.from_numpy(pred_logits), dim=-1)\n",
    "    \n",
    "    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id\n",
    "    \n",
    "    pred_str = processor.tokenizer.batch_decode(pred_ids)\n",
    "    label_str = processor.tokenizer.batch_decode(\n",
    "        pred.label_ids, \n",
    "        group_tokens=False\n",
    "    )\n",
    "    \n",
    "    wer = wer_metric.compute(predictions=pred_str, references=label_str)\n",
    "    return {\"wer\": wer}\n",
    "\n",
    "# Initialize model\n",
    "model = Wav2Vec2ForCTC.from_pretrained(\n",
    "    BASE_MODEL,\n",
    "    ctc_loss_reduction=\"mean\",\n",
    "    pad_token_id=processor.tokenizer.pad_token_id,\n",
    "    vocab_size=len(processor.tokenizer)\n",
    ")\n",
    "\n",
    "# Freeze feature encoder layers\n",
    "model.freeze_feature_encoder()\n",
    "\n",
    "# Training arguments\n",
    "training_args = TrainingArguments(\n",
    "    output_dir=MODEL_DIR,\n",
    "    group_by_length=True,\n",
    "    per_device_train_batch_size=8,\n",
    "    per_device_eval_batch_size=8,\n",
    "    gradient_accumulation_steps=4,\n",
    "    evaluation_strategy=\"steps\",\n",
    "    num_train_epochs=30,\n",
    "    fp16=torch.cuda.is_available(),\n",
    "    save_steps=500,\n",
    "    eval_steps=500,\n",
    "    logging_steps=100,\n",
    "    learning_rate=3e-4,\n",
    "    warmup_steps=200,\n",
    "    save_total_limit=3,\n",
    "    push_to_hub=False,\n",
    "    load_best_model_at_end=True,\n",
    "    metric_for_best_model=\"wer\",\n",
    "    greater_is_better=False,\n",
    "    report_to=\"none\"\n",
    ")\n",
    "\n",
    "# Initialize Trainer\n",
    "trainer = Trainer(\n",
    "    model=model,\n",
    "    data_collator=data_collator,\n",
    "    args=training_args,\n",
    "    compute_metrics=compute_metrics,\n",
    "    train_dataset=processed_dataset[\"train\"],\n",
    "    eval_dataset=processed_dataset[\"test\"]\n",
    ")\n",
    "\n",
    "print(\"Training setup complete\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# @markdown ### Training Execution\n",
    "print(\"Starting training...\")\n",
    "\n",
    "training_results = trainer.train()\n",
    "\n",
    "# Save final model\n",
    "trainer.save_model(MODEL_DIR)\n",
    "processor.save_pretrained(MODEL_DIR)\n",
    "print(f\"Training complete. Model saved to {MODEL_DIR}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# @markdown ### Evaluation and Results\n",
    "# Load best model\n",
    "model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR).to(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "\n",
    "# Evaluate\n",
    "eval_results = trainer.evaluate()\n",
    "\n",
    "print(\"\\n=== Final Evaluation Results ===\")\n",
    "print(f\"Word Error Rate (WER): {eval_results['eval_wer']:.2%}\")\n",
    "print(f\"Loss: {eval_results['eval_loss']:.4f}\")\n",
    "\n",
    "# Sample predictions\n",
    "print(\"\\n=== Sample Predictions ===\")\n",
    "test_sample = processed_dataset[\"test\"].select(range(5))\n",
    "\n",
    "def map_to_result(batch):\n",
    "    with torch.no_grad():\n",
    "        input_values = torch.tensor(batch[\"input_values\"][0], device=model.device).unsqueeze(0)\n",
    "        logits = model(input_values).logits\n",
    "    \n",
    "    pred_ids = torch.argmax(logits, dim=-1)\n",
    "    batch[\"pred_str\"] = processor.batch_decode(pred_ids)[0]\n",
    "    batch[\"text\"] = processor.decode(batch[\"labels\"], group_tokens=False)\n",
    "    \n",
    "    return batch\n",
    "\n",
    "results = test_sample.map(map_to_result)\n",
    "\n",
    "for i, result in enumerate(results):\n",
    "    print(f\"\\nSample {i+1}:\")\n",
    "    print(f\"Prediction: {result['pred_str']}\")\n",
    "    print(f\"Reference: {result['text']}\")\n",
    "\n",
    "# Save evaluation results\n",
    "with open(os.path.join(MODEL_DIR, \"evaluation_results.json\"), \"w\") as f:\n",
    "    json.dump({\"wer\": eval_results[\"eval_wer\"], \"loss\": eval_results[\"eval_loss\"]}, f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# @markdown ### Export for MarianMT Training\n",
    "\n",
    "# Create CSV file for MarianMT training\n",
    "print(\"\\nCreating CSV dataset for MarianMT training...\")\n",
    "\n",
    "def create_marianmt_dataset(original_metadata, processed_dataset, output_path):\n",
    "    \"\"\"Creates a CSV with Kapampangan-english pairs for MarianMT.\"\"\"\n",
    "    # Load original transcriptions\n",
    "    with open(original_metadata, 'r', encoding='utf-8') as f:\n",
    "        metadata = json.load(f)\n",
    "    \n",
    "    # For this example, we'll assume source is Kapampangan\n",
    "    # In reality you would have the English translations or generate dummy ones\n",
    "    kapampangan_texts = [item[\"transcription\"] for item in metadata]\n",
    "    # English translations - in a real scenario these would be your actual translations\n",
    "    english_texts = [f\"English translation of: {item['transcription']}\" for item in metadata]\n",
    "    \n",
    "    # Create DataFrame\n",
    "    df = pd.DataFrame({\"kapampangan\": kapampangan_texts, \"english\": english_texts})\n",
    "    df.to_csv(output_path, index=False)\n",
    "    \n",
    "    return len(df)\n",
    "\n",
    "marianmt_csv = \"data/kapampangan_english.csv\"\n",
    "num_samples = create_marianmt_dataset(METADATA_FILE, processed_dataset, marianmt_csv)\n",
    "print(f\"Created MarianMT training CSV with {num_samples} Kapampangan-English pairs at {marianmt_csv}\")\n",
    "print(\"\\nYou can now use this CSV file to train your MarianMT model following your previous script.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
