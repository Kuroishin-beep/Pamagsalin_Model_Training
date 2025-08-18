import os
import subprocess
import csv
import json

# Config
INPUT_DIR = "data/Audio_Files"
OUTPUT_DIR = "data/Cleaned_Audio_Files"
INPUT_METADATA = "data/kapampangan_audio_tads.csv"  # Original CSV file
OUTPUT_METADATA_CSV = os.path.join(OUTPUT_DIR, "metadata.csv")
OUTPUT_METADATA_JSON = os.path.join(OUTPUT_DIR, "metadata.json")

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1  # mono

def process_audio(input_path, output_path):
    """Converts audio to WAV, mono, 16kHz without changing filename."""
    try:
        command = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ac", str(TARGET_CHANNELS),
            "-ar", str(TARGET_SAMPLE_RATE),
            output_path
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ Processed: {input_path} → {output_path}")
    except Exception as e:
        print(f"❌ Failed to process {input_path}: {e}")

def scan_and_clean(metadata_map):
    """Scans INPUT_DIR, processes audio, and collects metadata for cleaned files."""
    cleaned_metadata = []

    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if not file.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac")):
                continue

            input_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, INPUT_DIR)
            output_folder = os.path.join(OUTPUT_DIR, relative_path)
            os.makedirs(output_folder, exist_ok=True)

            output_file_name = os.path.splitext(file)[0] + ".wav"
            output_path = os.path.join(output_folder, output_file_name)

            process_audio(input_path, output_path)

            # Find matching transcription from original metadata
            transcription = metadata_map.get(file, "")
            cleaned_metadata.append({
                "file_path": output_path.replace("\\", "/"),  # Normalize for JSON
                "transcription": transcription
            })

    return cleaned_metadata

def load_metadata(csv_file):
    """Loads original metadata into a dict mapping filename -> transcription."""
    mapping = {}
    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = os.path.basename(row["file_path"])
            mapping[filename] = row.get("transcription", "")
    return mapping

def save_metadata(cleaned_metadata):
    """Saves cleaned metadata to CSV and JSON."""
    # Save CSV
    with open(OUTPUT_METADATA_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_path", "transcription"])
        writer.writeheader()
        writer.writerows(cleaned_metadata)

    # Save JSON
    with open(OUTPUT_METADATA_JSON, mode="w", encoding="utf-8") as f:
        json.dump(cleaned_metadata, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Metadata saved to:\n  CSV: {OUTPUT_METADATA_CSV}\n  JSON: {OUTPUT_METADATA_JSON}")

if __name__ == "__main__":
    if not os.path.exists(INPUT_METADATA):
        print(f"❌ ERROR: Metadata CSV '{INPUT_METADATA}' not found.")
        exit(1)

    metadata_map = load_metadata(INPUT_METADATA)
    cleaned_metadata = scan_and_clean(metadata_map)
    save_metadata(cleaned_metadata)
