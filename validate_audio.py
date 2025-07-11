import os
import librosa
import soundfile as sf
import pandas as pd

def validate_and_process_audio(
    source_folder,
    output_folder,
    metadata_csv,
    target_sr=16000
):
    """
    Validates audio files to ensure they are mono and have the target sample rate.
    Processes and saves valid files to a new directory.

    Args:
        source_folder (str): Path to the folder containing the original audio files.
        output_folder (str): Path to the folder where validated audio files will be saved.
        metadata_csv (str): Path to the CSV file with transcription data.
        target_sr (int): The target sample rate for the audio files.
    """
    print(f"Starting audio validation and processing...")
    print(f"Source folder: {'D:/Github/Project/Pamagsalin_Model_Training/data/Audio Files'}")
    print(f"Output folder: {'D:/Github/Project/Pamagsalin_Model_Training/data/Valdiate_Audio_Files'}")
    print(f"Metadata CSV: {'D:/Github/Project/Pamagsalin_Model_Training/data/kapampangan_english.csv'}")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    try:
        metadata = pd.read_csv(metadata_csv)
        print("Successfully loaded metadata CSV.")
    except FileNotFoundError:
        print(f"Error: Metadata CSV file not found at {metadata_csv}")
        return
    except Exception as e:
        print(f"Error reading metadata CSV: {e}")
        return

    processed_files = 0
    skipped_files = 0
    new_metadata = []

    audio_files = [f for f in os.listdir(source_folder) if f.endswith('.wav')]
    print(f"Found {len(audio_files)} .wav files in the source folder.")

    for filename in audio_files:
        source_path = os.path.join(source_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Check if the file has a corresponding transcription
        # Extract entry number from filename (e.g., "cat03_entry001_spk013.wav" -> "entry001")
        import re
        entry_match = re.search(r'entry(\d+)', filename)
        if not entry_match:
            print(f"- WARNING: Could not extract entry number from {filename}. Skipping.")
            skipped_files += 1
            continue
        
        entry_num = int(entry_match.group(1))
        # The CSV has entries 1-18, so we need to find the corresponding row
        if entry_num > len(metadata):
            print(f"- WARNING: Entry {entry_num} exceeds available transcriptions. Skipping.")
            skipped_files += 1
            continue
        
        # Get the transcription from the CSV (entry_num - 1 because CSV is 0-indexed)
        transcription = metadata.iloc[entry_num - 1]['english']

        try:
            # Load audio file
            audio, sr = librosa.load(source_path, sr=None, mono=False)

            # 1. Check and convert to mono if necessary
            if audio.ndim > 1:
                print(f"- INFO: {filename} is stereo. Converting to mono.")
                audio = librosa.to_mono(audio)

            # 2. Check and resample if necessary
            if sr != target_sr:
                print(f"- INFO: {filename} has sample rate {sr}Hz. Resampling to {target_sr}Hz.")
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

            # Save the validated and processed audio file
            sf.write(output_path, audio, target_sr)

            # Add to our new metadata list for the final dataset
            new_metadata.append({
                "file_path": os.path.abspath(output_path),
                "transcription": transcription
            })
            processed_files += 1

        except Exception as e:
            print(f"- ERROR: Could not process {filename}. Reason: {e}")
            skipped_files += 1

    print("\n--- Validation Summary ---")
    print(f"Successfully processed and saved: {processed_files} files.")
    print(f"Skipped files: {skipped_files} files.")
    print(f"Validated audio files are located in: {output_folder}")

    # Save the new metadata file that points to the validated audio
    if new_metadata:
        final_metadata_df = pd.DataFrame(new_metadata)
        final_metadata_path = os.path.join(output_folder, "metadata.csv")
        final_metadata_df.to_csv(final_metadata_path, index=False)
        print(f"A new metadata file 'metadata.csv' has been created in the output folder.")
        print("This file contains the absolute paths to the validated audio and their transcriptions.")

if __name__ == '__main__':
    # --- Configuration ---
    # IMPORTANT: Update these paths to match your folder structure
    SOURCE_AUDIO_FOLDER = 'data/Audio Files'
    VALIDATED_AUDIO_FOLDER = 'data/validated_audio'
    METADATA_CSV_FILE = 'data/kapampangan_english.csv' # This should contain 'file_name' and 'transcription' columns

    # --- Run the script ---
    validate_and_process_audio(
        source_folder=SOURCE_AUDIO_FOLDER,
        output_folder=VALIDATED_AUDIO_FOLDER,
        metadata_csv=METADATA_CSV_FILE
    )
