import os
import pandas as pd
import shutil
import re

def create_kapampangan_dataset_correct():
    """
    Creates a Kapampangan dataset by:
    1. Reading the Kapampangan-English translations
    2. Creating a new metadata file with correct Kapampangan transcriptions
    3. Copying audio files to a new directory
    4. Returning a list of (audio_path, kapampangan_transcription)
    """
    
    # Read the Kapampangan-English translations
    kapampangan_data = pd.read_csv('data/kapampangan_english.csv')
    
    # Create output directory
    output_dir = 'data/kapampangan_audio1'
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all audio files from validated_audio directory
    audio_dir = 'data/validated_audio'
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    audio_files.sort()
    
    print(f"Found {len(audio_files)} audio files")
    print(f"Found {len(kapampangan_data)} Kapampangan translations")
    
    # Map entry number to Kapampangan transcription
    entry_to_kapampangan = {}
    for i, row in kapampangan_data.iterrows():
        entry_num = i + 1
        kapampangan_text = str(row['kapampangan']).strip().strip('"').strip("'")
        entry_to_kapampangan[entry_num] = kapampangan_text

    metadata_entries = []
    paired_data = []

    for audio_file in audio_files:
        match = re.search(r'entry(\d+)', audio_file)
        if match:
            entry_num = int(match.group(1))
            if entry_num in entry_to_kapampangan:
                kapampangan_text = entry_to_kapampangan[entry_num]
                
                src_path = os.path.join(audio_dir, audio_file)
                dst_path = os.path.join(output_dir, audio_file)
                shutil.copy2(src_path, dst_path)
                
                metadata_entries.append({
                    'file_path': dst_path,
                    'transcription': kapampangan_text
                })
                paired_data.append((dst_path, kapampangan_text))

                print(f"Processed: {audio_file} (entry{entry_num:03d}) -> {kapampangan_text[:50]}...")
            else:
                print(f"Warning: No translation found for entry {entry_num}")
        else:
            print(f"Warning: Could not extract entry number from {audio_file}")

    # Save metadata to CSV
    metadata_df = pd.DataFrame(metadata_entries)
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    metadata_df.to_csv(metadata_path, index=False)
    
    print(f"\nDataset created successfully!")
    print(f"Audio files: {output_dir}")
    print(f"Metadata CSV: {metadata_path}")
    print(f"Total paired samples: {len(paired_data)}")

    return paired_data

if __name__ == '__main__':
    audio_transcription_pairs = create_kapampangan_dataset_correct()

    # Example usage:
    print("\nExample returned values:")
    for path, text in audio_transcription_pairs[:5]:
        print(f"{os.path.basename(path)} => {text[:60]}")
