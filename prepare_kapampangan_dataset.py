import os
import pandas as pd
import shutil
import re
from pathlib import Path

def create_kapampangan_dataset_correct():
    """
    Creates a Kapampangan dataset by:
    1. Reading the Kapampangan-English translations
    2. Creating a new metadata file with correct Kapampangan transcriptions
    3. Copying audio files to a new directory
    """
    
    # Read the Kapampangan-English translations
    kapampangan_data = pd.read_csv('data/kapampangan_english.csv')
    
    # Create output directory
    output_dir = 'data/kapampangan_audio'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a new metadata file with Kapampangan transcriptions
    metadata_entries = []
    
    # Get all audio files from Audio Files directory (72 files total)
    audio_dir = 'data/validated_audio'
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    audio_files.sort()  # Sort to ensure consistent ordering
    
    print(f"Found {len(audio_files)} audio files")
    print(f"Found {len(kapampangan_data)} Kapampangan translations")
    
    # Create a mapping from entry number to Kapampangan translation
    entry_to_kapampangan = {}
    for i, (_, row) in enumerate(kapampangan_data.iterrows()):
        entry_num = i + 1
        kapampangan_text = row['kapampangan']
        # Clean the Kapampangan text (remove quotes and extra spaces)
        kapampangan_text = kapampangan_text.strip().strip('"').strip("'")
        entry_to_kapampangan[entry_num] = kapampangan_text
    
    # Process each audio file
    for audio_file in audio_files:
        # Extract entry number from filename (e.g., "cat03_entry001_spk013.wav" -> 1)
        match = re.search(r'entry(\d+)', audio_file)
        if match:
            entry_num = int(match.group(1))
            
            # Get the corresponding Kapampangan translation
            if entry_num in entry_to_kapampangan:
                kapampangan_text = entry_to_kapampangan[entry_num]
                
                # Copy audio file to new directory
                src_path = os.path.join(audio_dir, audio_file)
                dst_path = os.path.join(output_dir, audio_file)
                shutil.copy2(src_path, dst_path)
                
                # Add to metadata
                metadata_entries.append({
                    'file_path': dst_path,
                    'transcription': kapampangan_text
                })
                
                print(f"Processed: {audio_file} (entry{entry_num:03d}) -> {kapampangan_text[:50]}...")
            else:
                print(f"Warning: No translation found for entry {entry_num}")
        else:
            print(f"Warning: Could not extract entry number from {audio_file}")
    
    # Create metadata DataFrame and save
    metadata_df = pd.DataFrame(metadata_entries)
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    metadata_df.to_csv(metadata_path, index=False)
    
    print(f"\nDataset created successfully!")
    print(f"Audio files: {output_dir}")
    print(f"Metadata: {metadata_path}")
    print(f"Total samples: {len(metadata_df)}")
    
    # Show some examples of the mapping
    print(f"\nExample mappings:")
    for i in range(min(5, len(metadata_df))):
        entry = metadata_df.iloc[i]
        filename = os.path.basename(entry['file_path'])
        transcription = entry['transcription'][:60] + "..." if len(entry['transcription']) > 60 else entry['transcription']
        print(f"  {filename} -> {transcription}")
    
    return output_dir

if __name__ == '__main__':
    create_kapampangan_dataset_correct() 