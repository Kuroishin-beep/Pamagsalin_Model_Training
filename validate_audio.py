# validate_resample_audio.py

import os
import librosa
import soundfile as sf

def validate_and_resample_audio(source_dir="D:/Github/Project/Pamagsalin_Model_Training/data/Audio Files", target_dir="D:/Github/Project/Pamagsalin_Model_Training/data_resampled", target_sr=16000):
    os.makedirs(target_dir, exist_ok=True)

    for root, _, files in os.walk(source_dir):
        for filename in files:
            if filename.endswith(".wav"):
                source_path = os.path.join(root, filename)

                # Preserve subfolder structure in the output directory
                relative_path = os.path.relpath(source_path, source_dir)
                target_path = os.path.join(target_dir, relative_path)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)

                print(f"Processing {relative_path}...")

                try:
                    audio, sr = librosa.load(source_path, sr=None, mono=True)

                    # Resample if needed
                    if sr != target_sr:
                        print(f" - Resampling from {sr} Hz to {target_sr} Hz")
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

                    # Save resampled audio
                    sf.write(target_path, audio, target_sr)

                    print(f" - Saved to {target_path}")

                except Exception as e:
                    print(f" - Error processing {relative_path}: {e}")

    print("\n✅ All audio files have been validated and resampled.")

if __name__ == "__main__":
    validate_and_resample_audio()
