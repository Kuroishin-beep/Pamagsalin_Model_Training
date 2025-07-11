import torch
import librosa
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    pipeline
)

# --- Configuration ---
# Path to your fine-tuned Wav2Vec2 model directory
ASR_MODEL_PATH = './kapampangan_wav2vec2_model'

# Name of the translation model from Hugging Face Hub.
# NOTE: There isn't a dedicated public Kapampangan-English model.
# We will use a Tagalog-English model as the closest available relative.
# For better results, you would need to train your own translation model.
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-tl-en"

# Path to an audio file you want to translate
# Make sure this is a .wav file from your dataset or a new recording.
AUDIO_FILE_TO_TRANSLATE = 'path/to/your/test_audio.wav' 

# --- 1. Load the Models and Processors ---

def load_models():
    """
    Loads the fine-tuned ASR model and the translation pipeline.
    """
    print("--- Loading models, this might take a moment... ---")
    try:
        # Load your custom-trained Automatic Speech Recognition (ASR) model
        asr_model = Wav2Vec2ForCTC.from_pretrained(ASR_MODEL_PATH)
        asr_processor = Wav2Vec2Processor.from_pretrained(ASR_MODEL_PATH)
        
        # Load the translation model pipeline
        # This will download the model from the Hub on the first run.
        translator = pipeline("translation", model=TRANSLATION_MODEL)

        print("--- Models loaded successfully! ---")
        return asr_model, asr_processor, translator

    except OSError:
        print(f"ERROR: Could not find the ASR model at '{ASR_MODEL_PATH}'.")
        print("Please ensure the path is correct and you have run the training script.")
        return None, None, None
    except Exception as e:
        print(f"An error occurred while loading models: {e}")
        return None, None, None


# --- 2. Define the Translation Function ---

def translate_kapampangan_audio_to_english(audio_path, asr_model, asr_processor, translator):
    """
    Takes the path to a WAV audio file and returns the English translation.
    
    Args:
        audio_path (str): The file path to the 16kHz mono WAV file.
        asr_model: The loaded Wav2Vec2 CTC model.
        asr_processor: The loaded Wav2Vec2 processor.
        translator: The loaded translation pipeline.
    """
    if not audio_path or not os.path.exists(audio_path):
        print(f"ERROR: Audio file not found at '{audio_path}'")
        return

    print(f"\n--- Processing: {os.path.basename(audio_path)} ---")

    # 1. Load and preprocess the audio file
    try:
        speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
    except Exception as e:
        print(f"ERROR: Could not read audio file. Make sure it's a valid .wav file. Details: {e}")
        return

    # 2. Transcribe Kapampangan speech to text
    print("Step 1: Transcribing Kapampangan audio...")
    input_values = asr_processor(speech_array, return_tensors="pt", sampling_rate=sampling_rate).input_values
    
    # Get model logits and find the most likely token ids
    with torch.no_grad():
        logits = asr_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    
    # Decode the token ids to a string
    kapampangan_transcription = asr_processor.batch_decode(predicted_ids)[0]
    print(f"  > Kapampangan Transcription: '{kapampangan_transcription}'")

    # 3. Translate Kapampangan text to English
    print("\nStep 2: Translating text to English...")
    if not kapampangan_transcription.strip():
        print("  > Warning: Transcription is empty. Cannot translate.")
        english_translation_text = "[No text to translate]"
    else:
        # The translator expects a list of sentences
        translation_output = translator(kapampangan_transcription)
        english_translation_text = translation_output[0]['translation_text']

    print(f"  > English Translation: '{english_translation_text}'")
    print("--- Translation Complete ---")

    return kapampangan_transcription, english_translation_text


# --- Main Execution ---
if __name__ == '__main__':
    import os

    # Load the models once
    asr_model, asr_processor, translator = load_models()

    if all([asr_model, asr_processor, translator]):
        # Check if the example audio file path is valid
        if AUDIO_FILE_TO_TRANSLATE == 'path/to/your/test_audio.wav' or not os.path.exists(AUDIO_FILE_TO_TRANSLATE):
            print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!! PLEASE UPDATE 'AUDIO_FILE_TO_TRANSLATE' IN THE SCRIPT !!!")
            print("!!! Point it to a real .wav file to test the translator.    !!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        else:
            # Run the full translation process on the specified audio file
            translate_kapampangan_audio_to_english(
                AUDIO_FILE_TO_TRANSLATE,
                asr_model,
                asr_processor,
                translator
            )

