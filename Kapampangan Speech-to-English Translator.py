# Kapampangan Speech-to-English Translation Inference Pipeline

## Requirements
# - Trained Wav2Vec2 ASR model
# - Trained MarianMT translation model
# - Hugging Face Transformers, Torchaudio, PyTorch

import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import MarianTokenizer, MarianMTModel

# --- Configuration ---
ASR_MODEL_DIR = "./kapampangan_wav2vec2_model"         # Path to fine-tuned Wav2Vec2 model
MT_MODEL_DIR = "./kapampangan_mt_model"               # Path to fine-tuned MarianMT model

# --- Load ASR Components ---
asr_processor = Wav2Vec2Processor.from_pretrained(ASR_MODEL_DIR)
asr_model = Wav2Vec2ForCTC.from_pretrained(ASR_MODEL_DIR).to("cuda" if torch.cuda.is_available() else "cpu")
asr_model.eval()

# --- Load MarianMT Components ---
mt_tokenizer = MarianTokenizer.from_pretrained(MT_MODEL_DIR)
mt_model = MarianMTModel.from_pretrained(MT_MODEL_DIR).to("cuda" if torch.cuda.is_available() else "cpu")
mt_model.eval()

# --- Function: Audio -> Kapampangan Transcription ---
def asr_transcribe(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
    waveform = waveform.squeeze()
    inputs = asr_processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = asr_model(**inputs.to(asr_model.device)).logits
    pred_ids = torch.argmax(logits, dim=-1)
    transcription = asr_processor.batch_decode(pred_ids)[0]
    print(f"🔍 Raw prediction IDs: {pred_ids}")
    print(f"📝 Raw transcription output: {transcription}")
    return transcription

# --- Function: Kapampangan Text -> English Translation ---
def kapampangan_translate(text):
    inputs = mt_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(mt_model.device)
    with torch.no_grad():
        outputs = mt_model.generate(**inputs)
    return mt_tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Combined Inference Function ---
def speech_to_translation(audio_path):
    print(f"🔊 Processing audio: {audio_path}")
    kapampangan_text = asr_transcribe(audio_path)
    print(f"📝 Transcription: {kapampangan_text}")
    english_translation = kapampangan_translate(kapampangan_text)
    print(f"🌐 Translation: {english_translation}")
    return english_translation


print(asr_processor.tokenizer.get_vocab().keys())
# --- Example Run ---
# Replace with your actual .wav file path
example_audio = "data/validated_audio/cat03_entry001_spk013.wav"
speech_to_translation(example_audio)

manual_kapampangan = "Nanu ya ing kayang pamangan?"
print("🌐 Translation Test:", kapampangan_translate(manual_kapampangan))
