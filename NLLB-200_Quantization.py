from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

MODEL_DIR = "./kapampangan_mt_nllb"

# Load trained model in 8-bit precision
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_DIR,
    load_in_8bit=True,                # or load_in_4bit=True
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# Test translation after quantization
def batch_translate_quant(texts, batch_size=8):
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(0, len(texts), batch_size):
        src_texts = [f"<kap> {t}" for t in texts[i:i+batch_size]]
        inputs = tokenizer(src_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs)
        results.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return results
