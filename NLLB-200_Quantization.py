from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import os

MODEL_DIR = "./kapampangan_mt_nllb/checkpoint-3480"
ONNX_DIR = "./onnx_model"

os.makedirs(ONNX_DIR, exist_ok=True)

# Export and save ONNX model
model = ORTModelForSeq2SeqLM.from_pretrained(
    MODEL_DIR,
    export=True,            # forces ONNX export
    use_cache=False         # safer for export
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model.save_pretrained(ONNX_DIR)
tokenizer.save_pretrained(ONNX_DIR)

print("✅ Export complete. ONNX model saved at:", ONNX_DIR)
