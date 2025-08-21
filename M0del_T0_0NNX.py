from optimum.exporters.onnx import main_export
from transformers import AutoTokenizer
import os

model_id = "facebook/nllb-200-distilled-600M"
save_dir = "onnx_model"

# Make sure folder exists
os.makedirs(save_dir, exist_ok=True)

# Save tokenizer so ONNX + ORT can reload later
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(save_dir)

# Export model to ONNX
main_export(
    model_name_or_path=model_id,
    output=save_dir,
    task="seq2seq-lm",
    framework="pt",        # PyTorch backend
    opset=14               # good balance of support
)

print("✅ Export finished. ONNX models saved in:", save_dir)
