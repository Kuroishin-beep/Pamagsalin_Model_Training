from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM

# Point to your quantized dynamic model folder
model_path = "quantized_model_dynamic"

# Load tokenizer (same across all versions)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load quantized ONNX model
model = ORTModelForSeq2SeqLM.from_pretrained(model_path, provider="CPUExecutionProvider")

# Input text (Kapampangan → English)
text = "Pota na mu"

# Tokenize
inputs = tokenizer(text, return_tensors="pt")

# Run generation
outputs = model.generate(**inputs)

# Decode to English
translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Translation:", translation)
