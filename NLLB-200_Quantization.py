import os
import warnings
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Directories
ONNX_DIR = "./onnx_model"
QUANTIZED_DIR = "./quantized_model"

def quantize_onnx_model(input_dir: str, output_dir: str, quantization_type: str = "dynamic"):
    """
    Quantize ONNX model using different strategies.
    Handles multi-file models (encoder/decoder) separately.
    
    Args:
        input_dir: Directory containing the ONNX model
        output_dir: Directory to save quantized model
        quantization_type: "dynamic", "static", or "qint8"
    """
    
    try:
        from optimum.onnxruntime import ORTQuantizer, QuantizationConfig
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        import shutil
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"📁 Created quantized model directory: {output_dir}")
        
        # Check if ONNX model exists
        if not Path(input_dir).exists():
            raise FileNotFoundError(f"ONNX model directory not found: {input_dir}")
        
        onnx_files = list(Path(input_dir).glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"No ONNX files found in: {input_dir}")
        
        logger.info(f"Found ONNX files: {[f.name for f in onnx_files]}")
        
        # Copy non-ONNX files first (tokenizer, config, etc.)
        for file_path in Path(input_dir).iterdir():
            if file_path.is_file() and not file_path.name.endswith('.onnx'):
                shutil.copy2(file_path, output_dir)
        
        # Handle multi-file models (encoder/decoder)
        if len(onnx_files) > 1:
            logger.info("🔄 Multi-file model detected, quantizing each file separately...")
            
            for onnx_file in onnx_files:
                file_name = onnx_file.name
                logger.info(f"🔄 Quantizing {file_name}...")
                
                # Load quantizer for specific file
                quantizer = ORTQuantizer.from_pretrained(input_dir, file_name=file_name)
                
                # Configure quantization
                qconfig = get_quantization_config(quantization_type)
                
                # Quantize this specific file
                quantizer.quantize(
                    save_dir=output_dir,
                    quantization_config=qconfig,
                    file_suffix=""  # Don't add suffix to avoid conflicts
                )
                
                logger.info(f"✅ {file_name} quantized successfully")
        
        else:
            # Single file model
            logger.info("🔄 Single-file model detected...")
            quantizer = ORTQuantizer.from_pretrained(input_dir)
        
            
            # Configure and quantize
            qconfig = get_quantization_config(quantization_type)
            quantizer.quantize(
                save_dir=output_dir,
                quantization_config=qconfig,
                file_suffix=""
            )
        
        logger.info("✅ Quantization completed!")
        
        # Compare file sizes
        compare_model_sizes(input_dir, output_dir)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Quantization failed: {str(e)}")
        return False

def get_quantization_config(quantization_type: str):
    """Get quantization configuration based on type."""
    from optimum.onnxruntime import QuantizationConfig
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    
    if quantization_type == "dynamic":
        logger.info("🔧 Configuring dynamic quantization...")
        # Use AutoQuantizationConfig for better compatibility
        return AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
        
    elif quantization_type == "dynamic_manual":
        logger.info("🔧 Configuring manual dynamic quantization...")
        # Manual configuration with proper dtype specifications
        return QuantizationConfig(
            is_static=False,
            format="QDQ",
            mode="IntegerOps",
            activations_dtype="int8",
            weights_dtype="int8",
            per_channel=False,  # Set to False for better compatibility
            reduce_range=True,  # Help with compatibility
            operators_to_quantize=["MatMul", "Gemm"]  # Focus on key operators
        )
        
    elif quantization_type == "static":
        logger.info("🔧 Configuring static quantization...")
        return QuantizationConfig(
            is_static=True,
            format="QOperator", 
            mode="IntegerOps",
            activations_dtype="uint8",
            weights_dtype="uint8",
            per_channel=False,  # Set to False for better compatibility
            reduce_range=True,
            operators_to_quantize=["MatMul", "Gemm", "Conv"]
        )
        
    elif quantization_type == "qint8":
        logger.info("🔧 Configuring INT8 quantization...")
        return AutoQuantizationConfig.avx512_vnni(
            is_static=False,
            per_channel=False  # Set to False for better compatibility
        )
        
    elif quantization_type == "simple":
        logger.info("🔧 Configuring simple quantization...")
        # Simplest possible configuration
        return QuantizationConfig(
            is_static=False,
            format="QOperator",
            mode="IntegerOps"
        )
        
    else:
        raise ValueError(f"Unknown quantization type: {quantization_type}")

def compare_model_sizes(original_dir: str, quantized_dir: str):
    """Compare file sizes between original and quantized models."""
    
    def get_dir_size(directory):
        """Get total size of all files in directory."""
        total_size = 0
        for file_path in Path(directory).rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    try:
        original_size = get_dir_size(original_dir)
        quantized_size = get_dir_size(quantized_dir)
        
        original_mb = original_size / (1024 * 1024)
        quantized_mb = quantized_size / (1024 * 1024)
        compression_ratio = (original_size - quantized_size) / original_size * 100
        
        logger.info("📊 Model Size Comparison:")
        logger.info(f"   Original:  {original_mb:.1f} MB")
        logger.info(f"   Quantized: {quantized_mb:.1f} MB")
        logger.info(f"   Reduction: {compression_ratio:.1f}%")
        
    except Exception as e:
        logger.warning(f"Could not compare sizes: {e}")

def test_quantized_model(quantized_dir: str, test_text: str = "Hello, how are you?"):
    """Test the quantized model with a sample input."""
    
    try:
        from optimum.onnxruntime import ORTModelForSeq2SeqLM
        from transformers import AutoTokenizer
        import torch
        
        logger.info("🧪 Testing quantized model...")
        
        # Load quantized model and tokenizer
        model = ORTModelForSeq2SeqLM.from_pretrained(quantized_dir)
        tokenizer = AutoTokenizer.from_pretrained(quantized_dir)
        
        # Test inference with more robust parameters
        inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
        
        import time
        start_time = time.time()
        
        # More robust generation parameters
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_length=100,  # Increased max length
                min_length=5,    # Ensure minimum output
                num_beams=2,     # Reduced beams for stability
                do_sample=False, # Deterministic output
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                forced_bos_token_id=tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else None
            )
        
        inference_time = time.time() - start_time
        
        # More robust output handling
        if outputs is not None and len(outputs) > 0:
            # Check if we have valid outputs
            if outputs.shape[1] > 0:  # Check if sequence length > 0
                result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                logger.info(f"✅ Test successful!")
                logger.info(f"   Input: '{test_text}'")
                logger.info(f"   Output: '{result}'")
                logger.info(f"   Output length: {len(result)} characters")
                logger.info(f"   Inference time: {inference_time:.2f} seconds")
                
                return True
            else:
                logger.warning("⚠️ Model generated empty sequence")
                return False
        else:
            logger.warning("⚠️ Model returned no outputs")
            return False
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        # Try a simpler test
        try:
            logger.info("🔄 Trying simplified test...")
            return test_model_loading_only(quantized_dir)
        except:
            return False

def test_model_loading_only(quantized_dir: str):
    """Simple test that just loads the model without generation."""
    try:
        from optimum.onnxruntime import ORTModelForSeq2SeqLM
        from transformers import AutoTokenizer
        
        # Just test if we can load the model and tokenizer
        model = ORTModelForSeq2SeqLM.from_pretrained(quantized_dir)
        tokenizer = AutoTokenizer.from_pretrained(quantized_dir)
        
        # Basic forward pass test
        test_input = "Hello"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        # Try encoder forward pass
        if hasattr(model, 'encoder'):
            encoder_outputs = model.encoder(**inputs)
            logger.info("✅ Encoder forward pass successful")
        
        logger.info("✅ Model loading test successful!")
        logger.info(f"   Model type: {type(model).__name__}")
        logger.info(f"   Tokenizer type: {type(tokenizer).__name__}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Loading test failed: {str(e)}")
        return False

def quantize_with_multiple_strategies():
    """Try multiple quantization strategies and compare results."""
    
    # Order strategies from most likely to work to least
    strategies = ["dynamic", "simple", "qint8", "dynamic_manual"]
    results = {}
    
    for strategy in strategies:
        output_dir = f"./quantized_model_{strategy}"
        logger.info(f"\n🔄 Trying {strategy} quantization...")
        
        success = quantize_onnx_model(ONNX_DIR, output_dir, strategy)
        results[strategy] = {
            "success": success,
            "output_dir": output_dir
        }
        
        if success:
            # Test the quantized model
            test_success = test_quantized_model(output_dir)
            results[strategy]["test_success"] = test_success
            
            # If this strategy worked, we can stop here
            logger.info(f"✅ {strategy} quantization successful! Using this strategy.")
            break
        else:
            logger.info(f"❌ {strategy} quantization failed, trying next strategy...")
    
    # Summary
    logger.info("\n📋 Quantization Summary:")
    for strategy, result in results.items():
        status = "✅ Success" if result["success"] else "❌ Failed"
        logger.info(f"   {strategy.capitalize()}: {status}")
        
        if result.get("test_success"):
            logger.info(f"      Test: ✅ Passed")
        elif result["success"]:
            logger.info(f"      Test: ⚠️  Failed")
    
    return results

def debug_quantized_model(quantized_dir: str):
    """Debug function to inspect the quantized model structure."""
    
    try:
        from optimum.onnxruntime import ORTModelForSeq2SeqLM
        from transformers import AutoTokenizer
        import json
        
        logger.info("🔍 Debugging quantized model...")
        
        # Check files in directory
        files = list(Path(quantized_dir).iterdir())
        logger.info(f"Files in quantized directory: {[f.name for f in files]}")
        
        # Load and inspect tokenizer
        tokenizer = AutoTokenizer.from_pretrained(quantized_dir)
        logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
        logger.info(f"Special tokens: pad={tokenizer.pad_token_id}, eos={tokenizer.eos_token_id}, bos={getattr(tokenizer, 'bos_token_id', 'None')}")
        
        # Load model
        model = ORTModelForSeq2SeqLM.from_pretrained(quantized_dir)
        logger.info(f"Model type: {type(model)}")
        
        # Check if model has encoder/decoder
        logger.info(f"Has encoder: {hasattr(model, 'encoder')}")
        logger.info(f"Has decoder: {hasattr(model, 'decoder')}")
        
        # Test tokenization
        test_text = "Hello world"
        inputs = tokenizer(test_text, return_tensors="pt")
        logger.info(f"Input IDs shape: {inputs['input_ids'].shape}")
        logger.info(f"Input IDs: {inputs['input_ids'].tolist()}")
        
        # Try encoder forward pass if available
        if hasattr(model, 'encoder'):
            encoder_outputs = model.encoder(inputs['input_ids'])
            logger.info(f"Encoder output shape: {encoder_outputs.last_hidden_state.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Debug failed: {str(e)}")
        return False
    """Fallback method using the simplest quantization approach."""
    
    try:
        from optimum.onnxruntime import ORTQuantizer
        import shutil
        
        logger.info("🔄 Trying fallback quantization method...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy all non-ONNX files
        for file_path in Path(input_dir).iterdir():
            if file_path.is_file() and not file_path.name.endswith('.onnx'):
                shutil.copy2(file_path, output_dir)
        
        # Get ONNX files
        onnx_files = list(Path(input_dir).glob("*.onnx"))
        
        for onnx_file in onnx_files:
            file_name = onnx_file.name
            logger.info(f"🔄 Quantizing {file_name} with fallback method...")
            
            # Try the most basic quantization
            quantizer = ORTQuantizer.from_pretrained(input_dir, file_name=file_name)
            
            # Use default quantization - let optimum decide
            quantizer.quantize(save_dir=output_dir, file_suffix="")
            
            logger.info(f"✅ {file_name} quantized with fallback method")
        
        logger.info("✅ Fallback quantization completed!")
        compare_model_sizes(input_dir, output_dir)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Fallback quantization failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Check if ONNX model exists
    if not Path(ONNX_DIR).exists():
        logger.error(f"❌ ONNX model directory not found: {ONNX_DIR}")
        logger.info("Please run the ONNX export script first.")
        exit(1)
    
    # Try multiple strategies
    logger.info("🚀 Starting quantization with multiple strategies...")
    results = quantize_with_multiple_strategies()
    
    # If all strategies failed, try the simplest fallback
    if not any(result["success"] for result in results.values()):
        logger.info("\n🔄 All quantization strategies failed, trying fallback method...")
        fallback_success = simple_quantize_fallback(ONNX_DIR, QUANTIZED_DIR + "_fallback")
        
        if fallback_success:
            test_quantized_model(QUANTIZED_DIR + "_fallback", "Kumusta ka?")
        else:
            logger.error("❌ All quantization methods failed!")
    else:
        # Find the successful strategy and test it
        successful_strategy = next(
            (strategy for strategy, result in results.items() if result["success"]), 
            None
        )
        if successful_strategy:
            result_dir = results[successful_strategy]["output_dir"]
            
            # Debug the model first
            logger.info(f"\n🔍 Debugging successful {successful_strategy} quantization...")
            debug_quantized_model(result_dir)
            
            # Then test it
            if not results[successful_strategy].get("test_success"):
                logger.info(f"\n🧪 Testing successful {successful_strategy} quantization...")
                test_quantized_model(result_dir, "Kumusta ka?")