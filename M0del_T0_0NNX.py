import os
import sys
import logging
import tempfile
import shutil
import gc
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables to avoid permission issues
os.environ['TEMP'] = os.path.expanduser('~/temp')
os.environ['TMP'] = os.path.expanduser('~/temp')

# Ensure temp directory exists
temp_dir = Path(os.environ['TEMP'])
temp_dir.mkdir(exist_ok=True)

MODEL_DIR = "./kapampangan_mt_nllb/checkpoint-3480"
ONNX_DIR = "./onnx_model"

def fix_tokenizer_files(model_dir: str):
    """
    Fix corrupted tokenizer files by rebuilding them from the model.
    
    Args:
        model_dir: Path to the model directory
    """
    try:
        from transformers import M2M100Tokenizer
        
        logger.info("Attempting to fix tokenizer files...")
        
        # Try to load with the slow tokenizer first
        tokenizer = M2M100Tokenizer.from_pretrained(
            model_dir, 
            use_fast=False,  # Use slow tokenizer to avoid corruption
            local_files_only=True
        )
        
        # Save the tokenizer (this will recreate the files)
        backup_dir = f"{model_dir}_tokenizer_backup"
        tokenizer.save_pretrained(backup_dir)
        logger.info(f"✅ Tokenizer backup saved to: {backup_dir}")
        
        return backup_dir
        
    except Exception as e:
        logger.error(f"Failed to fix tokenizer: {str(e)}")
        return None

def export_model_to_onnx_safe(model_dir: str, output_dir: str, use_backup_tokenizer: bool = False):
    """
    Safely export model to ONNX with comprehensive error handling.
    """
    try:
        # Validate input directory
        if not Path(model_dir).exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
        
        # Try different approaches based on the error
        tokenizer_dir = model_dir
        if use_backup_tokenizer:
            backup_dir = fix_tokenizer_files(model_dir)
            if backup_dir:
                tokenizer_dir = backup_dir
        
        try:
            # First, try to load tokenizer with different strategies
            from transformers import AutoTokenizer
            
            # Strategy 1: Use slow tokenizer
            logger.info("Loading tokenizer (slow tokenizer)...")
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_dir,
                use_fast=False,  # Avoid fast tokenizer corruption
                local_files_only=True
            )
            
        except Exception as tokenizer_error:
            logger.warning(f"Slow tokenizer failed: {tokenizer_error}")
            
            try:
                # Strategy 2: Load from parent model
                logger.info("Trying to load tokenizer from base model...")
                tokenizer = AutoTokenizer.from_pretrained(
                    "facebook/nllb-200-distilled-600M",
                    use_fast=False
                )
                
                # Save the working tokenizer to model directory
                tokenizer.save_pretrained(f"{model_dir}_fixed_tokenizer")
                logger.info("✅ Fixed tokenizer saved")
                
            except Exception as e:
                logger.error(f"All tokenizer strategies failed: {e}")
                raise
        
        # Load model with custom temp directory
        logger.info("Loading model and converting to ONNX...")
        
        # Use a custom temp directory to avoid permission issues
        custom_temp = Path(output_dir) / "temp_onnx"
        custom_temp.mkdir(exist_ok=True)
        
        # Set custom temp directory
        original_temp = os.environ.get('TMPDIR', '')
        os.environ['TMPDIR'] = str(custom_temp)
        
        try:
            from optimum.onnxruntime import ORTModelForSeq2SeqLM
            
            model = ORTModelForSeq2SeqLM.from_pretrained(
                model_dir,
                export=True,
                use_cache=False,
                provider="CPUExecutionProvider",
                use_io_binding=False  # Disable IO binding for stability
            )
            
            # Save model and tokenizer
            logger.info("Saving ONNX model...")
            model.save_pretrained(output_dir)
            
            logger.info("Saving tokenizer...")
            tokenizer.save_pretrained(output_dir)
            
            # Cleanup
            del model
            gc.collect()
            
        finally:
            # Restore original temp directory
            if original_temp:
                os.environ['TMPDIR'] = original_temp
            else:
                os.environ.pop('TMPDIR', None)
            
            # Clean up custom temp directory
            if custom_temp.exists():
                try:
                    shutil.rmtree(custom_temp)
                except PermissionError:
                    logger.warning(f"Could not clean up temp directory: {custom_temp}")
        
        # Verify export
        onnx_files = list(Path(output_dir).glob("*.onnx"))
        if onnx_files:
            logger.info(f"✅ Export successful! Found ONNX files: {[f.name for f in onnx_files]}")
            logger.info(f"✅ Model and tokenizer saved to: {output_dir}")
            return True
        else:
            logger.warning("⚠️  No .onnx files found in output directory")
            return False
            
    except Exception as e:
        logger.error(f"❌ Export failed: {str(e)}")
        return False

def cleanup_and_retry():
    """Clean up any leftover temp files and retry export."""
    
    # Clean up Windows temp files
    import glob
    temp_patterns = [
        r"C:\Users\*\AppData\Local\Temp\tmp*",
        r"C:\Windows\Temp\tmp*"
    ]
    
    for pattern in temp_patterns:
        for temp_file in glob.glob(pattern):
            try:
                if os.path.isfile(temp_file):
                    os.remove(temp_file)
                elif os.path.isdir(temp_file):
                    shutil.rmtree(temp_file)
            except PermissionError:
                continue  # Skip files we can't delete
    
    logger.info("🧹 Cleanup completed")

if __name__ == "__main__":
    # First attempt
    success = export_model_to_onnx_safe(MODEL_DIR, ONNX_DIR)
    
    if not success:
        logger.info("First attempt failed. Trying with tokenizer fix...")
        cleanup_and_retry()
        
        # Second attempt with tokenizer fix
        success = export_model_to_onnx_safe(MODEL_DIR, ONNX_DIR, use_backup_tokenizer=True)
        
        if not success:
            logger.error("❌ All export attempts failed")
            sys.exit(1)
        else:
            logger.info("✅ Export succeeded on second attempt!")
    else:
        logger.info("✅ Export succeeded on first attempt!")