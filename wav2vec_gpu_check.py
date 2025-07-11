import torch
import platform

print("--- PyTorch Device Check ---")

# OS info
ios = platform.system()
print(f"Operating System: {ios}")

# CUDA (NVIDIA)
cuda_available = torch.cuda.is_available()
cuda_count = torch.cuda.device_count() if cuda_available else 0
print(f"CUDA available: {cuda_available}")
print(f"CUDA device count: {cuda_count}")
if cuda_available:
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

# ROCm (AMD)
rocm_version = getattr(torch.version, 'hip', None)
print(f"ROCm (HIP) version: {rocm_version}")

# MPS (Apple Silicon)
mps_available = getattr(torch.backends, 'mps', None)
if mps_available:
    print(f"MPS available: {torch.backends.mps.is_available()}")

# Device summary
device = None
if cuda_available:
    device = 'cuda'
elif rocm_version is not None:
    device = 'rocm'
elif mps_available and torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f"Selected device: {device}")

# Windows + AMD warning
if ios == 'Windows' and rocm_version is not None:
    print("WARNING: ROCm (AMD GPU) is not supported on Windows. You must use Linux for ROCm/AMD GPU acceleration.")

if device == 'cpu':
    print("WARNING: No supported GPU/accelerator found. Training will be very slow.")
else:
    print(f"SUCCESS: You can use {device.upper()} for training in PyTorch!") 