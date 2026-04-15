"""
GPU/CUDA Verification Script
Checks PyTorch GPU availability and details.
"""

import torch
import sys

print("=" * 60)
print("GPU/CUDA Verification for PyTorch")
print("=" * 60)

# PyTorch version info
print(f"\nPyTorch Version: {torch.__version__}")
print(f"Python Version: {sys.version}")

# CUDA availability
has_cuda = torch.cuda.is_available()
print(f"\n✓ CUDA Available: {has_cuda}")

if has_cuda:
    # GPU details
    gpu_count = torch.cuda.device_count()
    print(f"✓ GPU Count: {gpu_count}")
    
    for i in range(gpu_count):
        print(f"\n  GPU {i}:")
        print(f"    Name: {torch.cuda.get_device_name(i)}")
        print(f"    Capability: {torch.cuda.get_device_capability(i)}")
        
        props = torch.cuda.get_device_properties(i)
        print(f"    Memory: {props.total_memory / 1024**3:.2f} GB")
    
    # Current device
    current_device = torch.cuda.current_device()
    print(f"\n✓ Current Device: GPU {current_device}")
    
    # Test tensor operation
    print(f"\n✓ Testing tensor on GPU...")
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.mm(x, y)
    print(f"  Tensor operation successful!")
    print(f"  Result shape: {z.shape}")
    
else:
    print("\n⚠ WARNING: No CUDA-capable GPU detected!")
    print("\nTo enable GPU support:")
    print("  1. Uninstall PyTorch: pip uninstall torch torchvision torchaudio -y")
    print("  2. Install with CUDA 12.1: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("  3. Or CUDA 11.8: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

# DataLoader pin_memory check
print(f"\n{'=' * 60}")
print(f"DataLoader Memory Pinning: pin_memory should be {has_cuda}")
print(f"{'=' * 60}\n")
