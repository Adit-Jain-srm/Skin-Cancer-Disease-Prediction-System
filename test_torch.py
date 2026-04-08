import sys
print(f"Python: {sys.version}")
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"Error loading PyTorch: {e}")
    import traceback
    traceback.print_exc()
