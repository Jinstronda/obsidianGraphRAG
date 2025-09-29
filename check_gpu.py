"""
Check if CUDA/GPU is available for PyTorch
"""
import torch

print("="*60)
print("GPU AVAILABILITY CHECK")
print("="*60)

# Check CUDA
cuda_available = torch.cuda.is_available()
print(f"\nCUDA Available: {cuda_available}")

if cuda_available:
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
else:
    print("\nNo CUDA GPU detected.")
    print("You'll need to install PyTorch with CUDA support:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

print("\n" + "="*60)