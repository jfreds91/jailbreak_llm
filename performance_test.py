#!/usr/bin/env python3
"""
AMD RX 9070 XT Performance Test
Tests if the GPU is functioning properly despite misidentification
"""

import torch
import time

print("=" * 70)
print("AMD RX 9700 XT GPU Performance Test")
print("=" * 70)

# GPU Info
print(f"\n📊 GPU Information:")
print(f"  Detected name: {torch.cuda.get_device_name(0)}")
print(f"  Device count: {torch.cuda.device_count()}")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  PyTorch version: {torch.__version__}")
print(f"  ROCm version: {torch.version.hip}")

# Memory info
print(f"\n💾 GPU Memory:")
props = torch.cuda.get_device_properties(0)
print(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")
print(f"  Multi-processor count: {props.multi_processor_count}")

# Quick functionality test
print(f"\n✅ Functionality Test:")
try:
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    y = x * 2
    assert y.device.type == 'cuda'
    print(f"  ✓ Basic tensor operations work")
except Exception as e:
    print(f"  ✗ Error: {e}")
    exit(1)

# Performance benchmarks
print(f"\n⚡ Performance Benchmarks:")

sizes = [1024, 2048, 4096, 8192]
for size in sizes:
    x = torch.randn(size, size).cuda()
    y = torch.randn(size, size).cuda()
    
    # Warmup
    for _ in range(3):
        _ = torch.matmul(x, y)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    
    iterations = 10
    for _ in range(iterations):
        z = torch.matmul(x, y)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    avg_time = elapsed / iterations
    
    # Calculate GFLOPS (2*n^3 operations for matrix multiply)
    gflops = (2 * size**3) / (avg_time * 1e9)
    
    print(f"  Matrix multiply {size}x{size}:")
    print(f"    Average time: {avg_time*1000:.2f} ms")
    print(f"    Performance: {gflops:.1f} GFLOPS")

# Test mixed precision (important for modern GPUs)
print(f"\n🔬 Mixed Precision Test (FP16):")
try:
    x_fp16 = torch.randn(4096, 4096, dtype=torch.float16).cuda()
    y_fp16 = torch.randn(4096, 4096, dtype=torch.float16).cuda()
    
    torch.cuda.synchronize()
    start = time.time()
    z_fp16 = torch.matmul(x_fp16, y_fp16)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"  ✓ FP16 support working")
    print(f"  4096x4096 FP16 matmul: {elapsed*1000:.2f} ms")
except Exception as e:
    print(f"  ✗ FP16 error: {e}")

# Memory transfer test
print(f"\n🔄 Memory Transfer Test:")
size_mb = 1000
data = torch.randn(size_mb * 1024 * 1024 // 4)  # 1000 MB

start = time.time()
data_gpu = data.cuda()
torch.cuda.synchronize()
h2d_time = time.time() - start

start = time.time()
data_cpu = data_gpu.cpu()
torch.cuda.synchronize()
d2h_time = time.time() - start

print(f"  Host to Device ({size_mb} MB): {h2d_time*1000:.1f} ms ({size_mb/h2d_time:.1f} MB/s)")
print(f"  Device to Host ({size_mb} MB): {d2h_time*1000:.1f} ms ({size_mb/d2h_time:.1f} MB/s)")

print("\n" + "=" * 70)
print("Test complete! If all tests passed, your GPU is working correctly.")
print("Note: Performance may improve with ROCm 6.4 runtime optimized for RDNA 4")
print("=" * 70)