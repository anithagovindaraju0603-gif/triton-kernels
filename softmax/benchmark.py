import torch
import triton
from kernel import softmax
import matplotlib.pyplot as plt

def benchmark(func, x, n_runs=100):
    # warmup
    for _ in range(10):
        func(x)
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(n_runs):
        func(x)
    end.record()
    
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_runs  # ms per call


shapes = [(1823, 781), (4096, 1024), (4096, 4096), (16384, 8192)]

triton_times = []
pytorch_times = []

for (M, N) in shapes:
    x = torch.randn(M, N, device="cuda")
    
    t_triton = benchmark(softmax, x)
    t_pytorch = benchmark(lambda x: torch.softmax(x, dim=-1), x)
    
    triton_times.append(t_triton)
    pytorch_times.append(t_pytorch)
    
    # bandwidth calculation
    # bytes read + bytes written = 2 * M * N * 4 (float32 = 4 bytes)
    bytes_moved = 2 * M * N * 4
    bandwidth_gb = (bytes_moved / (t_triton * 1e-3)) / 1e9  # GB/s
    
    print(f"Shape ({M}, {N}):")
    print(f"  Triton:  {t_triton:.3f} ms")
    print(f"  PyTorch: {t_pytorch:.3f} ms")
    print(f"  Speedup: {t_pytorch/t_triton:.2f}x")
    print(f"  Bandwidth: {bandwidth_gb:.1f} GB/s")