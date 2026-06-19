import torch
import triton
from kernel import softmax
from naive_softmax import naive_softmax
import pandas as pd

def benchmark(func, x, n_runs=1000): 
    # warmup
    for _ in range(50):
        func(x)
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(n_runs):
        func(x)
    end.record()
    
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_runs


shapes = [(1823, 781), (4096, 1024), (4096, 4096), (16384, 8192)]

naive_rows = []
triton_rows = []
pytorch_rows = []

for (M, N) in shapes:
    x = torch.randn(M, N, device="cuda")
    
    t_naive = benchmark(naive_softmax, x)
    t_triton = benchmark(softmax, x)
    t_pytorch = benchmark(lambda x: torch.softmax(x, dim=-1), x)
    
    # bandwidth calculation
    # For fused kernels (Triton, torch.softmax): 1 read + 1 write = 2 * M * N * 4 bytes
    # For naive softmax: 5 reads + 3 writes = 8 * M * N * 4 bytes (per Triton tutorial)
    bytes_fused = 2 * M * N * 4
    bytes_naive = 8 * M * N * 4

    bw_naive = (bytes_naive / (t_naive / 1000)) / 1e9
    bw_triton = (bytes_fused / (t_triton / 1000)) / 1e9
    bw_pytorch = (bytes_fused / (t_pytorch / 1000)) / 1e9
    
    naive_rows.append({
        "Shape": f"({M}, {N})",
        "Time (ms)": round(t_naive, 4),
        "Bandwidth (GB/s)": round(bw_naive, 1),
    })
    triton_rows.append({
        "Shape": f"({M}, {N})",
        "Time (ms)": round(t_triton, 4),
        "Bandwidth (GB/s)": round(bw_triton, 1),
        "Speedup vs Naive": round(t_naive / t_triton, 2),
        "Speedup vs PyTorch": round(t_pytorch / t_triton, 2),
    })
    pytorch_rows.append({
        "Shape": f"({M}, {N})",
        "Time (ms)": round(t_pytorch, 4),
        "Bandwidth (GB/s)": round(bw_pytorch, 1),
    })
    
    print(f"Shape ({M}, {N}):")
    print(f"  Naive:   {t_naive:.3f} ms")
    print(f"  Triton:  {t_triton:.3f} ms")
    print(f"  PyTorch: {t_pytorch:.3f} ms")
    print(f"  Speedup vs Naive:   {t_naive/t_triton:.2f}x")
    print(f"  Speedup vs PyTorch: {t_pytorch/t_triton:.2f}x")
    print()

df_naive = pd.DataFrame(naive_rows)
df_triton = pd.DataFrame(triton_rows)
df_pytorch = pd.DataFrame(pytorch_rows)

with pd.ExcelWriter("../results/benchmark_results.xlsx", engine="openpyxl") as writer:
    df_naive.to_excel(writer, sheet_name="Naive Softmax", index=False)
    df_pytorch.to_excel(writer, sheet_name="PyTorch Softmax", index=False)
    df_triton.to_excel(writer, sheet_name="Triton Softmax", index=False)

print("Saved to ../results/benchmark_results.xlsx")