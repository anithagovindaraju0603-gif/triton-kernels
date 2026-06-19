import torch
from torch.profiler import profile, ProfilerActivity
from kernel import softmax

x = torch.randn(4096, 4096, device="cuda")

print("=== PyTorch eager softmax ===")
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    torch.softmax(x, dim=-1)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

print("=== Our fused softmax ===")
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    softmax(x)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))