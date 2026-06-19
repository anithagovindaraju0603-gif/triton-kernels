import torch
from torch.profiler import profile, ProfilerActivity
from kernel import softmax
from naive_softmax import naive_softmax

x = torch.randn(4096, 4096, device="cuda")

with open("../results/profile_output.txt", "w") as f:
    
    f.write("=== Naive softmax (manual) ===\n")
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        naive_softmax(x)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    f.write("\n\n")

    f.write("=== PyTorch torch.softmax ===\n")
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.softmax(x, dim=-1)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    f.write("\n\n")

    f.write("=== Our Triton fused softmax ===\n")
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        softmax(x)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    f.write("\n")

print("Profile output saved to ../results/profile_output.txt")