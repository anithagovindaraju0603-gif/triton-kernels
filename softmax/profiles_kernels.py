import torch
from torch.profiler import profile, ProfilerActivity
from kernel import softmax
from naive_softmax import naive_softmax
import pandas as pd

x = torch.randn(4096, 4096, device="cuda")

def profile_to_df(func, x, trace_name):
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        func(x)
    
    prof.export_chrome_trace(f"../results/{trace_name}.json")
    
    rows = []
    for evt in prof.key_averages():
        rows.append({
            "Name": evt.key,
            "CPU Total (us)": evt.cpu_time_total,
            "CUDA Total (us)": evt.device_time_total,
            "# of Calls": evt.count,
        })
    return pd.DataFrame(rows).sort_values("CUDA Total (us)", ascending=False)


df_naive = profile_to_df(naive_softmax, x, "naive_trace")
df_pytorch = profile_to_df(lambda x: torch.softmax(x, dim=-1), x, "pytorch_trace")
df_triton = profile_to_df(softmax, x, "triton_trace")

with pd.ExcelWriter("../results/profile_output.xlsx", engine="openpyxl") as writer:
    df_naive.to_excel(writer, sheet_name="Naive Softmax", index=False)
    df_pytorch.to_excel(writer, sheet_name="PyTorch Softmax", index=False)
    df_triton.to_excel(writer, sheet_name="Triton Softmax", index=False)

print("Saved to ../results/profile_output.xlsx")
print("Traces saved as naive_trace.json, pytorch_trace.json, triton_trace.json")
print("View them at chrome://tracing/ in Chrome browser")