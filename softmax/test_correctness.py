import torch
from kernel import softmax

x = torch.randn(100,100, device = "cuda")
print(f"Input matrix for softmax: {x}")
out_triton = softmax(x)
out_pytorch = torch.softmax(x, dim=-1)
print(f"Triton output:{out_triton}")
print(f"PyTorch output:{out_pytorch}")

assert torch.allclose(out_triton, out_pytorch, atol=1e-2)
print("Correct!")
