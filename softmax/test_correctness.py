import torch
from kernel import softmax

x = torch.randn(100,100, device = "cuda")
out_triton = softmax(x)
out_pytorch = torch.softmax(x, dim=-1)

assert torch.allclose(out_triton, out_pytorch, atol=1e-2)
print("Correct!")
