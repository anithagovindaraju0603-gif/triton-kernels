import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

x = torch.randn(1024, device='cuda')
y = torch.randn(1024, device='cuda')
out = torch.empty_like(x)
add_kernel[(triton.cdiv(1024, 256),)](x, y, out, 1024, BLOCK=256)
assert torch.allclose(out, x + y)
print('Triton works.')
