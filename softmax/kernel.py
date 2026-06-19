import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    inp_ptr,
    out_ptr,
    inp_stride,
    out_stride,
    n_col,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    row_starts = inp_ptr + pid * inp_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    inp_addrs = row_starts + col_offsets
    masked = col_offsets < n_col

    x = tl.load(inp_addrs, mask = masked, other = -float('inf'))

    x_max = tl.max(x, axis = 0)
    z = x - x_max
    numerator = tl.exp(z)
    denominator = tl.sum(numerator, axis = 0)

    sm = numerator/denominator
    
    #storing the results back to memory
    out_row_starts = out_ptr + pid * out_stride
    out_addrs = out_row_starts + col_offsets
    tl.store(out_addrs, sm, mask = masked)

def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)

    softmax_kernel[(n_rows,)](x, y, x.stride(0), y.stride(0), n_cols, BLOCK_SIZE)

    return y