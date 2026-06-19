import torch 

def naive_softmax(x):
    x_max = x.max(dim=-1, keepdim=True).values  # kernel 1
    z = x - x_max                                # kernel 2
    numerator = torch.exp(z)                     # kernel 3
    denominator = numerator.sum(dim=-1, keepdim=True)  # kernel 4
    return numerator / denominator               # kernel 5