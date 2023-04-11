import einsum
import torch

mask = torch.tensor([[0, 0, 0], [0, 1, 0], [0,1,1]])
x = torch.tensor([[1, 2, 3], [4, 5,0.6], [4, 6, 34]])

x = x.masked_fill(mask == 0, float("0.234"))
print(x)
