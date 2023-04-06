import einsum
import torch

x = torch.tensor([[1,2,3],[4,5,6]])
print(x.permute(1,0))
