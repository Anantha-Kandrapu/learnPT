import torch
import torch.nn as nn


class selfAttention(nn.Module):
    def __init__(self, embed_size, heads) -> None:
        assert embed_size % heads == 0, "Embed size needs to be divisible by heads"
        super(selfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask):
        N = values.shape
        len = values.shape[1]
        values = values.reshape(N, len, self.heads, self.head_dim)
        keys = keys.reshape(N, len, self.heads, self.head_dim)
        queries = queries.reshape(N, len, self.heads, self.head_dim)
        energy = torch.einsum("nqhd,nkhd->nhq", [queries, keys])
        if mask:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy / self.embed_size ** (0.5),dim=3)
