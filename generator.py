import torch
import torch.nn as nn
import string
import sys
import unidecode
import random
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file = unidecode.unidecode(open('../textdata/names.txt').read())

class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size) -> None:
        super(RNN,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(input_size,hidden_size)

class Generator():
    pass
