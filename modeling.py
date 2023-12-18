import torch
import torch.nn as nn
import torch.nn.functional as F

def set_device():
    print(torch.cuda.device_count())
    device = input('Choose device:')
    return device



class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted
    
class P2Q(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(P2Q, self).__init__()
        self.input_dim = input_dim
        self.bilstm_1 = nn.LSTM(input_dim,
                                hidden_dim,
                                num_layers=2,
                                bias=True,
                                batch_first=False,
                                dropout=0,
                                bidirectional=True)
        self.s_attn = SelfAttention()

    def forward(self, x):
        x = self.bilstm_1(x)
        x = self.s_attn(x)
        return x
    
    def train(model):