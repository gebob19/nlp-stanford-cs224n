import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    def __init__(self, e_word):
        super(Highway, self).__init__()
        self.proj = nn.Linear(e_word, e_word)
        self.gate = nn.Linear(e_word, e_word)

    def forward(self, conv_out):
        x_proj = F.relu(self.proj(conv_out))
        x_gate = F.sigmoid(self.gate(conv_out))
        return x_gate * x_proj + (1 - x_gate) * conv_out
