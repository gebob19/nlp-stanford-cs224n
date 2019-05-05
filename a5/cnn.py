import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, e_char):
        self.conv = nn.Conv1d(e_char, e_char, 5)

    def forward(self, reshaped):
        x = F.relu(conv(reshaped))
        return torch.max(x, dim=-1).squeeze()