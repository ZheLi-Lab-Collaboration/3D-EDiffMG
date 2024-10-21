# -*- coding: utf-8 -*-
# @Time    : 2024/8/5 21:05
# @Author  : TkiChus
# @Email   : XU.Chao.TkiChus@gmail.com


import torch
import torch.nn as nn
import torch.nn.functional as F

class gMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(gMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.gelu = nn.GELU()
        self.sgu = SpatialGatingUnit(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.norm2 = nn.LayerNorm(output_dim)

    def forward(self, x):
        # Feed-forward
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.gelu(x)
        x = self.sgu(x)
        x = self.fc2(x)
        x = self.norm2(x)
        return x

class SpatialGatingUnit(nn.Module):
    def __init__(self, hidden_dim):
        super(SpatialGatingUnit, self).__init__()
        self.norm = nn.LayerNorm(hidden_dim // 2)
        self.proj = nn.Conv1d(hidden_dim // 2, hidden_dim // 2, kernel_size=1)
        nn.init.constant_(self.proj.bias, 1.0)

    def forward(self, x):
        # Split the tensor along the last dimension
        u, v = x.chunk(2, dim=-1)
        # Normalize v
        v = self.norm(v)
        # Transpose v for convolution
        v = v.transpose(1, 2)
        # Apply convolution
        v = self.proj(v)
        # Transpose v back to original shape
        v = v.transpose(1, 2)
        # Element-wise multiplication
        return u * v

# Example usage:
input_dim = 512
hidden_dim = 1024
output_dim = 512
x = torch.randn(64, 128, input_dim)  # (batch_size, seq_len, input_dim)
model = gMLP(input_dim, hidden_dim, output_dim)
output = model(x)
print(output.shape)  # Should print: torch.Size([64, 128, output_dim])
