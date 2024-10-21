# -*- coding: utf-8 -*-
# @Time    : 2024/8/16 0:00
# @Author  : TkiChus
# @Email   : XU.Chao.TkiChus@gmail.com

import torch
import torch.nn as nn
import torch.nn.functional as F

# class PerformerAttention(nn.Module):
#     def __init__(self, dim, num_heads=8, dim_head=32, kernel_fn=F.relu):
#         super(PerformerAttention, self).__init__()
#         self.num_heads = num_heads
#         self.dim_head = dim_head
#         self.scale = dim_head ** -0.5
#         self.dim = dim
#
#         self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
#         self.kernel_fn = kernel_fn
#         self.to_out = nn.Linear(dim, dim)
#
#     def forward(self, x):
#         B, N = x.shape  # Batch size 和 输入维度
#
#         # 计算 Q, K, V 矩阵
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: t.view(B, 1, self.num_heads, self.dim_head).transpose(1, 2), qkv)
#
#         q = self.kernel_fn(q)
#         k = self.kernel_fn(k)
#
#         # 计算注意力权重
#         attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
#         attn = attn.softmax(dim=-1)
#
#         # 加权求和
#         out = torch.einsum('bhij,bhjd->bhid', attn, v)
#         out = out.transpose(1, 2).reshape(B, self.dim)
#
#         return self.to_out(out)

"""
在你描述的场景中，针对 PerformerAttention 的输入是 (42948, 256) 这样没有 batch_size 维度的数据，去除 batch_size 维度是合理且可行的。你已经通过 train_loader 将一个 batch 的数据整体输入给模型，这种情况下不需要显式地处理 batch_size 维度。因为整个数据集的输入可以视为单个实体，直接应用在 PerformerAttention 上。
合理性分析

    输入维度的匹配：PerformerAttention 处理的是二维数据，如果在前面的 MLP 和激活函数之后的数据已经是 (42948, 256) 这种维度，且不需要进行批次处理，则直接在现有维度上操作是合理的。

    资源优化：引入虚拟 batch_size 维度的情况下，数据会占用更多的 GPU 显存，尤其是像 (42948, 256) 这样大的数据集，可能会导致 GPU 显存不足的情况。所以，去掉 batch_size 维度可以有效地减少计算资源的占用。

    代码实现的可行性：如果你的实现已经在 PerformerAttention 之前经过了 MLP 和激活函数，且这些操作都是在二维数据上进行的，那么直接去除 batch_size 维度后，也可以在 PerformerAttention 中正确地处理这些输入，不会影响注意力机制的计算。

因此，在你的特定应用场景中，不考虑 batch_size 维度，并直接处理二维输入是合理且可行的方案。


"""

class PerformerAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_head=32, kernel_fn=F.relu):
        super(PerformerAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.dim = dim

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.kernel_fn = kernel_fn
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        # x = x.unsqueeze(0)
        N, C = x.shape  # 序列长度 和 输入维度

        assert C == self.dim, "C is equal to the input channel dim"

        # 计算 Q, K, V 矩阵
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(N, self.num_heads, self.dim_head).transpose(0, 1), qkv)

        q = self.kernel_fn(q)
        k = self.kernel_fn(k)

        # 计算注意力权重
        attn = torch.einsum('hid, hjd -> hij', q, k) * self.scale
        attn = attn.softmax(dim=-1)

        # 加权求和
        out = torch.einsum('hij, hjd -> hid', attn, v)
        out = out.transpose(0, 1).reshape(N, C)
        # out = out.squeeze(0)

        return self.to_out(out)





# # 假设实际应用中的输入维度为 [42948, 256]
# x = torch.randn(42948, 256)
#
# # 初始化 PerformerAttention 模块
# performer_attn = PerformerAttention(dim=256, num_heads=8, dim_head=32)
# output = performer_attn(x)
