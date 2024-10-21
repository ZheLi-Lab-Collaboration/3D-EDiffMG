# -*- coding: utf-8 -*-
# @Time    : 2024/8/8 22:28
# @Author  : TkiChus
# @Email   : XU.Chao.TkiChus@gmail.com



from math import pi as PI

import torch
import torch.nn.functional as F
from torch.nn import Module, Sequential, ModuleList, Linear, Embedding
from torch_geometric.nn import MessagePassing
from .DGA_Module.caFilter import CFCA_module
from .ACT.act_gelus import gelus_gt2_fun, soft_plus_fun

class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class AsymmetricSineCosineSmearing(Module):

    def __init__(self, num_basis=50):
        super().__init__()
        num_basis_k = num_basis // 2
        num_basis_l = num_basis - num_basis_k
        self.register_buffer('freq_k', torch.arange(1, num_basis_k + 1).float())
        self.register_buffer('freq_l', torch.arange(1, num_basis_l + 1).float())

    @property
    def num_basis(self):
        return self.freq_k.size(0) + self.freq_l.size(0)

    def forward(self, angle):
        # If we don't incorporate `cos`, the embedding of 0-deg and 180-deg will be the
        #  same, which is undesirable.
        s = torch.sin(angle.view(-1, 1) * self.freq_k.view(1, -1))  # (num_angles, num_basis_k)
        c = torch.cos(angle.view(-1, 1) * self.freq_l.view(1, -1))  # (num_angles, num_basis_l)
        return torch.cat([s, c], dim=-1)


class SymmetricCosineSmearing(Module):

    def __init__(self, num_basis=50):
        super().__init__()
        self.register_buffer('freq_k', torch.arange(1, num_basis + 1).float())

    @property
    def num_basis(self):
        return self.freq_k.size(0)

    def forward(self, angle):
        return torch.cos(angle.view(-1, 1) * self.freq_k.view(1, -1))  # (num_angles, num_basis)


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


# class CFConv(MessagePassing):
#     def __init__(self, in_channels, out_channels, num_filters, nn, cutoff, smooth):
#         super(CFConv, self).__init__(aggr='add')
#         self.lin1 = Linear(in_channels, num_filters, bias=False)
#         self.lin2 = Linear(num_filters, out_channels)
#         self.nn = nn
#         self.cutoff = cutoff
#         self.smooth = smooth
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         torch.nn.init.xavier_uniform_(self.lin1.weight)
#         torch.nn.init.xavier_uniform_(self.lin2.weight)
#         self.lin2.bias.data.fill_(0)
#
#     def forward(self, x, edge_index, edge_length, edge_attr):
#         if self.smooth:
#             C = 0.5 * (torch.cos(edge_length * PI / self.cutoff) + 1.0)
#             C = C * (edge_length <= self.cutoff) * (edge_length >= 0.0)  # Modification: cutoff
#         else:
#             C = (edge_length <= self.cutoff).float()
#         W = self.nn(edge_attr) * C.view(-1, 1)
#
#         x = self.lin1(x)
#         x = self.propagate(edge_index, x=x, W=W)
#         x = self.lin2(x)
#         return x
#
#     def message(self, x_j, W):
#         return x_j * W



class CACFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, ca_filter_nn, cutoff, smooth):
        super(CACFConv, self).__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.ca_filter_nn = ca_filter_nn
        self.cutoff = cutoff
        self.smooth = smooth
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_length, edge_attr):
        if self.smooth:
            C = 0.5 * (torch.cos(edge_length * PI / self.cutoff) + 1.0)
            C = C * (edge_length <= self.cutoff) * (edge_length >= 0.0)  # Modification: cutoff
        else:
            C = (edge_length <= self.cutoff).float()
        # 这里的nn就是要替换的ca_Filter
        W = self.ca_filter_nn(edge_attr) * C.view(-1, 1) # view(-1,1) --> (N, 1)

        x = self.lin1(x)
        """
    前向传播 (forward 方法)
    x: 节点特征矩阵，形状为 (num_nodes, in_channels)。
    edge_index: 边索引矩阵，形状为 (2, num_edges)，其中第一行是源节点索引，第二行是目标节点索引。
    edge_length: 每条边的长度，形状为 (num_edges,)。
    edge_attr: 每条边的属性，形状为 (num_edges, edge_attr_dim)。

    在 forward 方法中，首先根据 edge_length 和 cutoff 计算出修正因子 C，然后计算每条边的权重 W，最终调用 self.propagate(edge_index, x=x, W=W) 开始消息传递过程。

    调用 self.propagate:
    self.propagate(edge_index, x=x, W=W) 是消息传递的核心，它控制了整个消息传递的流程。propagate 方法的输入包括边的连接关系 edge_index，以及在 forward 方法中传递的节点特征 x 和边权重 W。

    propagate 内部执行 message:
    在 propagate 方法中，message 方法会被自动调用，用于计算从源节点到目标节点的消息。
    具体地，propagate 方法会遍历 edge_index 中的每一条边，并根据源节点的特征 x_j 和边的权重 W 计算消息。在 message 方法中，这一操作表现为 x_j * W，表示源节点特征 x_j 被权重 W 调整后作为消息传递到目标节点。

    message 函数的输入:
    x_j: 是从 edge_index[0]（即源节点索引）获取到的源节点的特征，形状为 (num_edges, num_filters)。
    W: 是前面通过边属性和修正因子计算出来的边权重，形状为 (num_edges, 1)。

    message 函数的输出:
    message 函数返回的结果是 x_j * W，这是消息的最终形式，即源节点特征按权重缩放后的值。这个消息将会被传递到目标节点，用于聚合（aggregation）。

    聚合 (aggregate) 和更新 (update):
    在 message 计算完每条边的消息后，propagate 方法会自动调用 aggregate 方法将所有传入的消息聚合到目标节点上。在这个例子中，由于 aggr='add'，聚合方式是将所有消息相加。
    之后，update 方法（如果存在）会用聚合后的结果更新节点特征。

    后续处理:

    最后，forward 方法中聚合后的节点特征会经过 self.lin2(x) 进行线性变换，并返回最终的输出。
        """
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W

# class InteractionBlock(torch.nn.Module):
#     def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff, smooth):
#         super(InteractionBlock, self).__init__()
#         mlp = Sequential(
#             Linear(num_gaussians, num_filters),
#             ShiftedSoftplus(),
#             Linear(num_filters, num_filters),
#         )
#         self.conv = CFConv(hidden_channels, hidden_channels, num_filters, mlp, cutoff, smooth)
#         self.act = ShiftedSoftplus()
#         self.lin = Linear(hidden_channels, hidden_channels)
#
#     def forward(self, x, edge_index, edge_length, edge_attr):
#         x = self.conv(x, edge_index, edge_length, edge_attr)
#         x = self.act(x)
#         x = self.lin(x)
#         return x



class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, caFilter_per_block, cutoff, smooth, module_name):
        super(InteractionBlock, self).__init__()
        # 替换这里的mlp
        # mlp = Sequential(
        #     Linear(num_gaussians, num_filters),
        #     ShiftedSoftplus(),
        #     Linear(num_filters, num_filters),
        # )

        self.ca_filter_nn = CFCA_module(hidden_channels,
                                        num_filters,
                                        num_filters,
                                        num_gaussians,
                                        layer_per_block=caFilter_per_block,
                                        module_name=module_name)


        self.cacfconv = CACFConv(hidden_channels,
                                 hidden_channels,
                                 num_filters,
                                 self.ca_filter_nn,
                                 cutoff,
                                 smooth)
        self.act = gelus_gt2_fun
        self.lin = Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_length, edge_attr):
        x = self.cacfconv(x, edge_index, edge_length, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class caSchNetEncoder(Module):

    def __init__(self,
                 hidden_channels=128,
                 num_filters=128,
                 num_interactions=7,
                 edge_channels=128,
                 # n_gaussians=25,
                 caFilter_per_block=4,
                 cutoff=10.0,
                 smooth=False,
                 input_dim=5,
                 time_emb=True,
                 context=False,
                 module_name="DGA_Layer",
                 ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.cutoff = cutoff
        self.input_dim = input_dim
        self.time_emb = time_emb
        self.embedding = Embedding(100, hidden_channels, max_norm=10.0)
        self.emblin = Linear(self.input_dim, hidden_channels)  # 16 or 8
        self.context = context

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            # block = InteractionBlock(hidden_channels, edge_channels,
            #                          num_filters, cutoff, smooth)
            block = InteractionBlock(hidden_channels,
                                     edge_channels,
                                     num_filters,
                                     caFilter_per_block,
                                     cutoff,
                                     smooth,
                                     module_name)
            self.interactions.append(block)

        if context:
            self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=hidden_channels)
            self.fc1_m = Linear(hidden_channels, 256)
            self.fc2_m = Linear(256, 64)
            self.fc3_m = Linear(64, input_dim)

            # Mapping to [c], cmean
            self.fc1_v = Linear(hidden_channels, 256)
            self.fc2_v = Linear(256, 64)
            self.fc3_v = Linear(64, input_dim)

    def forward(self, z, edge_index, edge_length, edge_attr, embed_node=True):
        if self.context:
            edge_attr = self.distance_expansion(edge_length)
        # 这里的z其实就是atom_type
        if z.dim() == 1 and z.dtype == torch.long:
            assert z.dim() == 1 and z.dtype == torch.long
            h = self.embedding(z)

        else:
            # h = z # default
            if self.time_emb:
                z, temb = z[:, :self.input_dim], z[:, self.input_dim:]
                h = self.emblin(z) + temb
            else:
                h = self.emblin(z)
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_length, edge_attr)

        if self.context:
            m = F.relu(self.fc1_m(h))
            m = F.relu(self.fc2_m(m))
            m = self.fc3_m(m)
            v = F.relu(self.fc1_v(h))
            v = F.relu(self.fc2_v(v))
            v = self.fc3_v(v)
            return m, v
        else:
            return h

