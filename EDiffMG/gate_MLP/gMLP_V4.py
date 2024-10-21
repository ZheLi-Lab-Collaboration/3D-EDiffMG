# -*- coding: utf-8 -*-
# @Time    : 2024/8/18 19:00
# @Author  : TkiChus
# @Email   : XU.Chao.TkiChus@gmail.com




from random import randrange
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

# functions

def exists(val):
    return val is not None

def  dropout_layers(layers, prob_survival):

    if prob_survival == 1:
        return layers

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) > prob_survival

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers

class Residual(nn.Module):

    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class Attention(nn.Module):
    """
    tiny attention
    """
    def __init__(self, dim_in, dim_out, dim_inner):
        super(Attention, self).__init__()
        self.scale = dim_inner ** -0.5
        self.to_qkv = nn.Linear(dim_in, dim_inner*3, bias=False)
        self.to_out = nn.Linear(dim_inner, dim_out)

    def forward(self, x):
        x = x.unsqueeze(0)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = out.squeeze(0)

        return self.to_out(out)


class SpatialGatingUnit(nn.Module):
    def __init__(self,
                 dim,
                 act=nn.Identity()):
        super(SpatialGatingUnit, self).__init__()

        dim_out = dim // 2

        self.norm = nn.LayerNorm(dim_out)
        self.act = act

        self.spatial_proj = nn.Linear(dim_out, dim_out)
        nn.init.constant_(self.spatial_proj.bias, 1.0)

    def forward(self, x, gate_res=None):
        res, gate = x.chunk(2, dim=-1)
        gate = self.norm(gate)
        if exists(gate_res):
            gate = gate_res + self.spatial_proj(gate)
        else:
            gate = self.spatial_proj(gate)

        out = res * gate
        return self.act(out)


class gMLP_Bolck(nn.Module):
    def __init__(self,
                 *,
                 dim,
                 dim_ff,
                 attn_dim=None,
                 act=nn.Identity(),
                 ):
        super(gMLP_Bolck, self).__init__()

        self.proj_in = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.GELU()
        )

        self.attn = Attention(dim, dim_ff//2, attn_dim) if exists(attn_dim) else None
        self.sgu = SpatialGatingUnit(dim_ff, act)
        self.proj_out = nn.Linear(dim_ff//2, dim)

    def forward(self, x):
        gate_res = self.attn(x) if exists(self.attn) else None
        x = self.proj_in(x)
        x = self.sgu(x, gate_res=gate_res)
        x = self.proj_out(x)
        return x


class gate_MultiLayerPerceptron(nn.Module):
    def __init__(self,
                 *,
                 num_tokens=None,
                 dim,
                 depth,
                 ff_mult=4,
                 attn_dim=None,
                 prob_survival=1.,
                 act=nn.Identity()):
        super(gate_MultiLayerPerceptron, self).__init__()

        dim_ff = dim * ff_mult
        self.prob_survival = prob_survival

        self.layers = nn.ModuleList([
            Residual(
                PreNorm(dim,
                        gMLP_Bolck(
                            dim=dim,
                            dim_ff=dim_ff,
                            attn_dim=attn_dim,
                            act=act,
                        )))
        for _ in range(depth)])

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        ) if exists(num_tokens) else nn.Identity

    def forward(self, x):
        layers = self.layers if not self.training else dropout_layers(
            self.layers,
            self.prob_survival,
        )
        out = nn.Sequential(*layers)(x)
        return self.to_logits(out)
