# -*- coding: utf-8 -*-
# @Time    : 2024/8/5 23:36
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
    '''
    Parameters
    ----------
    val: parameters or methods

    Returns
    -------

    '''
    return val is not None

def pair(val):
    '''
    Parameters
    ----------
    val: pair data

    Returns
    combine pair data
    -------

    '''
    return (val, val) if not isinstance(val,  tuple) else val

def  dropout_layers(layers, prob_survival):
    """
    Randomly discard some layers based on survival probability.
    If all layers are discarded, keep at least one layer.

    Parameters
    ----------
    layers: neural network layer
    prob_survival: %


    Returns
    layers
    -------

    """
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


def shift(t, amount, mask=None):
    """
    Shift the input tensor by a specified amount.
    If the number is 0, the original tensor is returned.
    Otherwise, shift the tensor by the specified number
    and fill it with zeros.
    Parameters
    ----------
    t
    amount
    mask

    Returns
    -------

    """
    if amount == 0:
        return t
    return F.pad(t, (0, 0, amount, -amount), value=0.)



### help classes

class Residual(nn.Module):
    """
    A residual block that adds the input back
    to the original input after it has been processed
    by a function.

    """
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class PreShiftTokens(nn.Module):
    """
    Translation of some part of the input tensor
    and then apply it to the specified function.
    shifts Indicates the amount of shifts per part.

    """
    def __init__(self, shifts, fn):
        super(PreShiftTokens, self).__init__()
        self.shifts = tuple(shifts)
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.shifts == (0., ):
            return self.fn(x, **kwargs)

        shifts = self.shifts
        segments = len(shifts)
        feats_per_shifts = x.shape[-1]

        splitted = x.split(feats_per_shifts, dim=-1)
        segments_to_shifts, rest = splitted[:segments], splitted[segments:]
        segments_to_shifts = list(map(lambda args: shift(*args), zip(segments_to_shifts, shifts)))
        x = torch.cat((*segments_to_shifts, *rest), dim=-1)
        return self.fn(x, **kwargs)


class PreNorm(nn.Module):
    """
    Performs layer normalization of the input
    before applying the specified function.
    """
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


# class Attention(nn.Module):
#     """
#     tiny attention
#     Attention mechanism implementation.
#     Project input to queries, keys, and values,
#     and calculate attention scores. If it is a
#     causal model, the future moment is masked.
#
#     """
#     def __init__(self, dim_in, dim_out, dim_inner, causal=False):
#         super(Attention, self).__init__()
#         self.scale = dim_inner ** -0.5
#         self.causal = causal
#
#         self.to_qkv = nn.Linear(dim_in, dim_inner*3, bias=False)
#         self.to_out = nn.Linear(dim_inner, dim_out)
#
#     def forward(self, x):
#         device = x.device
#         q, k, v = self.to_qkv(x).chunk(3, dim=-1)
#         sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
#
#         if self.causal:
#             mask = torch.ones(sim.shape[-2:], device = device).triu(1).bool()
#             sim.masked_fill_(mask[None, ...], -torch.finfo(q.dtype).max)
#
#         attn = sim.softmax(dim=-1)
#         out = einsum('b i j, b j d -> b i d', attn, v)
#         print("out.shape", out.shape)
#         print("self.to_out(out)", (self.to_out(out)).shape)
#         return self.to_out(out)

class Attention(nn.Module):
    """
    tiny attention
    Attention mechanism implementation.
    Project input to queries, keys, and values,
    and calculate attention scores. If it is a
    causal model, the future moment is masked.

    """
    def __init__(self, dim_in, dim_out, dim_inner, causal=False):
        super(Attention, self).__init__()
        self.scale = dim_inner ** -0.5
        self.causal = causal

        self.to_qkv = nn.Linear(dim_in, dim_inner*3, bias=False)
        self.to_out = nn.Linear(dim_inner, dim_out)

    def forward(self, x):
        device = x.device
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        sim = einsum('i d, j d -> i j', q, k) * self.scale

        if self.causal:
            mask = torch.ones(sim.shape[-2:], device=device).triu(1).bool()
            sim.masked_fill_(mask, -torch.finfo(q.dtype).max)

        attn = sim.softmax(dim=-1)
        out = einsum('i j, j d -> i d', attn, v)
        print("out.shape", out.shape)
        print("self.to_out(out)", (self.to_out(out)).shape)
        return self.to_out(out)



class SpatialGatingUnit(nn.Module):
    def __init__(self,
                 # 输入维度
                 dim,
                 # 序列维度
                 dim_seq,
                 # 因果掩码
                 causal=False,
                 act=nn.Identity(),
                 heads=1,
                 init_eps=1e-3,
                 # 是否使用循环矩阵
                 circulant_matrix=False):
        super(SpatialGatingUnit, self).__init__()

        dim_out = dim // 2
        self.heads = heads
        self.causal = causal
        self.norm = nn.LayerNorm(dim_out)
        self.act = act

        if circulant_matrix:
            #  这两个参数用于构建循环举证的位置偏置
            self.circulant_pos_x = nn.Parameter(torch.ones(heads, dim_seq))
            self.circulant_pos_y = nn.Parameter(torch.ones(heads, dim_seq))

        self.circulant_matrix = circulant_matrix

        # 如果使用循环矩阵,则形状为前者, 否则为后者
        shape = (heads, dim_seq,) if circulant_matrix else (heads, dim_seq, dim_seq)
        weight = torch.zeros(shape)

        self.weight = nn.Parameter(weight)

        init_eps /= dim_seq
        # 对权重进行均匀分布初始化
        nn.init.uniform_(self.weight, -init_eps, init_eps)
        # 对bias进行初始化
        self.bias = nn.Parameter(torch.ones(heads, dim_seq))

    def forward(self, x, gate_res=None):
        device, n, h = x.device, x.shape[0], self.heads
        # 函数会将输入张量（input）沿着指定维度（dim）均匀的分割成特定数量的张量块（chunks），
        # 并返回元素为张量块的元组。
        res, gate = x.chunk(2, dim=-1)
        # 对 gate进行标准化
        gate = self.norm(gate)

        weight, bias = self.weight, self.bias

        if self.circulant_matrix:
            # build the circulant matrix
            # 获取序列维度
            dim_seq = weight.shape[-1]
            # 对权重矩阵进行填充
            weight = F.pad(weight, (0, dim_seq), value=0)
            """
            # 初始权重
            weight = [[1, 2], 
                      [3, 4],
                      [5, 6]]  # 形状: (3, 2)
            dim_seq = 4

            # 应用重复操作
            new_weight = repeat(weight, '... n -> ... (r n)', r=dim_seq)

            # 结果
            # new_weight = [[1, 1, 1, 1, 2, 2, 2, 2],
            #               [3, 3, 3, 3, 4, 4, 4, 4],
            #               [5, 5, 5, 5, 6, 6, 6, 6]]
            # 新形状: (3, 8)
            """
            weight = repeat(weight, '... n -> ... (r n)', r=dim_seq)
            weight = weight[:, :-dim_seq].reshape(h, dim_seq, 2 * dim_seq - 1)
            weight = weight[:, :, (dim_seq - 1):]

            ## give circulant matrix absolute position awareness

            pos_x, pos_y = self.circulant_pos_x, self.circulant_pos_y
            # ()表示添加一个新的空维度
            weight = weight * rearrange(pos_x, 'h i -> h i ()') * rearrange(pos_y, 'h j -> h () j')

        if self.causal:
            # 截取权重和偏置，使其适应当前输入的序列长度
            weight, bias = weight[:, :n, :n], bias[:, :n]
            # 创建上三角掩码，防止未来的信息影响当前时刻。
            mask = torch.ones(weight.shape[-2:], device=device).triu_(1).bool()
            # mask = torch.triu(torch.ones(weight.shape[-2:], device=x.device), diagonal=1).bool()
            mask = rearrange(mask, 'i j -> () i j')
            # 对权重矩阵应用掩码，屏蔽未来时刻的权重。
            weight = weight.masked_fill(mask, 0.)

        # 将 gate 重新排列以匹配多头的形状。
        """
        输入形状 'b h n d':

        b: 批次大小（batch size）
        h: 可能代表头的数量（number of heads）
        n: 可能是序列长度（sequence length）
        d: 每个头的维度（dimension per head）
        输出形状 'b n (h d)':

        b: 批次大小保持不变
        n: 序列长度保持不变
        (h d): 头的数量和每个头的维度被合并成一个新的维度

        """
        gate = rearrange(gate, 'n (h d) -> h n d', h=h)
        # 使用 einsum 进行张量乘法，将 gate 和权重矩阵相乘，并加上偏置。
        # 这里的b h n d指的是gate，h m n指的是weight
        """
        具体操作：

        对每个批次 (b) 和每个头 (h)：
        gate 的一个切片形状是 (n, d)
        weight 的一个切片形状是 (m, n)

        执行矩阵乘法：(m, n) @ (n, d) -> (m, d)
        最终输出形状：(b, h, m, d)
        """
        gate = einsum('h n d, h m n -> h m d', gate, weight)
        gate = gate + rearrange(bias, 'h n -> h n ()')
        # 将 gate 恢复到原始的形状
        """
        输出形状 '() h n ()':

        (): 第一个空括号表示一个大小为1的新维度
        h: 保留原始的 h 维度
        n: 保留原始的 n 维度
        (): 最后的空括号表示另一个大小为1的新维度`

        """
        gate = rearrange(gate, 'h n d -> n (h d)')

        if exists(gate_res):
            gate = gate + gate_res

        return self.act(gate) * res

class gMLP_Bolck(nn.Module):
    """
    函数定义中的 * 是一个特殊的语法，用于表示之后的参数必须作为关键字参数
    （keyword-only arguments）来传递。这样可以防止函数调用时参数的顺
    序错误，提高代码的可读性和安全性。
    """
    def __init__(self,
                 *,
                 dim,
                 dim_ff,
                 seq_len,
                 heads=1,
                 attn_dim=None,
                 causal=False,
                 act=nn.Identity(),
                 circulant_matrix=False):
        super(gMLP_Bolck, self).__init__()

        """
        self.proj_in 定义了一个线性变换层和 GELU 激活函数的组合，
        将输入从 dim 投影到 dim_ff
        """

        self.proj_in = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.GELU()
        )

        self.attn = Attention(dim, dim_ff//2, attn_dim, causal) if exists(attn_dim) else None

        self.sgu = SpatialGatingUnit(dim_ff, seq_len, causal, act, heads, circulant_matrix=circulant_matrix)
        self.proj_out = nn.Linear(dim_ff//2, dim)

    def forward(self, x):
        gate_res = self.attn(x) if exists(self.attn) else None
        print("gate shape", gate_res.shape)
        print("before x shape", x.shape)
        x = self.proj_in(x)
        print("x.shape", x.shape)
        x = self.sgu(x, gate_res=gate_res)
        x = self.proj_out(x)
        return x


#
# class gMLP_Bolck_edge(nn.Module):
#     """
#     函数定义中的 * 是一个特殊的语法，用于表示之后的参数必须作为关键字参数
#     （keyword-only arguments）来传递。这样可以防止函数调用时参数的顺
#     序错误，提高代码的可读性和安全性。
#     """
#     def __init__(self,
#                  *,
#                  dim,
#                  dim_ff,
#                  seq_len,
#                  heads=1,
#                  attn_dim=None,
#                  causal=False,
#                  act=nn.Identity(),
#                  circulant_matrix=False):
#         super(gMLP_Bolck_edge, self).__init__()
#
#         """
#         self.proj_in 定义了一个线性变换层和 GELU 激活函数的组合，
#         将输入从 dim 投影到 dim_ff
#         """
#
#         self.proj_in = nn.Sequential(
#             nn.Linear(dim, dim_ff),
#             nn.GELU()
#         )
#
#         self.attn = Attention(dim, dim_ff//2, attn_dim, causal) if exists(attn_dim) else None
#
#         self.sgu = SpatialGatingUnit(dim_ff, seq_len, causal, act, heads, circulant_matrix=circulant_matrix)
#         self.proj_out = nn.Linear(dim_ff//2, dim_ff)
#
#     def forward(self, x):
#         gate_res = self.attn(x) if exists(self.attn) else None
#         x = self.proj_in(x)
#         x = self.sgu(x, gate_res=gate_res)
#         x = self.proj_out(x)
#         return x

### core class

class gate_MultiLayerPerceptron(nn.Module):
    def __init__(self,
                 *,
                 num_tokens=None,
                 dim,
                 # 层数
                 depth,
                 seq_len,
                 heads,
                 ff_mult=4,
                 attn_dim=None,
                 # 每层的存活概率
                 prob_survival=1.,
                 causal=False,
                 circulant_matrix=False,
                 shift_tokens=0,
                 act=nn.Identity()):
        super(gate_MultiLayerPerceptron, self).__init__()

        assert (dim % heads) == 0, 'dim must be divisible by number of heads'

        dim_ff = dim * ff_mult
        self.seq_len = seq_len
        self.prob_survival = prob_survival

        # self.to_embed = nn.Embedding(num_tokens, dim) if exists(num_tokens) else nn.Identity()

        token_shifts = tuple(range(0 if causal else -shift_tokens, shift_tokens+1))
        # 定义一个包含多个 gMLP 块的层列表 self.layers，
        # 每个块由 Residual（残差连接）、
        # PreNorm（归一化）和 PreShiftTokens（预平移）
        # 以及 gMLPBlock 组成。这个操作会重复 depth 次。
        self.layers = nn.ModuleList([
            Residual(
                PreNorm(dim,
                        PreShiftTokens(token_shifts,
                                       gMLP_Bolck(
                                           dim=dim,
                                           heads=heads,
                                           dim_ff=dim_ff,
                                           seq_len=seq_len,
                                           attn_dim=attn_dim,
                                           causal=causal,
                                           act=act,
                                           circulant_matrix=circulant_matrix))))
        for _ in range(depth)])



        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        ) if exists(num_tokens) else nn.Identity

    def forward(self, x):
        # x = self.to_embed(x)
        # self.training 是nn.Module的内置函数
        layers = self.layers if not self.training else dropout_layers(
            self.layers,
            self.prob_survival,
        )
        out = nn.Sequential(*layers)(x)
        return self.to_logits(out)



class gate_MultiLayerPerceptron_edge(nn.Module):
    def __init__(self,
                 *,
                 num_tokens=None,
                 dim,
                 dim_ff,
                 # 层数
                 depth,
                 seq_len,
                 heads=None,
                 # ff_mult=4,
                 attn_dim=None,
                 # 每层的存活概率
                 prob_survival=1.,
                 causal=False,
                 circulant_matrix=False,
                 shift_tokens=0,
                 act=nn.Identity()):
        super(gate_MultiLayerPerceptron_edge, self).__init__()

        assert (dim % heads) == 0, 'dim must be divisible by number of heads'


        # dim_ff = dim * ff_mult


        self.seq_len = seq_len
        self.prob_survival = prob_survival

        # self.to_embed = nn.Embedding(num_tokens, dim) if exists(num_tokens) else nn.Identity()

        token_shifts = tuple(range(0 if causal else -shift_tokens, shift_tokens+1))
        # 定义一个包含多个 gMLP 块的层列表 self.layers，
        # 每个块由 Residual（残差连接）、
        # PreNorm（归一化）和 PreShiftTokens（预平移）
        # 以及 gMLPBlock 组成。这个操作会重复 depth 次。
        self.layers = nn.ModuleList([
            Residual(
                PreNorm(dim,
                        PreShiftTokens(token_shifts,
                                       gMLP_Bolck(
                                           dim=dim,
                                           heads=heads,
                                           dim_ff=dim_ff,
                                           seq_len=seq_len,
                                           attn_dim=attn_dim,
                                           causal=causal,
                                           act=act,
                                           circulant_matrix=circulant_matrix))))
        for _ in range(depth)])

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        ) if exists(num_tokens) else nn.Identity

    def forward(self, x):
        # x = self.to_embed(x)
        # self.training 是nn.Module的内置函数
        layers = self.layers if not self.training else dropout_layers(
            self.layers,
            self.prob_survival,
        )
        out = nn.Sequential(*layers)(x)
        return self.to_logits(out)






#
# if __name__ == '__main__':
#     grad_global_dist_mlp = gate_MultiLayerPerceptron(num_tokens=1,
#                                                       dim=2*256,
#                                                       depth=2,
#                                                       attn_dim=64,
#                                                       seq_len=128,
#                                                       causal=True,
#                                                       circulant_matrix=True,
#                                                       heads=4,  # 2 heads
#                                                       act=nn.Tanh()
#                                                       )
#
#     grad_global_node_mlp = gate_MultiLayerPerceptron(num_tokens=6,
#                                                       dim=256,
#                                                       depth=2,
#                                                       attn_dim=64,
#                                                       seq_len=128,
#                                                       causal=True,
#                                                       circulant_matrix=True,
#                                                       heads=4,  # 2 heads
#                                                       act=nn.Tanh()
#                                                       )



    # node那个还是尽量使用MLP比较靠谱，因为edge_length都是1维的数据
    """
    [[1.5117],
        [2.4745],
        [3.1966],
        [4.3850],
        [2.6360],
        [3.1929],
        [4.4490],
        [3.9108]]类似这样的数据
        所以使用MLP足够了

    
    """
    # gmlp_edge = gate_MultiLayerPerceptron_edge(dim=1,
    #                                            dim_ff=256,
    #                                            depth=2,
    #                                            seq_len=128,
    #                                            causal=True,
    #                                            circulant_matrix=True,
    #                                            # heads=4,  # 2 heads
    #                                            act=nn.Tanh()
    #                                            )






