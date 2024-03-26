import torch
from torch import nn
import torch.nn.functional as F
import math
from torch import einsum
from einops import rearrange

from visdialch.dim.selfAttention import SelfAttention
from visdialch.dim.router import Router

from visdialch.encoders.net_utils import FFN, LayerNorm


class FILTER(nn.Module):
    def __init__(self, __C, num_out_path):
        super(FILTER, self).__init__()
        self.keep_mapping = nn.ReLU()
        self.router = Router(num_out_path, __C["embed_size"], __C["hid_router"])

    def forward(self, x):
        path_prob = self.router(x)
        emb = self.keep_mapping(x)

        return emb, path_prob


class INTRA(nn.Module):
    def __init__(self, __C, num_out_path):
        super(INTRA, self).__init__()
        self.__C = __C
        self.router = Router(num_out_path, __C["embed_size"], __C["hid_router"])
        self.selfatt = SelfAttention(__C["embed_size"], __C["hid_DIM"], __C["num_head_DIM"])

    def forward(self, inp):
        path_prob = self.router(inp)
        if inp.dim() == 4:
            n_img, n_stc, n_local, dim = inp.size()
            x = inp.view(-1, n_local, dim)
        else:
            x = inp

        selfatt_emb = self.selatt(x)
        if inp.dim() == 4:
            n_img, n_stc, n_local, dim = inp.size()
            selfatt_emb = selfatt_emb.view(n_img, n_stc, n_local, -1)
        return selfatt_emb, path_prob


class INTER(nn.Module):
    def __init__(self, __C, num_out_path):
        super(INTER, self).__init__()

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C['transformer_num_layers'])])
        self.dec_list = nn.ModuleList([SGA(__C) for _ in range(__C['transformer_num_layers'])])
        self.router = Router(num_out_path, __C["embed_size"], __C["hid_router"])

    def forward(self, x, y, x_mask, y_mask):
        for enc in self.enc_list:
            x = enc(x, x_mask)

        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)
        path_prob = self.router(y)
        return y, path_prob


class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C['hidden_size'], __C['hidden_size'])
        self.linear_k = nn.Linear(__C['hidden_size'], __C['hidden_size'])
        self.linear_q = nn.Linear(__C['hidden_size'], __C['hidden_size'])
        self.linear_merge = nn.Linear(__C['hidden_size'], __C['hidden_size'])

        self.dropout = nn.Dropout(__C['model_dropout'])

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C['multi_head'],
            self.__C['hidden_size_head']
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C['multi_head'],
            self.__C['hidden_size_head']
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C['multi_head'],
            self.__C['hidden_size_head']
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C['hidden_size']
        )

        atted = self.linear_merge(atted)
        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()

        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C['model_dropout'])
        self.norm1 = LayerNorm(__C['hidden_size'])

        self.dropout2 = nn.Dropout(__C['model_dropout'])
        self.norm2 = LayerNorm(__C['hidden_size'])
        self.attn = AttentionOnAttention(
            dim=__C['hidden_size'],
            heads=8
        )

    def forward(self, x):
        x = self.norm1(self.attn(x) + x)

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


class SGA(nn.Module):
    def __init__(self, __C):
        super(SGA, self).__init__()

        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C['model_dropout'])
        self.norm1 = LayerNorm(__C['hidden_size'])

        self.dropout2 = nn.Dropout(__C['model_dropout'])
        self.norm2 = LayerNorm(__C['hidden_size'])

        self.dropout3 = nn.Dropout(__C['model_dropout'])
        self.norm3 = LayerNorm(__C['hidden_size'])

        self.attn = AttentionOnAttention(
            dim=__C['hidden_size'],
            heads=8
        )

    def forward(self, x, y):

        x = self.norm1(self.attn(x) + x)

        x = self.norm2(self.attn(x, context=y) + x)

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class AttentionOnAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        aoa_dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)

        self.aoa = nn.Sequential(
            nn.Linear(2 * inner_dim, 2 * dim),
            nn.GLU(),
            nn.Dropout(aoa_dropout)
        )

    def forward(self, x, context = None):
        h = self.heads
        q_ = self.to_q(x)

        context = default(context, x)
        kv = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q_, *kv))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim = -1)
        attn = self.dropout(attn)

        attn_out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(attn_out, 'b h n d -> b n (h d)', h = h)

        heads = torch.cat((out, q_), dim = -1)
        out = self.aoa(heads)
        return out




