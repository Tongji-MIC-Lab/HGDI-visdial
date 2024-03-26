import torch
import torch.nn.functional as F

from torch import nn
from .structed import SANet
from .net_utils import FFN, MLP, LayerNorm
from visdialch.dim.blocks import SA


class NodeEmbeddingModule(nn.Module):
    def __init__(self, __C):
        super(NodeEmbeddingModule, self).__init__()
        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C['transformer_num_layers'])])

    def forward(self, x, x_mask):
        for enc in self.enc_list:
            x = enc(x, x_mask)
        return x


class StructedGraphModule(nn.Module):
    def __init__(self, __C):
        super(StructedGraphModule, self).__init__()
        self.sparse = SANet(__C)
        self.upm = nn.ModuleList([UP(__C) for _ in range(__C['update_num_layers'])])

    def forward(self, x, y):
        binary, w_att = self.sparse(x, y)
        return binary, w_att

    def update(self, x, y, adj):
        x = torch.cat((y, x), dim=1)
        for up in self.upm:
            x = up(x, adj)
        return x[:, -1:, :]


class UP(nn.Module):
    def __init__(self, __C):
        super(UP, self).__init__()
        self.dropout1 = nn.Dropout(__C['model_dropout'])
        self.dropout2 = nn.Dropout(__C['model_dropout'])
        self.norm1 = LayerNorm(__C['hidden_size'])
        self.norm2 = LayerNorm(__C['hidden_size'])
        self.ffn = FFN(__C)

    def forward(self, x, adj):
        output = torch.matmul(adj, x)
        x = self.norm1(x + self.dropout1(output))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x


class FuseGraph(nn.Module):
    def __init__(self, t_graph_node_size, spa_graph_node_size, fuse_graph_node_size):
        super(FuseGraph, self).__init__()
        self.gate_layer = nn.Linear(t_graph_node_size + spa_graph_node_size, t_graph_node_size + spa_graph_node_size)
        self.fuse_layer = nn.Linear(t_graph_node_size + spa_graph_node_size, fuse_graph_node_size)

    def forward(self, t_graph, spa_graph):
        cat = torch.cat((t_graph, spa_graph), dim=2)
        cat = self.gate_layer(cat)
        gate = torch.sigmoid(cat)
        fuse_graph = self.fuse_layer(gate * cat)
        return fuse_graph


class AttFlat(nn.Module):
    def __init__(self, in_channel, glimpses=1, dropout_r=0.1):
        super(AttFlat, self).__init__()
        self.glimpses = glimpses

        self.mlp = MLP(
            in_size=in_channel,
            mid_size=in_channel,
            out_size=glimpses,
            dropout_r=dropout_r,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            in_channel * glimpses,
            in_channel
        )
        self.norm = LayerNorm(in_channel)

    def forward(self, x, x_mask):
        att = self.mlp(x)

        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)
        x_atted = self.norm(x_atted)
        return x_atted

