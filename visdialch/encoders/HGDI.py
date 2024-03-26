import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from .module import NodeEmbeddingModule, StructedGraphModule, UP, FuseGraph
from .net_utils import MLP, LayerNorm

from visdialch.dim import DynamicInteraction_Layer0, DynamicInteraction_Layer


class HGDI(nn.Module):
    def __init__(self, __C, vocabulary):
        super().__init__()
        self.__C = __C

        self.word_embed = nn.Embedding(
            num_embeddings=len(vocabulary),
            embedding_dim=__C["word_embedding_size"],
            padding_idx=vocabulary.PAD_INDEX
        )
        self.h_rnn = nn.LSTM(
            input_size=__C["word_embedding_size"],
            hidden_size=__C["hidden_size"],
            num_layers=__C["lstm_num_layers"],
            batch_first=True,
        )
        self.q_rnn = nn.LSTM(
            input_size=__C["word_embedding_size"],
            hidden_size=__C["hidden_size"],
            num_layers=__C["lstm_num_layers"],
            batch_first=True,
        )
        self.v_proj = nn.Linear(
            __C["img_feature_size"],
            __C["hidden_size"]
        )
        self.j_proj = nn.Linear(
            __C["hidden_size"],
            __C["hidden_size"]
        )

        self.q_norm = LayerNorm(__C['flat_out_size'])
        self.h_norm = LayerNorm(__C['flat_out_size'])

        self.q_attflat_lang = AttFlat(__C)
        self.q_attflat_img = AttFlat(__C)

        self.h_attflat_lang = AttFlat(__C)
        self.h_attflat_img = AttFlat(__C)

        self.q_nem = NodeEmbeddingModule(__C)
        self.h_nem = NodeEmbeddingModule(__C)
        self.sgm = StructedGraphModule(__C)

        num_blocks = 3
        num_layer_routing = 3
        self.dynamic_itr_l0 = DynamicInteraction_Layer0(__C, num_blocks, num_blocks)
        self.dynamic_itr_l1 = DynamicInteraction_Layer(__C, num_blocks, num_blocks)
        self.dynamic_itr_l2 = DynamicInteraction_Layer(__C, num_blocks, 1)
        total_paths = num_blocks ** 2 * (num_layer_routing - 1) + num_blocks
        path_hid = 128
        self.path_mapping = nn.Linear(total_paths, path_hid)

        self.upm = nn.ModuleList([UP(__C) for _ in range(__C['sgl_update_num_layers'])])
        self.fuse_graph = FuseGraph(__C["hidden_size"], __C["hidden_size"], __C["hidden_size"])

    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)

    def lang_emb(self, seq, lang_type='ques'):
        rnn_cls = None
        if lang_type == 'hist':
            rnn_cls = self.h_rnn
        if lang_type == 'ques':
            rnn_cls = self.q_rnn

        lang_feat_mask = self.make_mask(seq.unsqueeze(2))
        lang_feat = self.word_embed(seq)

        rnn_cls.flatten_parameters()
        lang_feat, _ = rnn_cls(lang_feat)
        return lang_feat, lang_feat_mask

    def forward(self, batch):
        q = batch["ques"]
        h = batch["hist"]
        v = batch["img_feat"]
        cap, _ = self.lang_emb(h[:, 0, :], 'hist')

        img_feat = self.v_proj(v)
        img_feat_mask = self.make_mask(img_feat)

        n_batch, n_round, _ = q.size()
        enc_outs = []
        edge_strct = []
        weighted_strct = Variable(torch.zeros(n_batch, 1, n_round + 1).cuda())
        h_embs = None

        for i in range(n_round):
            ques_feat, ques_feat_mask = self.lang_emb(q[:, i, :], 'ques')
            hist_feat, hist_feat_mask = self.lang_emb(h[:, i, :], 'hist')

            pairs_emb_lst_q, paths_l0_q = self.dynamic_itr_l0(img_feat, ques_feat, ques_feat_mask, img_feat_mask)
            pairs_emb_lst_q, paths_l1_q = self.dynamic_itr_l1(pairs_emb_lst_q, ques_feat, ques_feat_mask, img_feat_mask)
            pairs_emb_lst_q, paths_l2_q = self.dynamic_itr_l2(pairs_emb_lst_q, ques_feat, ques_feat_mask, img_feat_mask)

            pairs_emb_lst_h, paths_l0_h = self.dynamic_itr_l0(img_feat, hist_feat, hist_feat_mask, img_feat_mask)
            pairs_emb_lst_h, paths_l1_h = self.dynamic_itr_l1(pairs_emb_lst_h, hist_feat, hist_feat_mask, img_feat_mask)
            pairs_emb_lst_h, paths_l2_h = self.dynamic_itr_l2(pairs_emb_lst_h, hist_feat, hist_feat_mask, img_feat_mask)

            q_emb = self.q_nem(
                ques_feat,
                ques_feat_mask,
            )
            qi_emb = pairs_emb_lst_q[0]
            q_emb = self.q_attflat_lang(
                q_emb,
                ques_feat_mask
            )
            qi_emb = self.q_attflat_img(
                qi_emb,
                img_feat_mask
            )
            q_emb = self.q_norm(q_emb + qi_emb)

            h_emb = self.h_nem(
                hist_feat,
                hist_feat_mask,
            )
            hi_emb = pairs_emb_lst_h[0]
            h_emb = self.h_attflat_lang(
                h_emb,
                hist_feat_mask
            )
            hi_emb = self.h_attflat_img(
                hi_emb,
                img_feat_mask
            )
            h_emb = self.h_norm(h_emb + hi_emb)

            h_emb = h_emb.unsqueeze(1)
            q_emb = q_emb.unsqueeze(1)

            if i == 0:
                h_embs = h_emb
            else:
                h_embs = torch.cat((h_embs, h_emb), dim=1)

            edge, w_att = self.sgm(q_emb, h_embs)

            b_pad = Variable(torch.zeros(n_batch, 1, n_round - (i + 1)).cuda())
            edge = torch.cat((edge, b_pad), dim=2)
            edge_strct.append(edge)

            w_pad = Variable(torch.zeros(n_batch, 1, n_round - i).cuda())
            w_att = torch.cat((w_att, w_pad), dim=2)
            weighted_strct = torch.cat((weighted_strct, w_att), dim=1)

            h_emb_last = h_embs[:, -1:, :]
            z = q_emb + h_emb_last
            enc_outs.append(z)

        _, n_stc = paths_l2_q.size()[:2]

        paths_l0 = paths_l0_q.contiguous().view(n_batch, -1).unsqueeze(1).expand(-1, n_stc, -1)
        paths_l1 = paths_l1_q.view(n_batch, n_stc, -1)
        paths_l2 = paths_l2_q.view(n_batch, n_stc, -1)
        paths = torch.cat([paths_l0, paths_l1, paths_l2], dim=-1)
        paths = paths.mean(dim=1)

        paths = self.path_mapping(paths)
        paths = F.normalize(paths, dim=-1)
        sim_paths = paths.matmul(paths.t())

        caption = F.normalize(cap.mean(dim=1) , dim=-1)
        sim_label = caption.matmul(caption.t())

        edge_strct = torch.cat(edge_strct, dim=1)
        graph_out = torch.cat(enc_outs, dim=1)

        a = torch.triu(torch.ones(n_batch, 10, 10), diagonal=1)*torch.tril(torch.ones(n_batch, 10, 10), diagonal=1)
        b = torch.triu(torch.ones(n_batch, 10, 10), diagonal=-1)*torch.tril(torch.ones(n_batch, 10, 10), diagonal=-1)
        t_adj = a + b
        for up in self.upm:
            t_graph_out = up(graph_out, t_adj.cuda())
        t_graph_out = self.j_proj(t_graph_out)
        t_graph = torch.tanh(t_graph_out)

        s_graph_out = self.j_proj(graph_out)
        s_graph = torch.tanh(s_graph_out)

        fusegraph_out = self.fuse_graph(t_graph, s_graph)
        return fusegraph_out, edge_strct, sim_paths, sim_label


class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C['hidden_size'],
            mid_size=__C['flat_mlp'],
            out_size=__C['flat_glimpses'],
            dropout_r=__C['model_dropout'],
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C['hidden_size'] * __C['flat_glimpses'],
            __C['flat_out_size']
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C['flat_glimpses']):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )
        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted

