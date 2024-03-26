import torch
import torch.nn as nn

from visdialch.dim.blocks import FILTER, INTRA, INTER


def unsqueeze2d(x):
    return x.unsqueeze(-1).unsqueeze(-1)


class DynamicInteraction_Layer0(nn.Module):
    def __init__(self, __C, num_block, num_out_path):
        super(DynamicInteraction_Layer0, self).__init__()
        self.__C = __C
        self.threshold = 0.0001
        self.eps = 1e-8
        self.num_block = num_block
        self.num_out_path = num_out_path
        self.filter = FILTER(__C, num_out_path)
        self.intra= INTRA(__C, num_out_path)
        self.inter = INTER(__C, num_out_path)

    def forward(self, img_feat, language_feat,language_feat_mask,img_feat_mask):
        aggr_res_lst = self.findpath(img_feat, language_feat,language_feat_mask,img_feat_mask)
        return aggr_res_lst
         
    def findpath(self, img_feat, language_feat,language_feat_mask,img_feat_mask):
        path_prob = [None] * self.num_block
        emb_lst = [None] * self.num_block
        emb_lst[0], path_prob[0] = self.filter(img_feat)
        emb_lst[1], path_prob[1] = self.intra(img_feat)
        emb_lst[2], path_prob[2] = self.inter(language_feat,img_feat,language_feat_mask,img_feat_mask)

        y = sum(path_prob)
        gate_mask = (y < self.threshold).float()

        all_path_prob = torch.stack(path_prob, dim=2)
        all_path_prob = all_path_prob / (all_path_prob.sum(dim=-1, keepdim=True) + self.eps)
        path_prob = [all_path_prob[:, :, i] for i in range(all_path_prob.size(2))]

        aggr_res_lst = []
        for i in range(self.num_out_path):
            a = gate_mask[:, i]
            skip_emb = unsqueeze2d(a) * emb_lst[0]
            res = 0
            for j in range(self.num_block):
                cur_path = unsqueeze2d(path_prob[j][:, i])
                cur_emb = emb_lst[j]
                res = res + cur_path * cur_emb
            res = res + skip_emb
            aggr_res_lst.append(res)

        return aggr_res_lst, all_path_prob


class DynamicInteraction_Layer(nn.Module):
    def __init__(self, __C, num_block, num_out_path):
        super(DynamicInteraction_Layer, self).__init__()
        self.__C = __C
        self.threshold = 0.0001
        self.eps = 1e-8
        self.num_block = num_block
        self.num_out_path = num_out_path

        self.filter = FILTER(__C, num_out_path)
        self.intra = INTRA(__C, num_out_path)
        self.inter = INTER(__C, num_out_path)

    def forward(self, img_feat, language_feat, language_feat_mask, img_feat_mask):
        aggr_res_lst = self.findpath(img_feat, language_feat, language_feat_mask, img_feat_mask)
        return aggr_res_lst

    def findpath(self, img_feat, language_feat, language_feat_mask, img_feat_mask):
        path_prob = [None] * self.num_block
        emb_lst = [None] * self.num_block
        emb_lst[0], path_prob[0] = self.filter(img_feat[0])
        emb_lst[1], path_prob[1] = self.intra(img_feat[1])
        emb_lst[2], path_prob[2] = self.inter(language_feat, img_feat[2], language_feat_mask, img_feat_mask)

        if self.num_out_path == 1:
            aggr_res_lst = []
            gate_mask_lst = []
            res = 0
            for j in range(self.num_block):
                gate_mask = (path_prob[j] < self.threshold / self.num_block).float()
                gate_mask_lst.append(gate_mask)
                skip_emb = gate_mask.unsqueeze(-1) * img_feat[j]
                res += path_prob[j].unsqueeze(-1) * emb_lst[j]
                res += skip_emb

            res = res / (sum(gate_mask_lst) + sum(path_prob)).unsqueeze(-1)
            all_path_prob = torch.stack(path_prob, dim=2)
            aggr_res_lst.append(res)
        else:
            gate_mask = (sum(path_prob) < self.threshold).float()
            all_path_prob = torch.stack(path_prob, dim=2)
            all_path_prob = all_path_prob / (all_path_prob.sum(dim=-1, keepdim=True) + self.eps)
            path_prob = [all_path_prob[:, :, i] for i in range(all_path_prob.size(2))]
            aggr_res_lst = []
            for i in range(self.num_out_path):
                skip_emb = unsqueeze2d(gate_mask[:, i]) * emb_lst[0]
                res = 0
                for j in range(self.num_block):
                    cur_path = unsqueeze2d(path_prob[j][:, i])
                    res = res + cur_path * emb_lst[j]
                res = res + skip_emb
                aggr_res_lst.append(res)

        return aggr_res_lst, all_path_prob





