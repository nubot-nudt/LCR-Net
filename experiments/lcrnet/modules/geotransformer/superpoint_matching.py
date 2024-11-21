import torch
import torch.nn as nn

from experiments.lcrnet.modules.ops import pairwise_distance
from experiments.lcrnet.modules.sinkhorn import LearnableLogOptimalTransport

class SuperPointMatching(nn.Module):
    def __init__(self, num_correspondences, dual_normalization=True):
        super(SuperPointMatching, self).__init__()
        self.num_correspondences = num_correspondences
        self.dual_normalization = dual_normalization

    def forward(self, ref_feats, src_feats, ref_masks=None, src_masks=None):
        r"""Extract superpoint correspondences.

        Args:
            ref_feats (Tensor): features of the superpoints in reference point cloud.
            src_feats (Tensor): features of the superpoints in source point cloud.
            ref_masks (BoolTensor=None): masks of the superpoints in reference point cloud (False if empty).
            src_masks (BoolTensor=None): masks of the superpoints in source point cloud (False if empty).

        Returns:
            ref_corr_indices (LongTensor): indices of the corresponding superpoints in reference point cloud.
            src_corr_indices (LongTensor): indices of the corresponding superpoints in source point cloud.
            corr_scores (Tensor): scores of the correspondences.
        """
        if ref_masks is None:
            ref_masks = torch.ones(size=(ref_feats.shape[0],), dtype=torch.bool).cuda()
        if src_masks is None:
            src_masks = torch.ones(size=(src_feats.shape[0],), dtype=torch.bool).cuda()
        # remove empty patch
        ref_indices = torch.nonzero(ref_masks, as_tuple=True)[0]
        src_indices = torch.nonzero(src_masks, as_tuple=True)[0]
        ref_feats = ref_feats[ref_indices]
        src_feats = src_feats[src_indices]
        # select top-k proposals
        matching_scores = torch.exp(-pairwise_distance(ref_feats, src_feats, normalized=True))
        if self.dual_normalization:
            ref_matching_scores = matching_scores / matching_scores.sum(dim=1, keepdim=True)
            src_matching_scores = matching_scores / matching_scores.sum(dim=0, keepdim=True)
            matching_scores = ref_matching_scores * src_matching_scores
        num_correspondences = min(self.num_correspondences, matching_scores.numel())
        corr_scores, corr_indices = matching_scores.view(-1).topk(k=num_correspondences, largest=True)
        ref_sel_indices = corr_indices // matching_scores.shape[1]
        src_sel_indices = corr_indices % matching_scores.shape[1]
        # recover original indices
        ref_corr_indices = ref_indices[ref_sel_indices]
        src_corr_indices = src_indices[src_sel_indices]

        return ref_corr_indices, src_corr_indices, corr_scores


# sch
class SuperPointMatching_OT(nn.Module):
    def __init__(self, num_correspondences, num_iter=100):
        super(SuperPointMatching_OT, self).__init__()
        self.num_correspondences = num_correspondences
        # self.dual_normalization = dual_normalization

    def correspondences_from_thres(self, score, thres=0.0, supp=False, return_score=True):
        '''
        Return estimated rough matching regions from score matrix
        param: score: score matrix, [N, M]
        return: correspondences [K, 2]
        '''

        x = torch.arange(score.shape[0] - 1).cuda().unsqueeze(-1)
        x = x.repeat([1, score.shape[1] - 1])

        y = torch.arange(score.shape[1] - 1).cuda().unsqueeze(0)
        y = y.repeat([score.shape[0] - 1, 1])

        mask = score[:-1, :-1] > thres

        x, y = x[mask].unsqueeze(-1), y[mask].unsqueeze(-1)


        correspondences = torch.cat([x, y], dim=-1)

        if supp and correspondences.shape[0] == 0:
            cur_item = torch.zeros(size=(1, 2), dtype=torch.int32).cuda()
            cur_item[0, 0], cur_item[0, 1] = 0, 0
            correspondences = torch.cat([correspondences, cur_item], dim=0)
        if return_score:
            corr_score = score[correspondences[:, 0], correspondences[:, 1]]
            return correspondences, corr_score.view(-1)
        else:
            return correspondences


    def forward(self, matching_scores, ref_masks=None, src_masks=None, ref_n2p_scores_c=None, src_n2p_scores_c=None):
        r"""Extract superpoint correspondences.

        Args:
            ref_feats (Tensor): features of the superpoints in reference point cloud.
            src_feats (Tensor): features of the superpoints in source point cloud.
            ref_masks (BoolTensor=None): masks of the superpoints in reference point cloud (False if empty).
            src_masks (BoolTensor=None): masks of the superpoints in source point cloud (False if empty).

        Returns:
            ref_corr_indices (LongTensor): indices of the corresponding superpoints in reference point cloud.
            src_corr_indices (LongTensor): indices of the corresponding superpoints in source point cloud.
            corr_scores (Tensor): scores of the correspondences.
        """
       

        '''using dustbin thres(one side)'''
        # matching_scores = torch.exp(matching_scores)
        # src_topk_scores, src_topk_indices = matching_scores.topk(k=1, dim=0)  # (B, K, N)
        # src_indices = torch.arange(matching_scores.shape[1]).cuda().view(1, matching_scores.shape[1])  # (B, K, N)
        # src_score_mat = torch.zeros_like(matching_scores)
        # src_score_mat[src_topk_indices, src_indices] = src_topk_scores
        # src_corr_mat = torch.gt(src_score_mat, matching_scores[-1,:].unsqueeze(0))
        
        # src_corr_mat = src_corr_mat[:-1, :-1]  
        # try:
        #     corr_indices = src_corr_mat.nonzero()
        #     ref_corr_indices, src_corr_indices = corr_indices[:,0], corr_indices[:,1]
        #     corr_scores = matching_scores[ref_corr_indices, src_corr_indices]


        #     # ref_corr_indices = ref_indices[ref_corr_indices]
        #     # src_corr_indices = src_indices[src_corr_indices]
        # except:
        #     ref_corr_indices = torch.tensor([])
        #     src_corr_indices = torch.tensor([])
        #     corr_scores = torch.tensor([])

        if self.num_correspondences is None:
            '''using dustbin thres(both side)'''
            matching_scores = torch.exp(matching_scores)
            src_topk_scores, src_topk_indices = matching_scores.topk(k=1, dim=0)  # (B, K, N)
            src_indices = torch.arange(matching_scores.shape[1]).cuda().view(1, matching_scores.shape[1])  # (B, K, N)
            src_score_mat = torch.zeros_like(matching_scores)
            src_score_mat[src_topk_indices, src_indices] = src_topk_scores
            src_corr_mat = torch.gt(src_score_mat, matching_scores[-1,:].unsqueeze(0))

            ref_topk_scores, ref_topk_indices = matching_scores.topk(k=1, dim=1)  # (B, K, N)
            ref_indices = torch.arange(matching_scores.shape[0]).cuda().view(matching_scores.shape[0],1)  # (B, K, N)
            ref_score_mat = torch.zeros_like(matching_scores)
            ref_score_mat[ref_indices, ref_topk_indices] = ref_topk_scores
            ref_corr_mat = torch.gt(ref_score_mat, matching_scores[:,-1].unsqueeze(1))


            # corr_mat = torch.logical_and(ref_corr_mat, src_corr_mat)
            corr_mat = torch.logical_or(ref_corr_mat, src_corr_mat)
            
            corr_mat = corr_mat[:-1, :-1]
            try:
                corr_indices = corr_mat.nonzero()
                ref_corr_indices, src_corr_indices = corr_indices[:,0], corr_indices[:,1]
                corr_scores = matching_scores[ref_corr_indices, src_corr_indices]


                # ref_corr_indices = ref_indices[ref_corr_indices]
                # src_corr_indices = src_indices[src_corr_indices]
            except:
                ref_corr_indices = torch.tensor([])
                src_corr_indices = torch.tensor([])
                corr_scores = torch.tensor([])

        else:
            '''extract fix number of nodes'''
            matching_scores = torch.exp(matching_scores)
            if ref_n2p_scores_c is not None:
                ref_n2p_scores_c = ref_n2p_scores_c.unsqueeze(1)
                src_n2p_scores_c = src_n2p_scores_c.unsqueeze(0)
                n2p_scores_c = ref_n2p_scores_c*src_n2p_scores_c

                matching_scores[:-1,:-1] = matching_scores[:-1,:-1]*n2p_scores_c


            begin_thres = 0.5
            if matching_scores.shape[0]<self.num_correspondences or matching_scores.shape[1]<self.num_correspondences:
                if matching_scores.shape[0]<matching_scores.shape[1]:
                    self.num_correspondences=matching_scores.shape[0]
                else:
                    self.num_correspondences=matching_scores.shape[1]
            while True:
                corr, corr_scores = self.correspondences_from_thres(matching_scores, thres=begin_thres, supp=False, return_score=True)
                if corr.shape[0] >= self.num_correspondences:
                    break
                begin_thres -= 0.01
            ref_corr_indices = corr[:, 0].long()
            src_corr_indices = corr[:, 1].long()


        return ref_corr_indices, src_corr_indices, corr_scores