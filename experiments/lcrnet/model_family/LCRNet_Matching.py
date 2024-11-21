'''vote encoder'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

from experiments.lcrnet.modules.ops import point_to_node_partition, index_select
from experiments.lcrnet.modules.registration import get_node_correspondences, get_node_correspondences_disance, get_node_overlap
from experiments.lcrnet.modules.sinkhorn import LearnableLogOptimalTransport
from experiments.lcrnet.modules.geotransformer import (
    # GeometricTransformer,
    SuperPointMatching, SuperPointMatching_OT,
    SuperPointTargetGenerator,
    LocalGlobalRegistration,
)

# from models.fa.GAT_plusplus import GeometricTransformer, APETransformer
from experiments.lcrnet.modules.thdroformer import ThDRoFormer_linear as ThDRoFormer
from experiments.lcrnet.backbone4 import KPEncoder, Vote_Encoder, KPDecoder
from utils.utils.visualization import vis_shifte_node, visualization


   
class LCRNet_Matching(nn.Module):
    def __init__(self, cfg):
        super(LCRNet_Matching, self).__init__()
        self.num_points_in_patch = cfg.model.num_points_in_patch
        self.matching_radius = cfg.model.ground_truth_matching_radius
        self.corres_radius = cfg.model.ground_truth_corres_radius
        self.n2p_score_threshold = cfg.model.n2p_score_threshold
        self.p2p_score_threshold = cfg.model.p2p_score_threshold



        # Keypoint Detection Module
        self.encoder = KPEncoder(
            cfg.backbone.input_dim,
            cfg.backbone.init_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.init_radius,
            cfg.backbone.init_sigma,
            cfg.backbone.group_norm
            )

        cfg.Vote.input_feats_dim  = 256
        self.vote_encoder = Vote_Encoder(
            cfg.backbone.input_dim,
            cfg.backbone.init_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.init_radius,
            cfg.backbone.init_sigma,
            cfg.backbone.group_norm,
            cfg.Vote,
            cfg.neighbor_limits
        )
            

        self.proj_node_overlap_score = nn.Linear(cfg.GAT.output_dim*2,1)
        self.sigmoid = nn.Sigmoid()
      

        self.transformer = ThDRoFormer(
            cfg.GAT.input_dim,
            cfg.GAT.output_dim,
            cfg.GAT.hidden_dim,
            cfg.GAT.num_heads,
            cfg.GAT.num_layers,
            cfg.GAT.k
        )

        # Dense Point Matching HEAD
        self.kpdecoder = KPDecoder(
            cfg.backbone.output_dim,
            cfg.backbone.init_dim,
            cfg.backbone.group_norm,
            cfg.neighbor_limits,
            cfg.backbone.init_radius,
        )


        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold
        )

        self.node_optimal_transport = LearnableLogOptimalTransport(cfg.model.num_sinkhorn_iterations)

        self.coarse_matching = SuperPointMatching_OT(
            cfg.coarse_matching.num_correspondences
        )

        self.fine_matching = LocalGlobalRegistration(
            cfg.fine_matching.topk,
            cfg.fine_matching.acceptance_radius,
            mutual=cfg.fine_matching.mutual,
            confidence_threshold=cfg.fine_matching.confidence_threshold,
            use_dustbin=cfg.fine_matching.use_dustbin,
            use_global_score=cfg.fine_matching.use_global_score,
            correspondence_threshold=cfg.fine_matching.correspondence_threshold,
            correspondence_limit=cfg.fine_matching.correspondence_limit,
            num_refinement_steps=cfg.fine_matching.num_refinement_steps,
        )
        
        self.optimal_transport = LearnableLogOptimalTransport(cfg.model.num_sinkhorn_iterations)


    def KeypointDetection(self, data_dict):
        # Downsample point clouds
        feats = data_dict['features'].detach()
        points_c = data_dict['points'][-1].detach()
        pos_length_c = data_dict['lengths'][-1][0].item()
        anc_length_c = data_dict['lengths'][-1][1].item()

        pos_points_c = points_c[:pos_length_c]
        anc_points_c = points_c[pos_length_c:anc_length_c+pos_length_c]


        

        # #########################################
        # # 1. KPVote Encoder
        feats_list = self.encoder(feats, data_dict)
        feats_c = feats_list[-1]

        pos_feats_c = feats_c[:pos_length_c]
        anc_feats_c = feats_c[pos_length_c:]


        # #########################################
        # # 2. transformer
        pos_feats_c, anc_feats_c, pos_emb, anc_emb = self.transformer(
            pos_points_c.unsqueeze(0),
            anc_points_c.unsqueeze(0),
            pos_feats_c.unsqueeze(0),
            anc_feats_c.unsqueeze(0),
            return_pos_emb=True
        )

        # 2.1 overlap score
        feats_c = torch.cat([pos_feats_c.squeeze(0), anc_feats_c.squeeze(0)], dim=0)
        feats_list[-1] = feats_c


        vote_dict = self.vote_encoder(feats_c, data_dict)
        feats_c = vote_dict['feats_c']

        score = torch.clamp(self.sigmoid(self.proj_node_overlap_score(feats_c).view(-1)),min=0,max=1)
        vote_dict['score'] = score


        return vote_dict, feats_list, pos_emb, anc_emb
    
    def DenseMatchingHEAD(self, data_dict, feats_list, vote_dict):
        output_dict={}

        transform = data_dict['transform'].detach()
        points_f = data_dict['points'][0].detach()
        pos_length_f = data_dict['lengths'][0][0].item()
        anc_length_f = data_dict['lengths'][0][1].item()
        pos_points_f = points_f[:pos_length_f]
        anc_points_f = points_f[pos_length_f:anc_length_f+pos_length_f]

        anc_points_c = vote_dict['anc_points_c'] 
        pos_points_c = vote_dict['pos_points_c']
        anc_feats_c = vote_dict['anc_feats_c']
        pos_feats_c = vote_dict['pos_feats_c']


        # #########################################
        # # 1. Sparse Matching
        
        # 1.1 Grouping nearby points around the nodes (voted keypoints)
        _, pos_node_masks, pos_node_knn_indices, pos_node_knn_masks = point_to_node_partition(
            pos_points_f, pos_points_c, self.num_points_in_patch
        )
        _, anc_node_masks, anc_node_knn_indices, anc_node_knn_masks = point_to_node_partition(
            anc_points_f, anc_points_c, self.num_points_in_patch
        )

        output_dict['anc_node_knn_indices']=anc_node_knn_indices,
        output_dict['anc_node_knn_masks']=anc_node_knn_masks,
        output_dict['pos_node_knn_indices']=pos_node_knn_indices,
        output_dict['pos_node_knn_masks']=pos_node_knn_masks,

        pos_padded_points_f = torch.cat([pos_points_f, torch.zeros_like(pos_points_f[:1])], dim=0)
        anc_padded_points_f = torch.cat([anc_points_f, torch.zeros_like(anc_points_f[:1])], dim=0)
        pos_node_knn_points = index_select(pos_padded_points_f, pos_node_knn_indices, dim=0)
        anc_node_knn_points = index_select(anc_padded_points_f, anc_node_knn_indices, dim=0)

        # 1.2 Generate ground truth node correspondences
        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
            pos_points_c,
            anc_points_c,
            pos_node_knn_points,
            anc_node_knn_points,
            transform,
            self.matching_radius,
            ref_masks=pos_node_masks,
            src_masks=anc_node_masks,
            ref_knn_masks=pos_node_knn_masks,
            src_knn_masks=anc_node_knn_masks,
        )

        output_dict['gt_node_corr_indices'] = gt_node_corr_indices
        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps

        # 1.3 Sparse patch matching using optimal transport
        node_matching_scores = torch.einsum('bnd,bmd->bnm', pos_feats_c.unsqueeze(0), anc_feats_c.unsqueeze(0))  # (P, K, K)
        node_matching_scores = node_matching_scores / pos_feats_c.shape[1] ** 0.5
        node_matching_scores = self.node_optimal_transport(node_matching_scores, pos_node_masks.unsqueeze(0), anc_node_masks.unsqueeze(0))
        node_matching_scores = node_matching_scores.squeeze(0)

        output_dict['node_matching_scores'] = node_matching_scores
        output_dict['pos_node_masks'] = pos_node_masks
        output_dict['anc_node_masks'] = anc_node_masks


        
        # 1.4 Select topk nearest node correspondences
        with torch.no_grad():
            # 1.4.1 Random select ground truth node correspondences during training
            if self.training:
                pos_node_corr_indices, anc_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps
                )
            else:
                pos_node_corr_indices, anc_node_corr_indices, node_corr_scores = self.coarse_matching(
                    # node_matching_scores, pos_node_masks, anc_node_masks, pos_scores_c, anc_scores_c
                    node_matching_scores, pos_node_masks, anc_node_masks
                )

                output_dict['pos_node_corr_indices'] = pos_node_corr_indices
                output_dict['anc_node_corr_indices'] = anc_node_corr_indices



        # #########################################
        # # 2. KPDecoder
        feats_list = self.kpdecoder(feats_list, data_dict)
        feats_f = feats_list[0]


        pos_feats_f = feats_f[:pos_length_f]
        anc_feats_f = feats_f[pos_length_f:]
        output_dict['pos_feats_f'] = pos_feats_f
        output_dict['anc_feats_f'] = anc_feats_f


        # #########################################
        # # 3. Dense Point Matching

        # 3.1 Generate batched node points & feats
        pos_node_corr_knn_indices = pos_node_knn_indices[pos_node_corr_indices]  # (P, K)
        anc_node_corr_knn_indices = anc_node_knn_indices[anc_node_corr_indices]  # (P, K)
        pos_node_corr_knn_masks = pos_node_knn_masks[pos_node_corr_indices]  # (P, K)
        anc_node_corr_knn_masks = anc_node_knn_masks[anc_node_corr_indices]  # (P, K)
        pos_node_corr_knn_points = pos_node_knn_points[pos_node_corr_indices]  # (P, K, 3)
        anc_node_corr_knn_points = anc_node_knn_points[anc_node_corr_indices]  # (P, K, 3)

        pos_padded_feats_f = torch.cat([pos_feats_f, torch.zeros_like(pos_feats_f[:1])], dim=0)
        anc_padded_feats_f = torch.cat([anc_feats_f, torch.zeros_like(anc_feats_f[:1])], dim=0)
        pos_node_corr_knn_feats = index_select(pos_padded_feats_f, pos_node_corr_knn_indices, dim=0)  # (P, K, C)
        anc_node_corr_knn_feats = index_select(anc_padded_feats_f, anc_node_corr_knn_indices, dim=0)  # (P, K, C)

        output_dict['pos_node_corr_knn_points'] = pos_node_corr_knn_points
        output_dict['anc_node_corr_knn_points'] = anc_node_corr_knn_points
        output_dict['pos_node_corr_knn_masks'] = pos_node_corr_knn_masks
        output_dict['anc_node_corr_knn_masks'] = anc_node_corr_knn_masks

        # 3.2 Optimal transport
        matching_scores = torch.einsum('bnd,bmd->bnm', pos_node_corr_knn_feats, anc_node_corr_knn_feats)  # (P, K, K)
        matching_scores = matching_scores / feats_f.shape[1] ** 0.5
        matching_scores = self.optimal_transport(matching_scores, pos_node_corr_knn_masks, anc_node_corr_knn_masks)

        output_dict['matching_scores'] = matching_scores

        # 3.3 Generate final correspondences during testing
        if not self.training:
            with torch.no_grad():
                if not self.fine_matching.use_dustbin:
                    matching_scores = matching_scores[:, :-1, :-1]

                pos_corr_points, anc_corr_points, corr_scores, estimated_transform = self.fine_matching(
                    pos_node_corr_knn_points,
                    anc_node_corr_knn_points,
                    pos_node_corr_knn_masks,
                    anc_node_corr_knn_masks,
                    matching_scores,
                    node_corr_scores,
                )

                output_dict['pos_corr_points'] = pos_corr_points
                output_dict['anc_corr_points'] = anc_corr_points
                output_dict['corr_scores'] = corr_scores
                output_dict['estimated_transform'] = estimated_transform

        return output_dict
    
    def forward(self, data_dict):
        output_dict= {}

        # Downsample point clouds
        transform = data_dict['transform'].detach()
        pos_length_c = data_dict['lengths'][-1][0].item()
        pos_length_f = data_dict['lengths'][0][0].item()
        anc_length_c = data_dict['lengths'][-1][1].item()
        anc_length_f = data_dict['lengths'][0][1].item()
        points_c = data_dict['points'][-1].detach()
        points_f = data_dict['points'][0].detach()

        pos_points_c = points_c[:pos_length_c]
        anc_points_c = points_c[pos_length_c:anc_length_c+pos_length_c]
        pos_points_f = points_f[:pos_length_f]
        anc_points_f = points_f[pos_length_f:anc_length_f+pos_length_f]

        output_dict['ori_pos_points_c'] = pos_points_c
        output_dict['ori_anc_points_c'] = anc_points_c
        output_dict['pos_points_f'] = pos_points_f
        output_dict['anc_points_f'] = anc_points_f

        if self.training:
            # for voteloss calculation
            mask = get_node_correspondences_disance(
                pos_points_c,
                anc_points_c,
                transform,
                self.corres_radius
            )
            output_dict['mask'] = mask
        
        vote_dict, feats_list, pos_emb, anc_emb = self.KeypointDetection(data_dict)
        output_dict['pos_emb'] = pos_emb
        output_dict['anc_emb'] = anc_emb

        # vis_shifte_node(output_dict['ori_anc_points_c'], vote_dict['shifted_anc_points_c'], output_dict['anc_points_f'], centers=vote_dict['anc_points_c'],
        #     color=[1,0.5,0],
        #     src_node_color = [1, 0, 0],
        #     src_point_color = [0.1, 0.6, 0.1],
        #     )

        match_dict = self.DenseMatchingHEAD(data_dict, feats_list, vote_dict)
        output_dict = {**output_dict, **vote_dict, **match_dict}


        # visualization(
        #     output_dict,
        #     transform,
        #     src_node_color = [0.2, 1, 0.2],
        #     ref_node_color = [0.2, 0.2, 1],
        #     src_point_color = [0.1, 0.6, 0.1],
        #     ref_point_color = [0.1, 0.1, 0.6],
        #     offsets=(0, 0, -30),
        #     find_true=True
        # )
        

        return output_dict
    
from experiments.lcrnet.loss_reg import node_gap, VoteLoss_new, SingleSideChamferLoss_Brute, node_overlap_Loss, gap
class OverallLoss_new(nn.Module):
    def __init__(self, cfg):
        super(OverallLoss_new, self).__init__()
        self.coarse_loss = node_gap(cfg)
        self.distribution = gap(cfg)

        self.vote_loss = VoteLoss_new(cfg.Vote)
        self.node_on_pc_loss = SingleSideChamferLoss_Brute()
        self.node_overlap_loss = node_overlap_Loss(cfg)
            
        self.weight_coarse_loss = cfg.loss.weight_coarse_loss
        self.weight_vote_loss = cfg.loss.weight_vote_loss
        self.weight_gap_loss = cfg.loss.weight_gap_loss


    def forward(self, output_dict, data_dict):
        loss_all={}
        loss=0
        coarse_loss = self.coarse_loss(output_dict)
        gap_loss = self.distribution(output_dict, data_dict)

        loss = self.weight_coarse_loss * coarse_loss + 5 * gap_loss
        loss_all['c_loss'] = self.weight_coarse_loss * coarse_loss
        loss_all['g_loss'] = 5 * gap_loss

        if torch.isnan(gap_loss):
            print('gap loss nan')
            pos_length_c = data_dict['lengths'][-1][0].item()
            pos_length_f = data_dict['lengths'][0][0].item()
            anc_length_c = data_dict['lengths'][-1][1].item()
            anc_length_f = data_dict['lengths'][0][1].item()
            print('pos ',data_dict['anc_idx'],' ', pos_length_c, ' ', pos_length_f)
            print('anc ',data_dict['pos_idx'],' ', anc_length_c, ' ', anc_length_f)


        regular_loss = (torch.mean(torch.clamp(abs(output_dict['pos_emb'])-3.1415926,0)) + torch.mean(torch.clamp(abs(output_dict['anc_emb'])-3.1415926,0)))/2
        loss_all['reg_loss'] =  regular_loss
        loss = loss + regular_loss


        vote_loss = self.vote_loss(output_dict, data_dict)
        node_on_pc_loss = self.node_on_pc_loss(output_dict)
        node_overlap_loss = self.node_overlap_loss(output_dict)

        loss = loss + (vote_loss + node_on_pc_loss)*self.weight_vote_loss + node_overlap_loss
        # loss = loss + (vote_loss + node_on_pc_loss)*self.weight_vote_loss

        loss_all['v_loss'] = vote_loss * self.weight_vote_loss
        loss_all['d_loss'] = node_on_pc_loss * self.weight_vote_loss
        loss_all['n_loss'] = node_overlap_loss
        # loss_all['nf_loss'] = node_feat_loss

        loss_all['loss'] = loss

        return loss_all

def create_model(cfg):
    model = LCRNet_Matching(cfg)
    return model


def main():
    from experiments.lcrnet.config_reg import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == '__main__':
    main()
