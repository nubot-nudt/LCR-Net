from copy import deepcopy
from numpy import zeros_like
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.lcrnet.modules.ops import apply_transform, pairwise_distance
from experiments.lcrnet.modules.registration.metrics import isotropic_transform_error


def best_pos_distance(query, pos_vecs):
    num_pos = pos_vecs.shape[0]
    query_copies = query.repeat(int(num_pos), 1)
    diff = ((pos_vecs - query_copies) ** 2).sum(1)

    min_pos, _ = diff.min(0)
    max_pos, _ = diff.max(0)
    mean_pos = diff.mean(0)
    return min_pos, max_pos, mean_pos

class SingleSideChamferLoss_Brute(nn.Module):
    def __init__(self):
        super(SingleSideChamferLoss_Brute, self).__init__()

    def forward(self, output_dict):
        '''
        :param pc_anc_input: Bx3xM Variable in GPU
        :param pc_dst_input: Bx3xN Variable in GPU
        :return:
        '''

        pos_node = output_dict['shifted_pos_points_c']
        anc_node = output_dict['shifted_anc_points_c']
        pos_points_f = output_dict['pos_points_f']
        anc_points_f = output_dict['anc_points_f']

        pos_on_pc_dsit_mat = torch.sqrt(pairwise_distance(pos_node, pos_points_f, normalized=False))
        pos_min_dist, _ = torch.min(pos_on_pc_dsit_mat, dim=1, keepdim=False)  # BxM

        anc_on_pc_dsit_mat = torch.sqrt(pairwise_distance(anc_node, anc_points_f, normalized=False))
        anc_min_dist, _ = torch.min(anc_on_pc_dsit_mat, dim=1, keepdim=False)  # BxM

        loss = (pos_min_dist.mean()+anc_min_dist.mean())/2

        return loss


class VoteLoss_new(nn.Module):
    def __init__(self, cfg):
        super(VoteLoss_new, self).__init__()
        self.BCE_loss = nn.BCELoss(reduction='none')

        self.NMS_radius = cfg.NMS_radius
    
    def get_weighted_bce_loss(self, prediction, gt):

        class_loss = self.BCE_loss(prediction, gt) 

        weights = torch.ones_like(gt)
        w_negative = gt.sum()/gt.size(0) 
        w_positive = 1 - w_negative  
        
        weights[gt >= 0.5] = w_positive
        weights[gt < 0.5] = w_negative
        w_class_loss = torch.mean(weights * class_loss)

        return w_class_loss

    def forward(self, output_dict, data_dict):

        pos_node = output_dict['shifted_pos_points_c']
        anc_node = output_dict['shifted_anc_points_c']
        transform = data_dict['transform']

        anc_node = apply_transform(anc_node, transform)
        dist_mat = torch.sqrt(pairwise_distance(pos_node, anc_node, normalized=False))

        ###################
        # chamfer loss
        mask = output_dict['mask']

        anc_dst_min_dist, _ = torch.min(dist_mat, dim=1, keepdim=False)  # BxM
        pos_mask = mask.sum(1)>0
        forward_loss = anc_dst_min_dist[pos_mask].mean()
        dst_anc_min_dist, _ = torch.min(dist_mat, dim=0, keepdim=False)  # BxN
        anc_mask = mask.sum(0)>0
        backward_loss = dst_anc_min_dist[anc_mask].mean()

        chamfer_pure = forward_loss + backward_loss


        return chamfer_pure  



class gap(nn.Module):
    '''all postive mean'''
    def __init__(self, cfg):
        super(gap, self).__init__()
        self.triplet_loss_gamma = cfg.distribution_loss.triplet_loss_gamma
        self.positive_radius = cfg.fine_loss.positive_radius

    def forward(self, output_dict, data_dict):
        pos_node_corr_knn_points = output_dict['pos_node_corr_knn_points']
        anc_node_corr_knn_points = output_dict['anc_node_corr_knn_points']
        pos_node_corr_knn_masks = output_dict['pos_node_corr_knn_masks']
        anc_node_corr_knn_masks = output_dict['anc_node_corr_knn_masks']
        matching_scores = output_dict['matching_scores']
        transform = data_dict['transform']

        b, n, m = matching_scores.size()

        anc_node_corr_knn_points = apply_transform(anc_node_corr_knn_points, transform)
        dists = pairwise_distance(pos_node_corr_knn_points, anc_node_corr_knn_points)  # (B, N, M)
        gt_masks = torch.logical_and(pos_node_corr_knn_masks.unsqueeze(2), anc_node_corr_knn_masks.unsqueeze(1))

        
        pos_mask = torch.zeros_like(matching_scores, dtype=bool)
        gt_corr_mask = torch.lt(dists, self.positive_radius ** 2)
        gt_corr_mask = torch.logical_and(gt_corr_mask, gt_masks)
        pos_mask[:,:-1,:-1] = gt_corr_mask
        pos_mask[:,:-1,-1] = torch.eq(gt_corr_mask.sum(2), 0)
        pos_mask[:,-1,:-1] = torch.eq(gt_corr_mask.sum(1), 0)

        neg_mask = torch.zeros_like(matching_scores, dtype=bool)
        gt_neg_mask = torch.gt(dists, (self.positive_radius*2) ** 2)
        neg_mask[:,:-1,:-1] = gt_neg_mask
        neg_mask[:,:-1,-1] = ~torch.eq(gt_corr_mask.sum(2), 0)
        neg_mask[:,-1,:-1] = ~torch.eq(gt_corr_mask.sum(1), 0)

        pos_matching_score = torch.zeros_like(matching_scores[:,:-1,:])
        pos_matching_score[pos_mask[:,:-1,:]]=-matching_scores[:,:-1,:][pos_mask[:,:-1,:]]
        pos_matching_score = pos_matching_score.sum(2)/pos_mask[:,:-1,:].sum(2)
        neg_matching_score = torch.ones_like(matching_scores[:,:-1,:])*1e12
        neg_matching_score[neg_mask[:,:-1,:]]=-matching_scores[:,:-1,:][neg_mask[:,:-1,:]]

        gap = (pos_matching_score.unsqueeze(2) - neg_matching_score)
        gap = gap[~(pos_matching_score==1e12)]
        gap_loss = torch.clamp(gap + self.triplet_loss_gamma, min=0)
        gap_loss = torch.mean(torch.log(torch.sum(gap_loss, dim=1)+1))


        pos_matching_score_inv = torch.zeros_like(matching_scores[:,:,:-1])
        pos_matching_score_inv[pos_mask[:,:,:-1]]=-matching_scores[:,:,:-1][pos_mask[:,:,:-1]]
        pos_matching_score_inv = pos_matching_score_inv.sum(1)/pos_mask[:,:,:-1].sum(1)
        neg_matching_score_inv = torch.ones_like(matching_scores[:,:,:-1])*1e12
        neg_matching_score_inv[neg_mask[:,:,:-1]]=-matching_scores[:,:,:-1][neg_mask[:,:,:-1]]

        gap2 = (pos_matching_score_inv.unsqueeze(1) - neg_matching_score_inv).transpose(1,2)
        gap2 = gap2[~(pos_matching_score_inv==1e12)]
        gap_loss2 = torch.clamp(gap2 + self.triplet_loss_gamma, min=0)
        gap_loss2 = torch.mean(torch.log(torch.sum(gap_loss2, dim=1)+1))


        '' 'add all the loss'''
        loss_mean = (gap_loss + gap_loss2)/2
        if torch.isnan(loss_mean):
            print('gap loss nan')
        return loss_mean



class node_gap(nn.Module):
    '''use all pos mean'''
    def __init__(self, cfg):
        super(node_gap, self).__init__()
        self.triplet_loss_gamma = cfg.distribution_loss.triplet_loss_gamma
        self.positive_radius = cfg.coarse_loss.positive_overlap
        # self.lamda = lamda

    def forward(self, output_dict):
        pos_node_masks = output_dict['pos_node_masks']
        anc_node_masks = output_dict['anc_node_masks']
        
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        matching_scores = output_dict['node_matching_scores']
        gt_pos_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_anc_node_corr_indices = gt_node_corr_indices[:, 1]

        overlaps = torch.zeros_like(matching_scores)[:-1,:-1]
        overlaps[gt_pos_node_corr_indices, gt_anc_node_corr_indices] = gt_node_corr_overlaps

        overlaps = overlaps.unsqueeze(0)
        matching_scores = matching_scores.unsqueeze(0)

        b, n, m = matching_scores.size()

        # anc_node_corr_knn_points = apply_transform(anc_node_corr_knn_points, transform)
        # dists = pairwise_distance(pos_node_corr_knn_points, anc_node_corr_knn_points)  # (B, N, M)
        gt_masks = torch.logical_and(pos_node_masks.unsqueeze(1), anc_node_masks.unsqueeze(0)).unsqueeze(0)

        pos_mask = torch.zeros_like(matching_scores, dtype=bool)
        gt_corr_mask = torch.gt(overlaps, self.positive_radius)
        gt_corr_mask = torch.logical_and(gt_corr_mask, gt_masks)
        pos_mask[:,:-1,:-1] = gt_corr_mask
        pos_mask[:,:-1,-1] = torch.eq(gt_corr_mask.sum(2), 0)
        pos_mask[:,-1,:-1] = torch.eq(gt_corr_mask.sum(1), 0)

        neg_mask = torch.zeros_like(matching_scores, dtype=bool)
        gt_neg_mask = torch.eq(overlaps, 0)
        neg_mask[:,:-1,:-1] = gt_neg_mask
        neg_mask[:,:-1,-1] = ~torch.eq(gt_corr_mask.sum(2), 0)
        neg_mask[:,-1,:-1] = ~torch.eq(gt_corr_mask.sum(1), 0)

        pos_matching_score = torch.zeros_like(matching_scores[:,:-1,:])
        pos_matching_score[pos_mask[:,:-1,:]]=-matching_scores[:,:-1,:][pos_mask[:,:-1,:]]
        pos_matching_score = pos_matching_score.sum(2)/pos_mask[:,:-1,:].sum(2)
        neg_matching_score = torch.ones_like(matching_scores[:,:-1,:])*1e12
        neg_matching_score[neg_mask[:,:-1,:]]=-matching_scores[:,:-1,:][neg_mask[:,:-1,:]]

        gap = (pos_matching_score.unsqueeze(2) - neg_matching_score)
        gap = gap[~(pos_matching_score==1e12)]
        gap_loss = torch.clamp(gap + self.triplet_loss_gamma, min=0)
        gap_loss = torch.mean(torch.log(torch.sum(gap_loss, dim=1)+1))


        pos_matching_score_inv = torch.zeros_like(matching_scores[:,:,:-1])
        pos_matching_score_inv[pos_mask[:,:,:-1]]=-matching_scores[:,:,:-1][pos_mask[:,:,:-1]]
        pos_matching_score_inv = pos_matching_score_inv.sum(1)/pos_mask[:,:,:-1].sum(1)
        neg_matching_score_inv = torch.ones_like(matching_scores[:,:,:-1])*1e12
        neg_matching_score_inv[neg_mask[:,:,:-1]]=-matching_scores[:,:,:-1][neg_mask[:,:,:-1]]

        gap2 = (pos_matching_score_inv.unsqueeze(1) - neg_matching_score_inv).transpose(1,2)
        gap2 = gap2[~(pos_matching_score_inv==1e12)]
        gap_loss2 = torch.clamp(gap2 + self.triplet_loss_gamma, min=0)
        gap_loss2 = torch.mean(torch.log(torch.sum(gap_loss2, dim=1)+1))

        '' 'add all the loss'''
        loss_mean = (gap_loss + gap_loss2)/2
        return loss_mean


class node_overlap_Loss(nn.Module):
    def __init__(self, cfg):
        super(node_overlap_Loss, self).__init__()
        # self.n2p_overlap_threshold = cfg.n2p_overlap_threshold
        self.BCE_loss = nn.BCELoss(reduction='none')
    
    def get_weighted_bce_loss(self, prediction, gt):

        class_loss = self.BCE_loss(prediction, gt) 

        weights = torch.ones_like(gt)
        w_negative = gt.sum()/gt.size(0)
        w_positive = 1 - w_negative  
        
        weights[gt >= 0.5] = w_positive
        weights[gt < 0.5] = w_negative
        w_class_loss = torch.mean(weights * class_loss)

        return w_class_loss

    def forward(self, output_dict):

        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        
        score = output_dict['score']
        pos_points_c = output_dict['pos_points_c']
        anc_points_c = output_dict['anc_points_c']

        device = score.device

       ###################
        # n2p overlap loss
        
        pos_gt = torch.zeros(pos_points_c.shape[0], device=device)
        pos_gt[gt_node_corr_indices[:,0]] = 1.
        
        anc_gt = torch.zeros(anc_points_c.shape[0], device=device)
        anc_gt[gt_node_corr_indices[:,1]] = 1.

        gt_labels = torch.cat((pos_gt, anc_gt))
        n_loss = self.get_weighted_bce_loss(score, gt_labels)

        return n_loss

class Evaluator(nn.Module):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        self.acceptance_overlap = cfg.eval.acceptance_overlap
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.rre_threshold = cfg.eval.rre_threshold
        self.rte_threshold = cfg.eval.rte_threshold

    @torch.no_grad()
    def evaluate_coarse(self, output_dict):
        pos_length_c = output_dict['pos_points_c'].shape[0]
        anc_length_c = output_dict['anc_points_c'].shape[0]
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        masks = torch.gt(gt_node_corr_overlaps, self.acceptance_overlap)
        gt_node_corr_indices = gt_node_corr_indices[masks]
        gt_pos_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_anc_node_corr_indices = gt_node_corr_indices[:, 1]
        gt_node_corr_map = torch.zeros(size=(pos_length_c, anc_length_c)).cuda()
        gt_node_corr_map[gt_pos_node_corr_indices, gt_anc_node_corr_indices] = 1.0

        pos_node_corr_indices = output_dict['pos_node_corr_indices']
        anc_node_corr_indices = output_dict['anc_node_corr_indices']

        precision = gt_node_corr_map[pos_node_corr_indices, anc_node_corr_indices].mean()

        return precision

    @torch.no_grad()
    def evaluate_fine(self, output_dict, data_dict):
        transform = data_dict['transform']
        pos_corr_points = output_dict['pos_corr_points']
        anc_corr_points = output_dict['anc_corr_points']
        anc_corr_points = apply_transform(anc_corr_points, transform)
        corr_distances = torch.linalg.norm(pos_corr_points - anc_corr_points, dim=1)
        precision = torch.lt(corr_distances, self.acceptance_radius).float().mean()
        return precision

    @torch.no_grad()
    def evaluate_registration(self, output_dict, data_dict):
        transform = data_dict['transform']
        est_transform = output_dict['estimated_transform']
        rre, rte = isotropic_transform_error(transform, est_transform)
        recall = torch.logical_and(torch.lt(rre, self.rre_threshold), torch.lt(rte, self.rte_threshold)).float()
        return rre, rte, recall

    def forward(self, output_dict, data_dict):
        result={}
        c_precision = self.evaluate_coarse(output_dict)
        result['PIR'] = c_precision
        f_precision = self.evaluate_fine(output_dict, data_dict)
        rre, rte, recall = self.evaluate_registration(output_dict, data_dict)
        result['IR'] = f_precision
        result['RRE'] = rre
        result['RTE'] = rte
        result['RR'] = recall
        return result
