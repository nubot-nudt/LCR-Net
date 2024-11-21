from numpy import zeros_like
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_metric_learning import distances

def best_pos_distance(query, pos_vecs):
    num_pos = pos_vecs.shape[1]
    query_copies = query.repeat(1, int(num_pos), 1)
    diff = ((pos_vecs - query_copies) ** 2).sum(2)

    min_pos, _ = diff.min(1)
    max_pos, _ = diff.max(1)
    mean_pos = diff.mean(1)
    return min_pos, max_pos, mean_pos


class Evaluator(nn.Module):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
   

    def forward(self, output_dict, data_dict):
        result={}

        return result
    
class TripletLoss(nn.Module):
    def __init__(self, margin: float):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output_dict):

        loss={}
        pos_global = output_dict['pos_global']
        anc_global = output_dict['anc_global']
        neg_global = output_dict['neg_global']

        min_pos, max_pos, mean_pos = best_pos_distance(anc_global, pos_global)

        positive = max_pos

        num_neg = neg_global.shape[1]
        num_pos = pos_global.shape[1]
        query_copies = anc_global.repeat(1, int(num_neg), 1)
        positive = positive.view(-1, 1)
        positive = positive.repeat(1, int(num_neg))

        negative = ((neg_global - query_copies) ** 2).sum(2)

        triplet_loss = self.margin + positive - negative
        triplet_loss = triplet_loss.clamp(min=0.0)

        loss['loss'] = triplet_loss.sum(1).mean()

        return loss