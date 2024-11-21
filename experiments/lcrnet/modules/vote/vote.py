import torch
from torch import nn

from experiments.lcrnet.modules.ops import radius_search

class NMS(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.NMS_radius = cfgs.NMS_radius
        self.pdis = nn.PairwiseDistance(p=2)

    @torch.no_grad()
    def forward(self, nodes_dict, length_dict=None, overlap_score=None, features=None):
        '''
        :param nodes: list of node sets, each of shape Mx3
        :param overlap_score: M or None
        :param features: Mx256 or None
        :return: stacks of masks for each node set
        '''
        # if self.NMS_radius < 0.01:
        #     return keypoints_np, sigmas_np

        if length_dict is None:
            length_dict=[nodes_dict.shape[0]]

        length_increment = 0
        masks = []
        nms_length =[]
        for i in range(len(length_dict)):
            try:
                length = length_increment + length_dict[i].item()
            except:
                length = length_increment + length_dict[i]
            nodes = nodes_dict[length_increment:length]
            length_increment = length

            valid_node_counter = 0
            mask = torch.zeros(nodes.shape[0], dtype=torch.bool).cuda()  # (M,)
            mask[0] = True

            if overlap_score is not None:
                idx_sort = torch.argsort(overlap_score, descending=True)
                nodes = nodes[idx_sort,:]
                overlap_score = overlap_score[idx_sort]
                if features is not None:
                    features = features[idx_sort,:]

            for idx in range(1, nodes.shape[0]):
                # dis = torch.sqrt(pairwise_distance(nodes[idx], valid_node[:valid_node_counter+1]))
                # dis = torch.sqrt(pairwise_distance(sorted_nodes[idx], sorted_nodes[masks]))
                dis = self.pdis(nodes[idx].unsqueeze(0), nodes[mask])
                if (dis>self.NMS_radius).sum() == valid_node_counter+1:
                    mask[idx] = True
                    valid_node_counter+=1
                    
            if overlap_score is not None:
                mask_before_sorted = torch.zeros(nodes.shape[0], dtype=torch.bool).cuda()  # (M,)
                mask_before_sorted[idx_sort] = mask
                mask = mask_before_sorted
            # @BUG: 直接return features[mask]会使得前面的layernorm和linear garidient 为None,而无法 distribute train
            # 因为前面torch.no_grad()了

            
            masks.append(mask)
            nms_length.append(torch.tensor([mask.sum()]))
        
        if length_dict is None: 
            return torch.cat(masks, dim=0)
        else:
            return torch.cat(masks, dim=0), torch.cat(nms_length, dim=0)

class NMS_cpu(nn.Module):
    def __init__(self, cfgs, neighbor_limits):
        super().__init__()
        self.NMS_radius = cfgs.NMS_radius
        # self.pdis = nn.PairwiseDistance(p=2)
        self.neighbor_limits = neighbor_limits[-1]

    @torch.no_grad()
    def forward(self, nodes_dict, length_dict=None, overlap_score=None, features=None):
        '''
        :param nodes: list of node sets, each of shape Mx3
        :param overlap_score: M or None
        :param features: Mx256 or None
        :return: stacks of masks for each node set
        '''
        # if self.NMS_radius < 0.01:
        #     return keypoints_np, sigmas_np


        # msk,msl_length = radius_filter(nodes_dict,length_dict,self.NMS_radius)
        if length_dict is None:
            length_dict=[nodes_dict.shape[0]]
        node_knn_indices = radius_search(
            nodes_dict.cpu(),
            nodes_dict.cpu(),
            length_dict.cpu(),
            length_dict.cpu(),
            self.NMS_radius,
            self.neighbor_limits,
        )
        node_knn_indices = node_knn_indices.cuda()
        length_increment = 0
        masks = torch.zeros(node_knn_indices.shape[0]+1, dtype=torch.bool).cuda()
        nms_length =[]
        for i in range(node_knn_indices.shape[0]):
            if masks[node_knn_indices[i]].sum()==0:
                masks[i]=True

        return masks[:-1], node_knn_indices[masks[:-1]]

class Vote_layer(nn.Module):
    """ Light voting module with limitation"""
    def __init__(self, cfgs, r, output_feats=True):
        super(Vote_layer, self).__init__()

        # mlp_list = cfgs.MLPS2
        max_translate_range = cfgs.MAX_TRANSLATE_RANGE
        pre_channel = cfgs.input_feats_dim
        mlp_list=[cfgs.input_feats_dim*2,cfgs.input_feats_dim]

        if len(mlp_list) > 0:
            shared_mlps = []
            for i in range(len(mlp_list)):
                
                shared_mlps.extend([
                    nn.Linear(pre_channel, mlp_list[i]),
                    nn.LayerNorm(mlp_list[i]),
                    nn.ReLU()
                ])
                pre_channel = mlp_list[i]
            self.mlp_modules = nn.Sequential(*shared_mlps)
        else:
            self.mlp_modules = None

        self.output_feats = output_feats
        if output_feats:
            self.norm=nn.LayerNorm(cfgs.input_feats_dim)
            self.ctr_reg = nn.Linear(pre_channel, 3+cfgs.input_feats_dim)
        else:
            self.ctr_reg = nn.Linear(pre_channel, 3)

        # self.max_offset_limit = torch.tensor(max_translate_range).float()/r if max_translate_range is not None else None


        self.max_offset_limit = torch.tensor(max_translate_range) if max_translate_range is not None else None
 

    def forward(self, xyz, features, aug_rotation=None):
        if len(xyz.shape)<3:
            xyz = xyz.unsqueeze(0)
        if len(features.shape)<3:
            features = features.unsqueeze(0)

        xyz_select = xyz
        features_select = features

        if self.mlp_modules is not None: 
            new_features = self.mlp_modules(features_select) #([4, 256, 256]) ->([4, 128, 256])            
        else:
            new_features = features_select
        
        ctr_offsets = self.ctr_reg(new_features) #[4, 128, 256]) -> ([4, 3, 256])

        ctr_offsets = ctr_offsets#([4, 256, 3+input_feats_dim])
        feat_offets = ctr_offsets[..., 3:]
        ctr_offsets = ctr_offsets[..., :3]
        
        if self.max_offset_limit is not None:

            offset_dis = torch.norm(ctr_offsets,p=2,dim=2)
            alpha = torch.where(offset_dis > self.max_offset_limit, self.max_offset_limit/offset_dis, torch.tensor(1.0).cuda())
            limited_ctr_offsets = ctr_offsets*alpha.unsqueeze(2)
            vote_xyz = xyz_select + limited_ctr_offsets

        else:
            vote_xyz = xyz_select + ctr_offsets

        if self.output_feats:
            new_features = self.norm(features_select + feat_offets)
            return vote_xyz.squeeze(0), new_features.squeeze(0)
        else:
            return vote_xyz.squeeze(0), None