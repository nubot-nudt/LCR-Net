import torch
import torch.nn as nn

from experiments.lcrnet.modules.kpconv import ConvBlock, ResidualBlock, UnaryBlock, LastUnaryBlock, nearest_upsample

from experiments.lcrnet.modules.vote.vote import Vote_layer, NMS_cpu, NMS
from experiments.lcrnet.modules.ops import radius_search
from experiments.lcrnet.modules.ops import index_select
import time

class KPEncoder(nn.Module):
    def __init__(self, input_dim, init_dim, kernel_size, init_radius, init_sigma, group_norm):
        super(KPEncoder, self).__init__()

        self.encoder1_1 = ConvBlock(input_dim, init_dim, kernel_size, init_radius, init_sigma, group_norm)
        self.encoder1_2 = ResidualBlock(init_dim, init_dim * 2, kernel_size, init_radius, init_sigma, group_norm)

        self.encoder2_1 = ResidualBlock(
            init_dim * 2, init_dim * 2, kernel_size, init_radius, init_sigma, group_norm, strided=True
        )
        self.encoder2_2 = ResidualBlock(
            init_dim * 2, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm
        )
        self.encoder2_3 = ResidualBlock(
            init_dim * 4, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm
        )

        self.encoder3_1 = ResidualBlock(
            init_dim * 4,
            init_dim * 4,
            kernel_size,
            init_radius * 2,
            init_sigma * 2,
            group_norm,
            strided=True,
        )
        self.encoder3_2 = ResidualBlock(
            init_dim * 4, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, group_norm
        )
        self.encoder3_3 = ResidualBlock(
            init_dim * 8, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, group_norm
        )

        self.encoder4_1 = ResidualBlock(
            init_dim * 8,
            init_dim * 8,
            kernel_size,
            init_radius * 4,
            init_sigma * 4,
            group_norm,
            strided=True,
        )
        self.encoder4_2 = ResidualBlock(
            init_dim * 8, init_dim * 16, kernel_size, init_radius * 8, init_sigma * 8, group_norm
        )
        self.encoder4_3 = ResidualBlock(
            init_dim * 16, init_dim * 16, kernel_size, init_radius * 8, init_sigma * 8, group_norm
        )

    def forward(self, feats, data_dict):
        feats_list = []

        points_list = data_dict['points']
        neighbors_list = data_dict['neighbors']
        subsampling_list = data_dict['subsampling']

        feats_s1 = feats
        feats_s1 = self.encoder1_1(feats_s1, points_list[0], points_list[0], neighbors_list[0])
        feats_s1 = self.encoder1_2(feats_s1, points_list[0], points_list[0], neighbors_list[0])
        feats_list.append(feats_s1)

        feats_s2 = self.encoder2_1(feats_s1, points_list[1], points_list[0], subsampling_list[0])
        feats_s2 = self.encoder2_2(feats_s2, points_list[1], points_list[1], neighbors_list[1])
        feats_s2 = self.encoder2_3(feats_s2, points_list[1], points_list[1], neighbors_list[1])
        feats_list.append(feats_s2)

        feats_s3 = self.encoder3_1(feats_s2, points_list[2], points_list[1], subsampling_list[1])
        feats_s3 = self.encoder3_2(feats_s3, points_list[2], points_list[2], neighbors_list[2])
        feats_s3 = self.encoder3_3(feats_s3, points_list[2], points_list[2], neighbors_list[2])
        feats_list.append(feats_s3)

        feats_s4 = self.encoder4_1(feats_s3, points_list[3], points_list[2], subsampling_list[2])
        feats_s4 = self.encoder4_2(feats_s4, points_list[3], points_list[3], neighbors_list[3])
        feats_s4 = self.encoder4_3(feats_s4, points_list[3], points_list[3], neighbors_list[3])
        feats_list.append(feats_s4)

        # feats_list.reverse()

        return feats_list


class Vote_Encoder(nn.Module):
    def __init__(self, input_dim, init_dim, kernel_size, init_radius, init_sigma, group_norm, vote, neighbor_limits):
        super(Vote_Encoder, self).__init__()

        self.vote = Vote_layer(vote, 1, output_feats=False)
        self.nms = NMS(vote)
        self.NMS_radius=vote.NMS_radius

        self.encoder6_1 = ResidualBlock(
            init_dim * 4,
            init_dim * 4,
            kernel_size,
            init_radius * 8,    
            init_sigma * 8,    
            group_norm,
            strided=True,
        )

        self.encoder6_2 = ResidualBlock(
            init_dim * 4, init_dim * 8, kernel_size, init_radius * 16, init_sigma * 16, group_norm
        )
        self.encoder6_3 = ResidualBlock(
            init_dim * 8, init_dim * 8, kernel_size, init_radius * 16, init_sigma * 16, group_norm
        )

        self.init_radius = init_radius

        self.neighbor_limits = neighbor_limits

    def forward(self, feats, data_dict, neighbor_limit=None):

        vote_dict = {}



        pos_length_c = data_dict['lengths'][-1][0].item()
        anc_length_c = data_dict['lengths'][-1][1].item()

        points_c = data_dict['points'][-1].detach()
        feats_c = feats

        #####################################
        # 3.2. vote layer
        shifted_points_c, shifted_feats_c = self.vote(points_c, feats_c)

        shifted_pos_points_c = shifted_points_c[:pos_length_c]
        shifted_anc_points_c = shifted_points_c[pos_length_c:anc_length_c+pos_length_c]
        vote_dict['shifted_pos_points_c'] = shifted_pos_points_c
        vote_dict['shifted_anc_points_c'] = shifted_anc_points_c


        masks, length = self.nms(shifted_points_c, data_dict['lengths'][-1])
        nms_shifted_points_c = shifted_points_c[masks]


        vote_dict['length'] = length

        node_knn_indices = radius_search(
            nms_shifted_points_c.cpu(),
            shifted_points_c.cpu(),
            length,
            data_dict['lengths'][-1].cpu(),
            2.4,
            self.neighbor_limits[-1],
        )
        node_knn_indices=node_knn_indices.cuda()
        pos_neighbour=node_knn_indices[:length[0]]
        anc_neighbour=node_knn_indices[length[0]:length[0]+length[1]]

        pos_neighbour[pos_neighbour==pos_length_c+anc_length_c] = pos_length_c
        anc_neighbour=anc_neighbour-pos_length_c

        pos_neighbour_masks=pos_neighbour<pos_length_c
        anc_neighbour_masks=anc_neighbour<anc_length_c


        pos_padded_points_f = torch.cat([shifted_pos_points_c, torch.zeros_like(shifted_pos_points_c[:1])], dim=0)
        anc_padded_points_f = torch.cat([shifted_anc_points_c, torch.zeros_like(shifted_anc_points_c[:1])], dim=0)
        pos_node_knn_points = index_select(pos_padded_points_f, pos_neighbour, dim=0)
        anc_node_knn_points = index_select(anc_padded_points_f, anc_neighbour, dim=0)

        nms_shifted_pos_points_c = pos_node_knn_points.sum(1)/pos_neighbour_masks.sum(-1).unsqueeze(1)
        nms_shifted_anc_points_c = anc_node_knn_points.sum(1)/anc_neighbour_masks.sum(-1).unsqueeze(1)


        vote_dict['pos_points_c'] = nms_shifted_pos_points_c
        vote_dict['anc_points_c'] = nms_shifted_anc_points_c

        centers = torch.cat([nms_shifted_pos_points_c, nms_shifted_anc_points_c], dim=0)
        
       
        feats_s4 = feats_c
        q_points = centers
        s_points = data_dict['points'][-1].detach()
        q_length = length
        s_length = data_dict['lengths'][-1]
        
        if neighbor_limit is None:
            neighbor_limit = self.neighbor_limits
        subsampling = radius_search(
            q_points.cpu(),
            s_points.cpu(),
            q_length.cpu(),
            s_length.cpu(),
            self.init_radius*8,
            neighbor_limit[-2],
        )
        neighbors = radius_search(
            q_points.cpu(),
            q_points.cpu(),
            q_length.cpu(),
            q_length.cpu(),
            self.init_radius*16,
            neighbor_limit[-1],
        )

        feats_s5 = self.encoder6_1(feats_s4, q_points, s_points, subsampling.cuda())
        feats_s5 = self.encoder6_2(feats_s5, q_points, q_points, neighbors.cuda())
        feats_s5 = self.encoder6_3(feats_s5, q_points, q_points, neighbors.cuda())

        vote_dict['feats_c'] = feats_s5

        pos_feats_c = feats_s5[:length[0]]
        anc_feats_c = feats_s5[length[0]:]

        vote_dict['pos_feats_c'] = pos_feats_c
        vote_dict['anc_feats_c'] = anc_feats_c

        return vote_dict

class Vote_Encoder2(nn.Module):
    def __init__(self, input_dim, init_dim, kernel_size, init_radius, init_sigma, group_norm, vote, neighbor_limits):
        super(Vote_Encoder2, self).__init__()

        self.vote = Vote_layer(vote, 1, output_feats=False)
        self.nms = NMS_cpu(vote, neighbor_limits)
        self.NMS_radius=vote.NMS_radius

        self.encoder6_1 = ResidualBlock(
            init_dim * 4,
            init_dim * 4,
            kernel_size,
            init_radius * 8,    
            init_sigma * 8,    
            group_norm,
            strided=True,
        )

        self.encoder6_2 = ResidualBlock(
            init_dim * 4, init_dim * 8, kernel_size, init_radius * 16, init_sigma * 16, group_norm
        )
        self.encoder6_3 = ResidualBlock(
            init_dim * 8, init_dim * 8, kernel_size, init_radius * 16, init_sigma * 16, group_norm
        )

        self.init_radius = init_radius

        self.neighbor_limits = neighbor_limits

    def forward(self, feats, data_dict, neighbor_limit=None):

        vote_dict = {}

        a=time.time()

        pos_length_c = data_dict['lengths'][-1][0].item()
        anc_length_c = data_dict['lengths'][-1][1].item()

        points_c = data_dict['points'][-1].detach()
        feats_c = feats

        #####################################
        # 3.2. vote layer
        shifted_points_c, shifted_feats_c = self.vote(points_c, feats_c)

        shifted_pos_points_c = shifted_points_c[:pos_length_c]
        shifted_anc_points_c = shifted_points_c[pos_length_c:anc_length_c+pos_length_c]
        vote_dict['shifted_pos_points_c'] = shifted_pos_points_c
        vote_dict['shifted_anc_points_c'] = shifted_anc_points_c
        
        b=time.time()

        masks, node_knn_indices = self.nms(shifted_points_c, data_dict['lengths'][-1])
        length = torch.stack([masks[:data_dict['lengths'][-1][0]].sum(),masks[data_dict['lengths'][-1][0]:].sum()],dim=0)


        c=time.time()
        vote_dict['length'] = length

        node_knn_indices=node_knn_indices.cuda()

        shifted_padded_points_c = torch.cat([shifted_points_c, torch.zeros_like(shifted_points_c[:1])], dim=0)
        node_knn_points = index_select(shifted_padded_points_c, node_knn_indices, dim=0)
        centers=node_knn_points.sum(1)/(node_knn_indices<shifted_points_c.shape[0]).sum(-1).unsqueeze(1)
        vote_dict['nms_shifted_pos_points_c'] = centers[:length[0]]
        vote_dict['nms_shifted_anc_points_c'] = centers[length[0]:]


        d=time.time()
       
        feats_s4 = feats_c
        q_points = centers
        s_points = data_dict['points'][-1].detach()
        q_length = length
        s_length = data_dict['lengths'][-1]
        
        if neighbor_limit is None:
            neighbor_limit = self.neighbor_limits
        subsampling = radius_search(
            q_points.cpu(),
            s_points.cpu(),
            q_length.cpu(),
            s_length.cpu(),
            self.init_radius*8,
            neighbor_limit[-2],
        )
        neighbors = radius_search(
            q_points.cpu(),
            q_points.cpu(),
            q_length.cpu(),
            q_length.cpu(),
            self.init_radius*16,
            neighbor_limit[-1],
        )

        feats_s5 = self.encoder6_1(feats_s4, q_points, s_points, subsampling.cuda())
        feats_s5 = self.encoder6_2(feats_s5, q_points, q_points, neighbors.cuda())
        feats_s5 = self.encoder6_3(feats_s5, q_points, q_points, neighbors.cuda())

        e=time.time()
        # print('vote ', b-a)
        # print('nms ', c-b)
        # print('center ', d-c)
        # print('encode ', e-d)

        vote_dict['nms_shifted_feats_c'] = feats_s5

        return vote_dict
    


class KPDecoder(nn.Module):
    def __init__(self,  output_dim, init_dim, group_norm, neighbor_limit, init_radius):
        super(KPDecoder, self).__init__()

        self.decoder3 = UnaryBlock(init_dim * 12, init_dim * 8, group_norm)
        self.decoder2 = UnaryBlock(init_dim * 12, init_dim * 4, group_norm)
        self.decoder1 = LastUnaryBlock(init_dim * 6, int(init_dim*2))

        self.neighbor_limit = neighbor_limit
        self.init_radius=init_radius

    def forward(self, feats, data_dict):
        feats_list = []

        upsampling_list = data_dict['upsampling']


        feats_s1 = feats[0]
        feats_s2 = feats[1]
        feats_s3 = feats[2]
        feats_s4 = feats[3]

        latent_s3 = nearest_upsample(feats_s4, upsampling_list[2])
        latent_s3 = torch.cat([latent_s3, feats_s3], dim=1)
        latent_s3 = self.decoder3(latent_s3)
        feats_list.append(latent_s3)

        latent_s2 = nearest_upsample(latent_s3, upsampling_list[1])
        latent_s2 = torch.cat([latent_s2, feats_s2], dim=1)
        latent_s2 = self.decoder2(latent_s2)
        feats_list.append(latent_s2)

        latent_s1 = nearest_upsample(latent_s2, upsampling_list[0])
        latent_s1 = torch.cat([latent_s1, feats_s1], dim=1)
        latent_s1 = self.decoder1(latent_s1)
        feats_list.append(latent_s1)

        feats_list.reverse()


        return feats_list
 