"""

"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, num_heads):
        super(RotaryPositionalEmbedding, self).__init__()

    def forward(self, desc0, pos_emb):
        batch_dim, num_heads, num_points, feature_dim = desc0.size()
        # pos_emb = self.Linear(pos_emb)

        desc = desc0.view(batch_dim,num_heads,num_points,np.int(feature_dim/2),2)
        desc = torch.cat((-desc[:,:,:,:,1].unsqueeze(-1), desc[:,:,:,:,0].unsqueeze(-1)), 4)
        desc = desc.view(batch_dim,num_heads,num_points,feature_dim)


        theta = F.interpolate(pos_emb.squeeze(0), scale_factor=2, mode='nearest').unsqueeze(0)


        return torch.mul(desc0,torch.cos(theta))+torch.mul(desc,torch.sin(theta))

class LinearLearnablePosEmbedding(nn.Module):
    def __init__(self, hidden_dim, reduction_a='max'):
        super(LinearLearnablePosEmbedding, self).__init__()

        self.encoder = nn.Linear(3, int(hidden_dim))
        self.encoder2 = nn.Linear(int(hidden_dim), int(hidden_dim/2))


    def forward(self, points):

        embeddings = self.encoder2(self.encoder(points))

        return embeddings