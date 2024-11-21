'''vote encoder'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.lcrnet.modules.netvlad.NetVlad import NetVLADLoupe2 as NetVLADLoupe
from experiments.lcrnet.backbone4 import KPEncoder 


class LCRNet_GlobalDescrition(nn.Module):
    def __init__(self, cfg):
        super(LCRNet_GlobalDescrition, self).__init__()

        self.encoder = KPEncoder(
            cfg.backbone.input_dim,
            cfg.backbone.init_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.init_radius,
            cfg.backbone.init_sigma,
            cfg.backbone.group_norm)



        self.netvlad = NetVLADLoupe(feature_size=1024,
                                cluster_size=64,
                                output_dim=256,
                                gating=True, add_norm=True, is_training=self.training)    
        

        
        if 'train_mode' in cfg:
            self.train_mode = cfg.train_mode
    
    def GlobalDescritionHEAD(self, feats_c, data_length):
        if not self.training:
            feats_c = F.normalize(feats_c, dim=2)
            feature_global = self.netvlad(feats_c)
            feature_global = F.normalize(feature_global, dim=1)

        else:
            # starttime=time.time()
            max_length = data_length.max()
            feature_size = feats_c.shape[1]
            mask = torch.zeros((len(data_length), max_length), dtype=bool, device=feats_c.device)
            feats_c_split = torch.split(feats_c, data_length.tolist())

            padded_feats_c = feats_c_split[0].new_full((len(data_length), max_length, feature_size), 0)
            for i, tensor in enumerate(feats_c_split):
                length = data_length[i].item()
                while len(tensor) < max_length:
                    tensor = torch.cat((tensor, tensor), dim=0)
                padded_feats_c[i] = tensor[:max_length, :]
                mask[i, :length] = True

            padded_feats_c = F.normalize(padded_feats_c, dim=2)
            feature_global = self.netvlad(padded_feats_c, mask)
            feature_global = F.normalize(feature_global, dim=1)
        return feature_global

    def forward(self, data_dict):
        output_dict = {}

        feats_c = data_dict['features'].detach()
        data_length = data_dict['lengths']

        if not self.training:
            feats = data_dict['features'].detach()
            feats_list = self.encoder(feats, data_dict)
            feats_c = feats_list[-1]

            feats_c = feats_c.view(1, -1, feats_c.size(-1))
            feature_global = self.GlobalDescritionHEAD(feats_c, data_length)

            output_dict['anc_global'] = feature_global

        else:
            ########### online train
            if self.train_mode == 'half':
                with torch.no_grad():
                    feats = data_dict['features'].detach()
                    feats_list = self.encoder(feats, data_dict)
                    data_length = data_dict['lengths'][-1]
                    feats_c = feats_list[-1]

                    feats_c=torch.cat([feats_c,data_dict['feats_c']],dim=0)
                    data_length=torch.cat([data_length,data_dict['lengths_c']],dim=0)
            else:
                feats = data_dict['features'].detach()
                feats_list = self.encoder(feats, data_dict)
                data_length = data_dict['lengths'][-1]
                feats_c = feats_list[-1]


            feature_global = self.GlobalDescritionHEAD(feats_c, data_length)
          
            pos_num = data_dict['pos_num']
            batch_size = data_dict['batch_size']
            pos_num_sum = sum(pos_num)

            anc_global = feature_global[:batch_size].reshape(batch_size, -1, 256)
            pos_global = feature_global[batch_size:pos_num_sum+batch_size].reshape(batch_size, -1, 256)
            neg_global = feature_global[pos_num_sum+batch_size:].reshape(batch_size, -1, 256)

            output_dict['pos_global'] = pos_global
            output_dict['anc_global'] = anc_global
            output_dict['neg_global'] = neg_global

        return output_dict


from experiments.lcrnet.loss_ld import best_pos_distance
class TripletLoss(nn.Module):
    def __init__(self, margin: float):
        super(TripletLoss, self).__init__()
        self.margin = margin
        # self.triplet_selector = triplet_selector
    
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
    
    
def create_model(cfg):
    model = LCRNet_GlobalDescrition(cfg)
    return model


def main():
    from experiments.lcrnet.config_ld import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == '__main__':
    main()
