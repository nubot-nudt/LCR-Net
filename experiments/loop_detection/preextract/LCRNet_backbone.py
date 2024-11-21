'''vote encoder'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from experiments.lcrnet.backbone4 import KPEncoder 


class LCRNet_backbone(nn.Module):
    def __init__(self, cfg):
        super(LCRNet_backbone, self).__init__()
       

        self.encoder = KPEncoder(
            cfg.backbone.input_dim,
            cfg.backbone.init_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.init_radius,
            cfg.backbone.init_sigma,
            cfg.backbone.group_norm)

        

    def forward(self, data_dict):
        output_dict = {}


        feats_c = data_dict['features'].detach()
        data_length = data_dict['lengths']
            
        feats = data_dict['features'].detach()
        feats_list = self.encoder(feats, data_dict)
        feats_c = feats_list[-1]


        points_c = data_dict['points'][-1].detach()
        output_dict['feats'] = feats_c
        output_dict['points'] = points_c
        return output_dict

      
def create_model(cfg):
    model = LCRNet_backbone(cfg)
    return model


def main():
    from experiments.lcrnet.config_reg import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == '__main__':
    main()
