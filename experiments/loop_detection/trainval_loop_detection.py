import time

import torch.optim as optim

from utils.engine import EpochBasedTrainer

from experiments.lcrnet.dataset_loop_detection import train_valid_data_loader
from experiments.lcrnet.config_ld import make_cfg
from experiments.lcrnet.config_ld_online import make_cfg as make_cfg_online
from experiments.lcrnet.model_family.LCRNet_GlobalDescrition import create_model, TripletLoss
from experiments.lcrnet.adan import Adan

class Trainer(EpochBasedTrainer):
    def __init__(self, cfg):
        super().__init__(cfg, max_epoch=cfg.optim.max_epoch)

        if self.args.train_mode=='online':
            cfg = make_cfg_online()
        cfg.train_mode = self.args.train_mode

        # dataloader
        start_time = time.time()
        train_loader, val_loader, neighbor_limits = train_valid_data_loader(cfg, self.distributed, 'kitti')
        loading_time = time.time() - start_time
        message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
        self.logger.info(message)
        message = 'Calibrate neighbors: {}.'.format(neighbor_limits)
        self.logger.info(message)
        self.register_loader(train_loader, val_loader)


        cfg.neighbor_limits = neighbor_limits
        # model, optimizer, scheduler
        model = create_model(cfg).cuda()
        # model = self.register_model(model, incemental=False)
        model = self.register_model(model, incemental=True)
        optimizer = Adan(model.parameters(), lr=cfg.optimadan.lr, weight_decay=cfg.optimadan.weight_decay)
        # optimizer = optim.Adam(model.parameters(), lr=cfg.optim.lr, weight_decayw=cfg.optim.eight_decay)
        self.register_optimizer(optimizer)
        scheduler = optim.lr_scheduler.StepLR(optimizer, cfg.optim.lr_decay_steps, gamma=cfg.optim.lr_decay)
        self.register_scheduler(scheduler)

        self.loss_func = TripletLoss(cfg.triplet_loss.margin).cuda()

    def train_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(output_dict)
        # result_dict = self.evaluator(output_dict, data_dict)
        result_dict={}
        loss_dict.update(result_dict)
        return output_dict, loss_dict

    def val_step(self, epoch, iteration, data_dict):
        pass

def main():

    cfg = make_cfg()
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == '__main__':
    main()
