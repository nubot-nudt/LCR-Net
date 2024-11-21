import argparse
import os.path as osp
import time

import numpy as np

from utils.engine import SingleTester
from utils.utils.common import ensure_dir, get_log_string
from utils.utils.torch import release_cuda

from experiments.loop_detection.preextract.LCRNet_backbone import create_model
from experiments.lcrnet.config_ld import make_cfg

import torch.multiprocessing as mp

from experiments.lcrnet.data import (
    calibrate_neighbors_stack_mode,
    build_dataloader_stack_mode,
    all_collate_fn_stack_mode
)
from experiments.lcrnet.dataset_loop_detection import place_recognition_dataset_initialization

def pre_extract_data_loader(cfg, distributed, dataset):
    dataset_init = place_recognition_dataset_initialization()


    test_dataset = getattr(dataset_init, '%s_dataset_initialization' % dataset)(
                cfg, dataset, 'all'
            )
    neighbor_limits = calibrate_neighbors_stack_mode(
        test_dataset,
        all_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )
  
    test_loader = build_dataloader_stack_mode(
        test_dataset,
        all_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        distributed=distributed,
        precompute_data=True
    )

    return test_loader, neighbor_limits

def main(cfg, local_rank=None, logger=None):
    tester = Tester(cfg, local_rank, logger)
   
    snapshot = './weights/mixed/best_model.pth.tar'
    
    
    tester.run(snapshot)
    # tester.run(snapshot)

       
        
class Tester(SingleTester):
    def __init__(self, cfg, local_rank=None, logger=None):
        super().__init__(cfg, local_rank=local_rank, logger=logger)

        # dataloader
        start_time = time.time()

        data_loader, neighbor_limits = pre_extract_data_loader(cfg, self.distributed, cfg.dataset)

        loading_time = time.time() - start_time
        message = f'Data loader created: {loading_time:.3f}s collapsed.'
        self.logger.info(message)
        message = f'Calibrate neighbors: {neighbor_limits}.'
        self.logger.info(message)
        self.register_loader(data_loader)

        cfg.neighbor_limits=neighbor_limits
        # model     
        model = create_model(cfg).cuda()
        self.register_model(model)

        # preparation
        self.output_dir = osp.join(cfg.data.dataset_root)
        ensure_dir(self.output_dir)

    def test_step(self, iteration, data_dict):
        output_dict = self.model(data_dict)
        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
        pass

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        seq_id = data_dict['seq_id']
        idx = data_dict['idx']
        message = f'seq_id: {seq_id}, id: {idx}'
        # message += ', ' + get_log_string(result_dict=result_dict)
        # message += ', nCorr: {}'.format(output_dict['corr_scores'].shape[0])
        return message

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        seq_id = data_dict['seq_id']
        idx = data_dict['idx']
        points4 = output_dict['points']
        feats4 = output_dict['feats']

        dir_name = osp.join(self.output_dir, 'feature(kpconv4)' ,f'{seq_id}')
        ensure_dir(dir_name)
        file_name = osp.join(dir_name ,f'{seq_id}_{idx}.npz')
        np.savez_compressed(
            file_name,
            points=release_cuda(points4),
            feats=release_cuda(feats4),
        )



if __name__ == '__main__':

    cfg = make_cfg()

    cfg.dataset='kitti'
    cfg.seq=None
    
    nprocs = 4
    if False:
        cfg.nprocs = nprocs
        '''torch.multiprocessing.spawn: directly run the script'''
        mp.spawn(main,
            nprocs=nprocs,
            args=(0, cfg, logger),
            )
    else:
        mp.set_start_method('spawn')
        main(cfg)
