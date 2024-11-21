import argparse
import os.path as osp
import time

import numpy as np

from utils.engine import SingleTester
from utils.utils.common import ensure_dir, get_log_string, empty_dir
from utils.utils.torch import release_cuda

from experiments.lcrnet.dataset_registration import test_data_loader 
from experiments.lcrnet.loss_reg import Evaluator
from experiments.lcrnet.model_family.LCRNet_Matching import create_model
from experiments.lcrnet.config_reg import make_cfg

class Tester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg)

        cfg.dataset = self.args.dataset
        cfg.test.reg_text_root = self.args.reg_text_root
        cfg.feature_dir = cfg.feature_dir + f'{cfg.dataset}'
        cfg.vis = self.args.vis

        # dataloader
        start_time = time.time()
        data_loader, neighbor_limits = test_data_loader(cfg, self.distributed, cfg.dataset)

        loading_time = time.time() - start_time
        message = f'Data loader created: {loading_time:.3f}s collapsed.'
        self.logger.info(message)
        message = f'Calibrate neighbors: {neighbor_limits}.'
        self.logger.info(message)
        self.register_loader(data_loader)

        cfg.neighbor_limits = neighbor_limits
        # model 
        model = create_model(cfg).cuda()
        self.register_model(model)

        # evaluator
        self.evaluator = Evaluator(cfg).cuda()

        # preparation
        self.output_dir = osp.join(cfg.feature_dir)
        ensure_dir(self.output_dir)
        empty_dir(self.output_dir)

    def test_step(self, iteration, data_dict):
        output_dict = self.model(data_dict)
        # output_dict = []
        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
        result_dict = self.evaluator(output_dict, data_dict)
        # result_dict = []
        return result_dict

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        seq_id = data_dict['seq_id']
        positive_idx = data_dict['pos_idx']
        anchor_idx = data_dict['anc_idx']
        message = f'seq_id: {seq_id}, id0: {positive_idx}, id1: {anchor_idx}'
        message += ', ' + get_log_string(result_dict=result_dict)
        message += ', nCorr: {}'.format(output_dict['corr_scores'].shape[0])
        return message

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        seq_id = data_dict['seq_id']
        positive_idx = data_dict['pos_idx']
        anchor_idx = data_dict['anc_idx']

        file_name = osp.join(self.output_dir, f'{seq_id}_{anchor_idx}_{positive_idx}.npz')
        np.savez_compressed(
            file_name,
            pos_points_f=release_cuda(output_dict['pos_points_f']),
            anc_points_f=release_cuda(output_dict['anc_points_f']),
            pos_points_c=release_cuda(output_dict['pos_points_c']),
            anc_points_c=release_cuda(output_dict['anc_points_c']),
            # pos_feats_c=release_cuda(output_dict['pos_feats_c']),
            # anc_feats_c=release_cuda(output_dict['anc_feats_c']),
            pos_node_corr_indices=release_cuda(output_dict['pos_node_corr_indices']),
            anc_node_corr_indices=release_cuda(output_dict['anc_node_corr_indices']),
            pos_corr_points=release_cuda(output_dict['pos_corr_points']),
            anc_corr_points=release_cuda(output_dict['anc_corr_points']),
            corr_scores=release_cuda(output_dict['corr_scores']),
            gt_node_corr_indices=release_cuda(output_dict['gt_node_corr_indices']),
            gt_node_corr_overlaps=release_cuda(output_dict['gt_node_corr_overlaps']),
            estimated_transform=release_cuda(output_dict['estimated_transform']),
            transform=release_cuda(data_dict['transform'])
        )

def main():

    cfg = make_cfg()

    snapshot='./weights/best-model-reg.tar'

    tester = Tester(cfg)
    tester.run(snapshot)


if __name__ == '__main__':
    main()