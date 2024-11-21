import argparse
import os.path as osp
import time

import numpy as np

from utils.engine import SingleTester
from utils.utils.common import ensure_dir, get_log_string, empty_dir
from utils.utils.torch import release_cuda

from experiments.lcrnet.dataset_loop_detection import test_loop_detection_data_loader 

from experiments.lcrnet.model_family.LCRNet_GlobalDescrition import create_model
from experiments.lcrnet.config_ld import make_cfg

from eval_loop_detection_overlap_dataset import eval_one_epoch
class Tester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg)

        cfg.dataset = self.args.dataset
        cfg.seq = self.args.seq

        # dataloader
        start_time = time.time()
        data_loader, neighbor_limits = test_loop_detection_data_loader(cfg, self.distributed, cfg.dataset)

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
        self.output_dir = osp.join(cfg.ld_feature_dir+cfg.dataset)
        ensure_dir(self.output_dir)
        # empty_dir(self.output_dir)

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
        anc_global = output_dict['anc_global']

        file_name = osp.join(self.output_dir, f'{seq_id}_{idx}.npz')
        np.savez_compressed(
            file_name,
            anc_global=release_cuda(anc_global)
        )

        del data_dict, output_dict, idx, seq_id
           
    def eval_loop_closure(self, cfg):
        eval_one_epoch(cfg, [0])
        


def main():

    cfg = make_cfg()

    snapshot='./weights/best-model-ld.tar'

    tester = Tester(cfg)
    tester.run(snapshot)

    tester.eval_loop_closure(cfg)



if __name__ == '__main__':
    main()