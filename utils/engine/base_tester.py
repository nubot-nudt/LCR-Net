import sys
import argparse
import os
import os.path as osp
import time
import json
import abc
from collections import OrderedDict

import torch
import ipdb
import torch.nn as nn
import torch.distributed as dist

from utils.utils.torch import all_reduce_tensors, release_cuda, initialize
from utils.engine.logger import Logger



def inject_default_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', default=None, help='load from snapshot')
    parser.add_argument('--test_epoch', type=int, default=None, help='test epoch')
    parser.add_argument('--test_iter', type=int, default=None, help='test iteration')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank for ddp')
    # parser.add_argument('--dataset', type=str, default='kitti', help='kitti kitti360 apollo mulran self')
    parser.add_argument('--seq', type=int, nargs='+', default=None, help='')
    parser.add_argument('--dataset', type=str, default='kitti', help='kitti kitti360 apollo mulran ford')
    parser.add_argument('--lc_text_root', type=str, default='/loop_overlap0.3', help='/loop_overlap0.3 /loop_distance4')
    parser.add_argument('--reg_text_root', type=str, default='/icp10', help='/icp*')
    parser.add_argument('--vis', type=bool, default=False, help='enable visualization')
    '''for demo'''
    parser.add_argument('--pos_frame', type=int, default=[3854, 3528, 4481, 4481], nargs='+', help='')
    parser.add_argument('--anc_frame', type=int, default=[958, 560, 26, 958], nargs='+', help='')
    parser.add_argument('--dataset_root', type=str, default='demo/data_demo', help='')
    parser.add_argument('--gt_text_root', type=str, default=None, help='')
    return parser


class BaseTester(abc.ABC):
    def __init__(self, cfg, local_rank=None, logger=None, parser=None, cudnn_deterministic=True):
        # parser
        parser = inject_default_parser(parser)
        self.args = parser.parse_args()

        if local_rank is not None:
            self.args.local_rank = local_rank
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '54321'
            self.world_size = cfg.nprocs

        if local_rank == 0 and logger is not None:
            self.logger = logger
            # print(local_rank)
        else:
            # log_file = osp.join(cfg.log_dir, 'test-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
            log_file = None
            self.logger = Logger(log_file=log_file, local_rank=1)

            # print(local_rank)

        # command executed
        message = 'Command executed: ' + ' '.join(sys.argv)
        self.logger.info(message)

        # find snapshot
        if self.args.snapshot is None:
            if self.args.test_epoch is not None:
                self.args.snapshot = osp.join(cfg.snapshot_dir, 'epoch-{}.pth.tar'.format(self.args.test_epoch))
            elif self.args.test_iter is not None:
                self.args.snapshot = osp.join(cfg.snapshot_dir, 'iter-{}.pth.tar'.format(self.args.test_iter))
        # if self.args.snapshot is None:
        #     raise RuntimeError('Snapshot is not specified.')

        # print config
        # message = 'Configs:\n' + json.dumps(cfg, indent=4)
        # self.logger.info(message)

        # cuda and distributed
        if not torch.cuda.is_available():
            raise RuntimeError('No CUDA devices available.')
        self.distributed = self.args.local_rank != -1
        if self.distributed:
            torch.cuda.set_device(self.args.local_rank)
            # dist.init_process_group(backend='nccl')
            # self.world_size = dist.get_world_size()
            dist.init_process_group(backend='nccl', rank=self.args.local_rank, world_size=self.world_size)
            self.local_rank = self.args.local_rank
            self.logger.info(f'Using DistributedDataParallel mode (world_size: {self.world_size})')
        else:
            if torch.cuda.device_count() > 1:
                self.logger.warning('DataParallel is deprecated. Use DistributedDataParallel instead.')
            self.world_size = 1
            self.local_rank = 0
            self.logger.info('Using Single-GPU mode.')
        self.cudnn_deterministic = cudnn_deterministic
        self.seed = cfg.seed
        initialize(seed=self.seed, cudnn_deterministic=self.cudnn_deterministic)

        # state
        self.model = None
        self.iteration = None

        self.test_loader = None
        self.saved_states = {}

        # self.training = False
        # self.evaluating = False

    def load_snapshot(self, snapshot):
        self.logger.info('Loading from "{}".'.format(snapshot))
        state_dict = torch.load(snapshot, map_location=torch.device('cpu'), weights_only=True)
        assert 'model' in state_dict, 'No model can be loaded.'
        
        model_dict = state_dict['model']
        if self.distributed:
            model_dict = OrderedDict([('module.' + key, value) for key, value in model_dict.items()])
        
        # self.model.load_state_dict(model_dict, strict=True)
        self.model.load_state_dict(model_dict, strict=False)
        self.logger.info('Model has been loaded.')
        # self.logger.info(self.model.state_dict())

    def register_model(self, model):
        r"""Register model. DDP is automatically used."""
        if self.distributed:
            local_rank = self.local_rank
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        self.model = model
        # message = 'Model description:\n' + str(model)
        # self.logger.info(message)
        return model

    def register_loader(self, test_loader):
        r"""Register data loader."""
        self.test_loader = test_loader
    
    def release_tensors(self, result_dict):
        r"""All reduce and release tensors."""
        if self.distributed:
            result_dict = all_reduce_tensors(result_dict, world_size=self.world_size)
        result_dict = release_cuda(result_dict)
        return result_dict

    @abc.abstractmethod
    def run(self):
        raise NotImplemented
