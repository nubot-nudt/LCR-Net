import argparse
import os
import os.path as osp

from easydict import EasyDict as edict

from utils.utils.common import ensure_dir

from experiments.lcrnet.config_model import make_cfg as make_cfg_model

_C = edict()

# random seed
_C.seed = 7351

# dirs
_C.working_dir = osp.dirname(osp.realpath(__file__))
_C.root_dir = osp.dirname(osp.dirname(_C.working_dir))
_C.exp_name = osp.basename(_C.working_dir)
# _C.output_dir = osp.join(_C.root_dir, 'output', _C.exp_name)

# train data
_C.train = edict()
_C.train.batch_size = 6
_C.train.num_workers = 6
_C.train.point_limit = 30000
_C.train.use_augmentation = True
_C.train.augmentation_noise = 0.01
_C.train.augmentation_min_scale = 0.8
_C.train.augmentation_max_scale = 1.2
_C.train.augmentation_shift = 2.0
_C.train.augmentation_rotation = 1.0
_C.train.pos_num = 6
_C.train.neg_num = 6
_C.train.ground_segmentation = False
_C.train.pre_load = False


# test config
_C.test = edict()
_C.test.batch_size = 1
_C.test.num_workers = 8
_C.test.point_limit = None
_C.test.ground_segmentation = False

# eval config
_C.eval = edict()
_C.eval.acceptance_overlap = 0.0
_C.eval.acceptance_radius = 1.0
_C.eval.inlier_ratio_threshold = 0.05
_C.eval.rre_threshold = 5.0
_C.eval.rte_threshold = 2.0

# ransac
_C.ransac = edict()
_C.ransac.distance_threshold = 0.3
_C.ransac.num_points = 4
_C.ransac.num_iterations = 50000

# optim config
_C.optim = edict()
_C.optim.lr = 1e-4
_C.optim.lr_decay = 0.95
_C.optim.lr_decay_steps = 4
_C.optim.weight_decay = 1e-6
_C.optim.max_epoch = 180
_C.optim.grad_acc_steps = 1

_C.optimadan = edict()
_C.optimadan.lr = 1e-4
_C.optimadan.lr_decay = 0.95
_C.optimadan.lr_decay_steps = 4

_C.optimadan.weight_decay = 1e-6
_C.optimadan.opt_betas = None
_C.optimadan.opt_eps = None
_C.optimadan.max_grad_norm = 0
_C.optimadan.no_prox = False




def make_cfg():
    model_cfg = make_cfg_model()
    _C.update(model_cfg)

    _C.output_dir = osp.join(model_cfg.output_root, 'loop_detection')
    _C.snapshot_dir = osp.join(_C.output_dir, 'snapshots')
    _C.log_dir = osp.join(_C.snapshot_dir, 'logs')
    _C.event_dir = osp.join(_C.snapshot_dir, 'events')
    _C.ld_feature_dir = osp.join(_C.output_dir, 'features/')


    ensure_dir(_C.output_dir)
    ensure_dir(_C.snapshot_dir)
    ensure_dir(_C.log_dir)
    ensure_dir(_C.event_dir)
    ensure_dir(_C.ld_feature_dir)

    return _C


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--link_output', dest='link_output', action='store_true', help='link output dir')
    args = parser.parse_args()
    return args


def main():
    cfg = make_cfg()
    args = parse_args()
    if args.link_output:
        os.symlink(cfg.output_dir, 'output')


if __name__ == '__main__':
    main()
