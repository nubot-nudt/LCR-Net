import argparse
import os
import os.path as osp

from easydict import EasyDict as edict


_C = edict()

# random seed
_C.seed = 7351

# data
_C.data = edict()
# _C.data.dataset_root = osp.join(_C.root_dir, 'data', 'Kitti')
_C.data.dataset_root = '/mnt/Mount/Dataset/KITTI_odometry'
_C.data.dataset_ford_root = '/mnt/Mount/Dataset/ford-campus'
_C.data.dataset_360_root = '/mnt/Mount/Dataset/KITTI-360'
_C.data.mulran_root = '/mnt/Mount/Dataset/mulran_process'
_C.data.apollo_root = '/mnt/Mount/Dataset/apollo'

_C.output_root = osp.join('/mnt/Mount/LCRNet-output')

# ransac
_C.ransac = edict()
_C.ransac.distance_threshold = 0.3
_C.ransac.num_points = 4
_C.ransac.num_iterations = 50000

# model - backbone
_C.backbone = edict()
# _C.backbone.num_stages = 5
_C.backbone.num_stages = 4
_C.backbone.init_voxel_size = 0.3
_C.backbone.kernel_size = 15
_C.backbone.base_radius = 4.25
_C.backbone.base_sigma = 2.0
_C.backbone.init_radius = _C.backbone.base_radius * _C.backbone.init_voxel_size
_C.backbone.init_sigma = _C.backbone.base_sigma * _C.backbone.init_voxel_size
_C.backbone.group_norm = 32
_C.backbone.input_dim = 1
_C.backbone.init_dim = 64
_C.backbone.output_dim = 256

# model - Global
_C.model = edict()
_C.model.ground_truth_matching_radius = 0.45
_C.model.num_points_in_patch = 128
_C.model.num_sinkhorn_iterations = 100
_C.model.ground_truth_corres_radius = 2.4
_C.model.n2p_score_threshold = 0.1
_C.model.p2p_score_threshold = 0.1


# model - Coarse Matching
_C.coarse_matching = edict()
_C.coarse_matching.num_targets = 128
_C.coarse_matching.overlap_threshold = 0.1
# _C.coarse_matching.num_correspondences = 256
# _C.coarse_matching.num_correspondences = 128
_C.coarse_matching.num_correspondences = None


# model - THDRoformer
_C.GAT = edict()
# _C.GAT.input_dim = 2048
_C.GAT.input_dim = 1024
_C.GAT.hidden_dim = 128
_C.GAT.output_dim = 256
_C.GAT.num_heads = 4
_C.GAT.num_layers = 4
_C.GAT.k = None

_C.Vote = edict()
_C.Vote.MAX_TRANSLATE_RANGE=4.2
_C.Vote.MLPS=[512, 256]
_C.Vote.NMS_radius=2.4
_C.Vote.n2n_overlap_threshold=1.2
_C.Vote.n2p_overlap_threshold=0.6
_C.Vote.p2p_overlap_threshold=0.6


# model - Fine Matching using gap loss
_C.fine_matching = edict()
_C.fine_matching.acceptance_radius = 0.45
_C.fine_matching.mutual = False
_C.fine_matching.topk = 1
_C.fine_matching.confidence_threshold = 0
_C.fine_matching.use_dustbin = True
_C.fine_matching.use_global_score = False
_C.fine_matching.correspondence_threshold = 3
_C.fine_matching.correspondence_limit = None
_C.fine_matching.num_refinement_steps = 5

# loss - Coarse level
_C.coarse_loss = edict()
_C.coarse_loss.positive_margin = 0.1
_C.coarse_loss.negative_margin = 1.4
_C.coarse_loss.positive_optimal = 0.1
_C.coarse_loss.negative_optimal = 1.4
_C.coarse_loss.log_scale = 40
_C.coarse_loss.positive_overlap = 0.1
# _C.coarse_loss.positive_distance = 2.4

# loss - Fine level
_C.fine_loss = edict()
_C.fine_loss.positive_radius = 0.45

# loss - distribution level
_C.distribution_loss = edict()
_C.distribution_loss.triplet_loss_gamma = 0.5


_C.triplet_loss = edict()
_C.triplet_loss.margin = 0.5


# loss - Overall
_C.loss = edict()
_C.loss.weight_coarse_loss = 1.0
_C.loss.weight_vote_loss = 0.25
_C.loss.weight_gap_loss = 5



def make_cfg():
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
