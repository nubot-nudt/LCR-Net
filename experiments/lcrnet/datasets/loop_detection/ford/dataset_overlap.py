import os.path as osp
import numpy as np
import torch.utils.data

import open3d as o3d
from experiments.lcrnet.datasets.utils.ford import load_ford_gt_pose, load_ford_gt_pair_pr


def make_dataset_kitti(txt_path, mode, pre_load=False,feature_dict_dir=None):
        seq_list = [0]
      
        datasets = []
        poses = []

        for seq in seq_list:
            dataset = load_ford_gt_pose(txt_path, seq)
            datasets += (dataset)

        feats_dict=None
        return datasets, poses, feats_dict

class FordDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root,
        subset,
        point_limit=None,
        use_augmentation=False,
        augmentation_noise=0.005,
        augmentation_min_scale=0.8,
        augmentation_max_scale=1.2,
        augmentation_shift=2.0,
        augmentation_rotation=1.0,
        return_corr_indices=False,
        matching_radius=None,
        pos_num=6,
        neg_num=6,
        pre_load=False,
        ground_segmentation=False,
    ):
        super(FordDataset, self).__init__()

        self.dataset_root = dataset_root
        self.subset = subset
        self.point_limit = point_limit

        self.use_augmentation = use_augmentation
        self.augmentation_noise = augmentation_noise
        self.augmentation_min_scale = augmentation_min_scale
        self.augmentation_max_scale = augmentation_max_scale
        self.augmentation_shift = augmentation_shift
        self.augmentation_rotation = augmentation_rotation

        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius
        if self.return_corr_indices and self.matching_radius is None:
            raise ValueError('"matching_radius" is None but "return_corr_indices" is set.')

        self.feature_dict={}
        self.pre_load=pre_load
        self.metadata, self.poses, self.feature_dict = make_dataset_kitti(self.dataset_root,subset)
        


        self.pos_num=pos_num
        self.neg_num=neg_num

        self.ground_segmentation=ground_segmentation



    def _load_point_cloud(self, file_name):
        points = np.load(file_name)
        if self.point_limit is not None and points.shape[0] > self.point_limit:
            indices = np.random.permutation(points.shape[0])[: self.point_limit]
            points = points[indices]
        return points


    def __getitem__(self, index):
        data_dict = {}
        metadata = self.metadata[index]



            
        data_dict['seq_id'] = metadata['seq_id']
        data_dict['idx'] = metadata['idx']
        idx=data_dict['idx']
        seq_id=data_dict['seq_id']

        
        '''online test'''
        anc_points = self._load_point_cloud(osp.join(self.dataset_root, 'downsampled_xyzi','%02d'%data_dict['seq_id'], '%06d.npy' % (data_dict['idx'])))[:,:3]
        # data_dict['anc_points'] = anc_points.astype(np.float32)
        # data_dict['anc_feats'] = np.ones((anc_points.shape[0], 1), dtype=np.float32)
        anc_lengths = np.int64([anc_points.shape[0]])
        anc_feats = np.ones((anc_points.shape[0], 1), dtype=np.float32)


        data_dict['anc_points'] = anc_points.astype(np.float32)
        data_dict['anc_lengths'] = anc_lengths
        data_dict['anc_feats'] = anc_feats.astype(np.float32)
        data_dict['pass']=False

        return data_dict

    def __len__(self):
        return len(self.metadata)