import os.path as osp
import random

import numpy as np
import torch.utils.data

from utils.utils.common import load_pickle
from utils.utils.pointcloud import (
    random_sample_rotation,
    get_transform_from_rotation_translation,
    get_rotation_translation_from_transform,
)
from utils.utils.registration import get_correspondences

# from geotransformer.utils.kitti import load_kitti_gt_pair_registration
from experiments.lcrnet.datasets.utils.ford import load_ford_gt_pair_registration

def make_dataset_kitti(txt_path, mode):

        seq_list = [0]  

        dataset = []
        for seq in seq_list:
            dataset += (load_ford_gt_pair_registration(txt_path, seq))
           
        return dataset

class OdometryFordPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root,
        reg_text_root,
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
        ground_segmentation=False,
    ):
        super(OdometryFordPairDataset, self).__init__()

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
        self.ground_segmentation = ground_segmentation

        if self.return_corr_indices and self.matching_radius is None:
            raise ValueError('"matching_radius" is None but "return_corr_indices" is set.')

        # self.metadata = load_pickle(osp.join(self.dataset_root, 'metadata', f'{subset}.pkl'))
        self.metadata = make_dataset_kitti(dataset_root+reg_text_root,subset)
        # self.metadata = make_dataset_kitti('/Mount/Dataset/KITTI_odometry/icp_loop',subset)

   

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
        data_dict['pos_idx'] = metadata['frame0']
        data_dict['anc_idx'] = metadata['frame1']

        # ref_points = self._load_point_cloud(osp.join(self.dataset_root, metadata['pcd0']))
        # src_points = self._load_point_cloud(osp.join(self.dataset_root, metadata['pcd1']))
        transform = metadata['transform']

        if self.ground_segmentation:
            ref_points = self._load_point_cloud(osp.join(self.dataset_root, 'downsampled_xyzi_noground','%02d'%data_dict['seq_id'], '%06d.npy' % (data_dict['pos_idx'])))[:,:3]
            src_points = self._load_point_cloud(osp.join(self.dataset_root, 'downsampled_xyzi_noground','%02d'%data_dict['seq_id'], '%06d.npy' % (data_dict['anc_idx'])))[:,:3]
        else:

            ref_points = self._load_point_cloud(osp.join(self.dataset_root, 'downsampled_xyzi','%02d'%data_dict['seq_id'], '%06d.npy' % (data_dict['pos_idx'])))[:,:3]
            src_points = self._load_point_cloud(osp.join(self.dataset_root, 'downsampled_xyzi','%02d'%data_dict['seq_id'], '%06d.npy' % (data_dict['anc_idx'])))[:,:3]
           

        # for visualization
        # ref_points_raw = self._load_point_cloud(osp.join(self.dataset_root, 'downsampled_xyzi','%02d'%data_dict['seq_id'], '%06d.npy' % (data_dict['pos_idx'])))[:,:3]
        # src_points_raw = self._load_point_cloud(osp.join(self.dataset_root, 'downsampled_xyzi','%02d'%data_dict['seq_id'], '%06d.npy' % (data_dict['anc_idx'])))[:,:3]

        # ref_points_raw = self._load_point_cloud(osp.join(self.dataset_root, 'downsampled_xyzi_noground','%02d'%data_dict['seq_id'], '%06d.npy' % (data_dict['pos_idx'])))[:,:3]
        # src_points_raw = self._load_point_cloud(osp.join(self.dataset_root, 'downsampled_xyzi_noground','%02d'%data_dict['seq_id'], '%06d.npy' % (data_dict['anc_idx'])))[:,:3]
    
        
        # pos_aug_rotation=None
        # anc_aug_rotation=None
        # if self.use_augmentation:
        #     ref_points, src_points, transform, pos_aug_rotation, anc_aug_rotation = self._augment_point_cloud(ref_points, src_points, transform)

        if self.return_corr_indices:
            corr_indices = get_correspondences(ref_points, src_points, transform, self.matching_radius)
            data_dict['corr_indices'] = corr_indices
        
        data_dict['ref_points'] = ref_points.astype(np.float32)
        data_dict['src_points'] = src_points.astype(np.float32)
        data_dict['ref_feats'] = np.ones((ref_points.shape[0], 1), dtype=np.float32)
        data_dict['src_feats'] = np.ones((src_points.shape[0], 1), dtype=np.float32)
        data_dict['transform'] = transform.astype(np.float32)
        # data_dict['pos_aug_rotation'] = pos_aug_rotation
        # data_dict['anc_aug_rotation'] = anc_aug_rotation
        # data_dict['ref_points_raw'] = ref_points_raw
        # data_dict['src_points_raw'] = src_points_raw


        data_dict['pass'] = False
        

        return data_dict

    def __len__(self):
        return len(self.metadata)
        # return 10
        
