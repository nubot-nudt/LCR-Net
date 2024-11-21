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
from experiments.lcrnet.datasets.utils.kitti360 import load_gt_pair_registration


def make_dataset_kitti(txt_path, mode):

        seq_list = [0,2,3,4,5,6,7,9,10]

        dataset = []
        for seq in seq_list:
            dataset += (load_gt_pair_registration(txt_path, seq))
           
        return dataset

class OdometryKitti360PairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_360_root,
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
    ):
        super(OdometryKitti360PairDataset, self).__init__()

        self.dataset_360_root = dataset_360_root
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

        # self.metadata = load_pickle(osp.join(self.dataset_root, 'metadata', f'{subset}.pkl'))
        self.metadata = make_dataset_kitti(self.dataset_360_root+reg_text_root, subset)

    def _augment_point_cloud(self, ref_points, src_points, transform):
        rotation, translation = get_rotation_translation_from_transform(transform)
        # add gaussian noise
        ref_points = ref_points + (np.random.rand(ref_points.shape[0], 3) - 0.5) * self.augmentation_noise
        src_points = src_points + (np.random.rand(src_points.shape[0], 3) - 0.5) * self.augmentation_noise
        # random rotation
        aug_rotation = random_sample_rotation(self.augmentation_rotation)
        if random.random() > 0.5:
            ref_points = np.matmul(ref_points, aug_rotation.T)
            rotation = np.matmul(aug_rotation, rotation)
            translation = np.matmul(aug_rotation, translation)
        else:
            src_points = np.matmul(src_points, aug_rotation.T)
            rotation = np.matmul(rotation, aug_rotation.T)
        # random scaling
        scale = random.random()
        scale = self.augmentation_min_scale + (self.augmentation_max_scale - self.augmentation_min_scale) * scale
        ref_points = ref_points * scale
        src_points = src_points * scale
        translation = translation * scale
        # random shift
        ref_shift = np.random.uniform(-self.augmentation_shift, self.augmentation_shift, 3)
        src_shift = np.random.uniform(-self.augmentation_shift, self.augmentation_shift, 3)
        ref_points = ref_points + ref_shift
        src_points = src_points + src_shift
        translation = -np.matmul(src_shift[None, :], rotation.T) + translation + ref_shift
        # compose transform from rotation and translation
        transform = get_transform_from_rotation_translation(rotation, translation)
        return ref_points, src_points, transform

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
        ref_points = self._load_point_cloud(osp.join(self.dataset_360_root, 'downsampled_xyzi','%04d'%data_dict['seq_id'], '%010d.npy' % (data_dict['pos_idx'])))[:,:3]
        src_points = self._load_point_cloud(osp.join(self.dataset_360_root, 'downsampled_xyzi','%04d'%data_dict['seq_id'], '%010d.npy' % (data_dict['anc_idx'])))[:,:3]
        transform = metadata['transform']


        if self.use_augmentation:
            ref_points, src_points, transform = self._augment_point_cloud(ref_points, src_points, transform)

        if self.return_corr_indices:
            corr_indices = get_correspondences(ref_points, src_points, transform, self.matching_radius)
            data_dict['corr_indices'] = corr_indices
        
        data_dict['ref_points'] = ref_points.astype(np.float32)
        data_dict['src_points'] = src_points.astype(np.float32)
        data_dict['ref_feats'] = np.ones((ref_points.shape[0], 1), dtype=np.float32)
        data_dict['src_feats'] = np.ones((src_points.shape[0], 1), dtype=np.float32)
        data_dict['transform'] = transform.astype(np.float32)
 
        return data_dict

    def __len__(self):
        return len(self.metadata)
        # return 10
        
