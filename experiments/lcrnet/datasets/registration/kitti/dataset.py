import os.path as osp
import random

import numpy as np
import torch.utils.data

from utils.utils.common import load_pickle
from utils.utils.pointcloud import (
    random_sample_rotation,
    get_transform_from_rotation_translation,
    get_rotation_translation_from_transform,
    random_sample_yaw,
    random_sample_rotation2
)
from utils.utils.registration import get_correspondences

from experiments.lcrnet.datasets.utils.kitti import load_kitti_gt_pair_registration

def make_dataset_kitti(txt_path, mode):
    if mode == 'train':
        seq_list = [0,1,2,3,4,5]
    elif mode == 'val':
        seq_list = [6, 7]
    elif mode == 'test':
        seq_list = [8,9,10]
    else:
        raise Exception('Invalid mode.')


    dataset = []
    for seq in seq_list:
        dataset += (load_kitti_gt_pair_registration(txt_path, seq))
        
    return dataset

class OdometryKittiPairDataset(torch.utils.data.Dataset):
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
        super(OdometryKittiPairDataset, self).__init__()
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

        self.metadata = make_dataset_kitti(dataset_root+reg_text_root,subset)

    def _augment_point_cloud(self, ref_points, src_points, transform):
        rotation, translation = get_rotation_translation_from_transform(transform)
        # add gaussian noise
        ref_points = ref_points + (np.random.rand(ref_points.shape[0], 3) - 0.5) * self.augmentation_noise
        src_points = src_points + (np.random.rand(src_points.shape[0], 3) - 0.5) * self.augmentation_noise
        # random rotation
        # aug_rotation = random_sample_rotation(self.augmentation_rotation)
        aug_rotation = random_sample_yaw(self.augmentation_rotation)
        # aug_rotation = random_sample_rotation2(self.augmentation_rotation)
        
        if random.random() > 0.5:
            ref_points = np.matmul(ref_points, aug_rotation.T)
            rotation = np.matmul(aug_rotation, rotation)
            translation = np.matmul(aug_rotation, translation)
            pos_aug_rotation = aug_rotation.astype(np.float32)
            anc_aug_rotation = None
        else:
            src_points = np.matmul(src_points, aug_rotation.T)
            rotation = np.matmul(rotation, aug_rotation.T)
            pos_aug_rotation = None
            anc_aug_rotation = aug_rotation.astype(np.float32)
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
        return ref_points, src_points, transform, pos_aug_rotation, anc_aug_rotation

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
        transform = metadata['transform']

        if self.ground_segmentation:
            ref_points = self._load_point_cloud(osp.join(self.dataset_root, 'downsampled_xyzi_noground','%02d'%data_dict['seq_id'], '%06d.npy' % (data_dict['pos_idx'])))[:,:3]
            src_points = self._load_point_cloud(osp.join(self.dataset_root, 'downsampled_xyzi_noground','%02d'%data_dict['seq_id'], '%06d.npy' % (data_dict['anc_idx'])))[:,:3]
        else:

            ref_points = self._load_point_cloud(osp.join(self.dataset_root, 'downsampled_xyzi','%02d'%data_dict['seq_id'], '%06d.npy' % (data_dict['pos_idx'])))[:,:3]
            src_points = self._load_point_cloud(osp.join(self.dataset_root, 'downsampled_xyzi','%02d'%data_dict['seq_id'], '%06d.npy' % (data_dict['anc_idx'])))[:,:3]
           

        # # for visualization
        # ref_points_raw = self._load_point_cloud(osp.join(self.dataset_root, 'downsampled_xyzi','%02d'%data_dict['seq_id'], '%06d.npy' % (data_dict['pos_idx'])))[:,:3]
        # src_points_raw = self._load_point_cloud(osp.join(self.dataset_root, 'downsampled_xyzi','%02d'%data_dict['seq_id'], '%06d.npy' % (data_dict['anc_idx'])))[:,:3]

        pos_aug_rotation=None
        anc_aug_rotation=None
        if self.use_augmentation:
            ref_points, src_points, transform, pos_aug_rotation, anc_aug_rotation = self._augment_point_cloud(ref_points, src_points, transform)

        if self.return_corr_indices:
            corr_indices = get_correspondences(ref_points, src_points, transform, self.matching_radius)
            data_dict['corr_indices'] = corr_indices
        
        data_dict['ref_points'] = ref_points.astype(np.float32)
        data_dict['src_points'] = src_points.astype(np.float32)
        data_dict['ref_feats'] = np.ones((ref_points.shape[0], 1), dtype=np.float32)
        data_dict['src_feats'] = np.ones((src_points.shape[0], 1), dtype=np.float32)
        data_dict['transform'] = transform.astype(np.float32)
        data_dict['pos_aug_rotation'] = pos_aug_rotation
        data_dict['anc_aug_rotation'] = anc_aug_rotation
        # data_dict['ref_points_raw'] = ref_points_raw
        # data_dict['src_points_raw'] = src_points_raw


        data_dict['pass'] = False
        

        return data_dict

    def __len__(self):
        return len(self.metadata)