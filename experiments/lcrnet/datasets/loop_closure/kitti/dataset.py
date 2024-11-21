import os.path as osp
import random

import numpy as np
import torch.utils.data

from utils.utils.pointcloud import (
    random_sample_rotation,
    random_sample_yaw,
    get_transform_from_rotation_translation,
    get_rotation_translation_from_transform,
)

from experiments.lcrnet.datasets.utils.kitti import load_kitti_gt_pose,load_kitti_gt_pair_registration, load_kitti_gt_pair_distance_loop


def make_dataset_kitti(txt_path, mode, seq=None):
        if mode == 'train':
            seq_list = [5,6,7,9]  
        elif mode == 'val':
            seq_list = [2]
        elif mode == 'test':
            seq_list = [0]
        elif mode == 'infer':
            seq_list = [8]
        else:
            raise Exception('Invalid mode.')
        
        if seq is not None:
            seq_list = seq


        if mode=='train':
            datasets = []
            for seq in seq_list:
                datasets += [load_kitti_gt_pair_distance_loop(txt_path, seq)]
            datasets = np.concatenate(datasets)
        elif mode=='test':
            datasets = []
            for seq in seq_list:
                datasets += (load_kitti_gt_pair_registration(txt_path, seq))
        else :
            datasets = []
            for seq in seq_list:

                dataset = []

                with open(osp.join(txt_path, '%02d'%seq), 'r') as f:
                    lines_list = f.readlines()
                    for i, line_str in enumerate(lines_list):

                        line_splitted = line_str.split()
                        anc_idx = int(line_splitted[0])
                        pos_idx = int(line_splitted[1])


                        data = {'seq_id': seq, 'frame0':  pos_idx, 'frame1': anc_idx}
                        # data = {'seq': seq, 'anc_idx': anc_idx, 'pos_idx': pos_idx}
                        dataset.append(data)

                datasets += [dataset]
            datasets = np.concatenate(datasets)
            # datasets=datasets[:1000]


        return datasets


class OdometryKittiDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root,
        lc_text_root,
        subset,
        seq=None,
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
        super(OdometryKittiDataset, self).__init__()

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

        self.metadata = make_dataset_kitti(self.dataset_root+lc_text_root, subset, seq)
        
        self.ground_segmentation = ground_segmentation

        
        # self.position = get_position('/mnt/Mount/Dataset/KITTI_odometry',subset)

        

    def _augment_point_cloud(self, ref_points, src_points, transform):
        rotation, translation = get_rotation_translation_from_transform(transform)
        # add gaussian noise
        ref_points = ref_points + (np.random.rand(ref_points.shape[0], 3) - 0.5) * self.augmentation_noise
        src_points = src_points + (np.random.rand(src_points.shape[0], 3) - 0.5) * self.augmentation_noise
        # random rotation
        # aug_rotation = random_sample_rotation(self.augmentation_rotation)
        aug_rotation = random_sample_yaw(self.augmentation_rotation)
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
        if self.subset == 'train':
            data_dict['seq_id'] = metadata['seq_id']
            data_dict['anc_idx'] = metadata['anc_idx']
            ind = np.random.choice(np.arange(len(metadata['pos_idx'])))
            data_dict['pos_idx'] = metadata['pos_idx'][ind]
            transform = metadata['pose'][ind]
        elif self.subset == 'infer':
            data_dict['seq_id'] = metadata['seq_id']
            data_dict['anc_idx'] = metadata['frame1']
            data_dict['pos_idx'] = metadata['frame0']
        else:
            data_dict['seq_id'] = metadata['seq_id']
            data_dict['anc_idx'] = metadata['frame1']
            data_dict['pos_idx'] = metadata['frame0']
            transform = metadata['transform']
        
        if self.ground_segmentation:
            pos_points = self._load_point_cloud(osp.join(self.dataset_root, 'downsampled_xyzi_noground','%02d'%data_dict['seq_id'], '%06d.npy' % (data_dict['pos_idx'])))[:,:3]
            anc_points = self._load_point_cloud(osp.join(self.dataset_root, 'downsampled_xyzi_noground','%02d'%data_dict['seq_id'], '%06d.npy' % (data_dict['anc_idx'])))[:,:3]
        else:
            anc_points = self._load_point_cloud(osp.join(self.dataset_root, 'downsampled_xyzi','%02d'%data_dict['seq_id'], '%06d.npy' % (data_dict['anc_idx'])))[:,:3]
            pos_points = self._load_point_cloud(osp.join(self.dataset_root, 'downsampled_xyzi','%02d'%data_dict['seq_id'], '%06d.npy' % (data_dict['pos_idx'])))[:,:3]

        
        if self.use_augmentation:
            pos_points, anc_points, transform = self._augment_point_cloud(pos_points, anc_points, transform)


        if self.subset != 'infer':
            data_dict['transform'] = transform.astype(np.float32)


        data_dict['ref_points'] = pos_points.astype(np.float32)
        data_dict['src_points'] = anc_points.astype(np.float32)
        # data_dict['ref_points_raw'] = pos_points.astype(np.float32)
        # data_dict['src_points_raw'] = anc_points.astype(np.float32)
        data_dict['ref_feats'] = np.ones((pos_points.shape[0], 1), dtype=np.float32)
        data_dict['src_feats'] = np.ones((anc_points.shape[0], 1), dtype=np.float32)

        return data_dict

    def __len__(self):
        return len(self.metadata)