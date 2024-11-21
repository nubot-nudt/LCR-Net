import os.path as osp
import random
import time
import numpy as np
import torch.utils.data
import open3d as o3d

from utils.utils.pointcloud import (
    random_sample_yaw
)
from experiments.lcrnet.datasets.utils.kitti import load_kitti_gt_pose, load_kitti_gt_pair_overlap_loop, read_points


def make_dataset_kitti(txt_path, mode, pre_load=False,dataset_root=None,seq=None):
        if mode == 'train':
            seq_list = [3,4,5,6,7,8,9]
        elif mode == 'val':
            seq_list = [2]
        elif mode == 'test':
            seq_list = [0]
        elif mode == 'all':
            seq_list = [0,2,3,4,5,6,7,8,9]
        elif mode == 'infer':
            seq_list = [0]
        else:
            raise Exception('Invalid mode.')

        if seq is not None:
            seq_list = seq
        
        datasets = []
        poses = []

        if mode  == 'train':
            poses = {}
            for seq in seq_list:
                dataset = load_kitti_gt_pair_overlap_loop(txt_path, seq)
                datasets += (dataset)
                pose = load_kitti_gt_pose(txt_path, seq,True)
                poses['%d'%seq] = pose
        else:
            for seq in seq_list:
                dataset = load_kitti_gt_pose(txt_path, seq)
                datasets += (dataset)

        feats_dict=None
        points_dict=None
        if pre_load:
            feats_dict = {}
            points_dict = {}

            for seq in seq_list:
                points = []
                feats = []

                data_path = osp.join(dataset_root,'semantic-kitti-labels/dataset/sequences/%02d/poses.txt' % seq)
                idx=0
                with open(data_path, 'r') as f:
                    for _ in f:
                        file_name = osp.join(dataset_root, 'downsampled_xyzi','%02d'%seq, '%06d.npy' % idx)
                        points_c = read_points(file_name)

                        points.append(points_c)
                        idx+=1
                points_dict['%d'%seq] = (points)
                print('load seq ', seq)

        return datasets, poses, points_dict


class OdometryKittiPairDataset(torch.utils.data.Dataset):
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
        seq=None
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
        if self.return_corr_indices and self.matching_radius is None:
            raise ValueError('"matching_radius" is None but "return_corr_indices" is set.')


        self.point_dict={}
        self.pre_load=pre_load
        self.metadata, self.poses, self.point_dict = make_dataset_kitti(self.dataset_root,subset,pre_load, dataset_root, seq)
        


        self.pos_num=pos_num
        self.neg_num=neg_num

        self.ground_segmentation=ground_segmentation
        

    
    def _augment_point_cloud(self, ref_points, transform=None):
        if transform is None:
            transform=np.asarray([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        # rotation, translation = get_rotation_translation_from_transform(transform)
        # add gaussian noise
        ref_points = ref_points + (np.random.rand(ref_points.shape[0], 3) - 0.5) * self.augmentation_noise
        # random rotation
        # aug_rotation = random_sample_rotation(self.augmentation_rotation)
        aug_rotation = random_sample_yaw(self.augmentation_rotation)
        ref_points = np.matmul(ref_points, aug_rotation.T)

        # random scaling
        scale = random.random()
        scale = self.augmentation_min_scale + (self.augmentation_max_scale - self.augmentation_min_scale) * scale
        ref_points = ref_points * scale

        # random shift
        ref_shift = np.random.uniform(-self.augmentation_shift, self.augmentation_shift, 3)
        ref_points = ref_points + ref_shift
        return ref_points

    def _load_point_cloud(self, seq_id, idx):
        if self.pre_load:
            points = self.point_dict['%d'%seq_id][idx]
        else:
            file_name = osp.join(self.dataset_root, 'downsampled_xyzi','%02d'%seq_id, '%06d.npy' % idx)
            points = np.load(file_name)
            if self.point_limit is not None and points.shape[0] > self.point_limit:
                indices = np.random.permutation(points.shape[0])[: self.point_limit]
                points = points[indices]
        return points
    
    

    def __getitem__(self, index):
        data_dict = {}
        metadata = self.metadata[index]


        if self.subset == 'all':
            pos_num=self.pos_num
            neg_num=self.neg_num

            data_dict['idx'] = idx = int(metadata['idx'])
            data_dict['seq_id'] = int(metadata['seq_id'])

            ''' pre-extract feature'''
            if self.ground_segmentation:
                anc_points = self._load_point_cloud(osp.join(self.dataset_root, 'downsampled_xyzi_noground','%02d'%data_dict['seq_id'], '%06d.npy' % (idx)))[:,:3]
            else:
                anc_points = self._load_point_cloud(data_dict['seq_id'], idx)[:,:3]

            anc_lengths = np.int64([anc_points.shape[0]])
            anc_feats = np.ones((anc_points.shape[0], 1), dtype=np.float32)

            data_dict['anc_points'] = anc_points.astype(np.float32)
            data_dict['anc_lengths'] = anc_lengths
            data_dict['pos_num'] = pos_num
            data_dict['neg_num'] = neg_num
            data_dict['anc_feats'] = anc_feats.astype(np.float32)
            data_dict['pass']=False



        elif self.subset == 'train':
            pos_num=self.pos_num
            neg_num=self.neg_num

            anchor_idx = int(metadata['anchor_idx'])
            data_dict['seq_id'] = int(metadata['seq_id'])

            # debug
            data_dict['idx'] = index

            positive_idx = (np.random.choice(metadata['positive_idxs'],pos_num,replace=True))
            if metadata['neg_num']<neg_num:
                
                all_idx = np.arange(len(self.poses['%d'%data_dict['seq_id']]))
                pos_idx = metadata['positive_idxs'].astype(int)
                neg_mask = np.isin(all_idx,pos_idx)
                neg_idx = all_idx[~neg_mask]
                add_neg_idx = (np.random.choice(neg_idx,neg_num-metadata['neg_num'],replace=False))

                negative_idx = np.concatenate([metadata['negative_idxs'].astype(int),add_neg_idx])
            else:
                negative_idx = (np.random.choice(metadata['negative_idxs'],neg_num,replace=False))

            anc_points = self._augment_point_cloud(self._load_point_cloud(data_dict['seq_id'], anchor_idx)[:,:3])
            pos_points = [self._augment_point_cloud(self._load_point_cloud(data_dict['seq_id'], int(positive_idx[i]))[:,:3]) for i in range(pos_num)]
            neg_points = [self._augment_point_cloud(self._load_point_cloud(data_dict['seq_id'], int(negative_idx[i]))[:,:3]) for i in range(neg_num)]
            pos_lengths = np.int64([points.shape[0] for points in pos_points])
            pos_points = np.concatenate(pos_points, axis=0)
            neg_lengths = np.int64([points.shape[0] for points in neg_points])
            neg_points = np.concatenate(neg_points, axis=0)
            anc_lengths = np.int64([anc_points.shape[0]])
            anc_feats = np.ones((anc_points.shape[0], 1), dtype=np.float32)
            pos_feats = np.ones((pos_points.shape[0], 1), dtype=np.float32)
            neg_feats = np.ones((neg_points.shape[0], 1), dtype=np.float32)

            
            
            data_dict['anc_points'] = anc_points.astype(np.float32)
            data_dict['pos_points'] = pos_points.astype(np.float32)
            data_dict['neg_points'] = neg_points.astype(np.float32)
            data_dict['anc_lengths'] = anc_lengths
            data_dict['pos_lengths'] = pos_lengths
            data_dict['neg_lengths'] = neg_lengths
            data_dict['pos_num'] = pos_num
            data_dict['neg_num'] = neg_num
            data_dict['anc_feats'] = anc_feats.astype(np.float32)
            data_dict['pos_feats'] = pos_feats.astype(np.float32)
            data_dict['neg_feats'] = neg_feats.astype(np.float32)


        else:
            
            data_dict['seq_id'] = metadata['seq_id']
            data_dict['idx'] = metadata['idx']
            idx=data_dict['idx']
            seq_id=data_dict['seq_id']

            
            '''online test'''
            anc_points = self._load_point_cloud(data_dict['seq_id'], int(data_dict['idx']))[:,:3]
            anc_lengths = np.int64([anc_points.shape[0]])
            anc_feats = np.ones((anc_points.shape[0], 1), dtype=np.float32)


            data_dict['anc_points'] = anc_points.astype(np.float32)
            data_dict['anc_lengths'] = anc_lengths
            data_dict['anc_feats'] = anc_feats.astype(np.float32)


        return data_dict

    def __len__(self):
        return len(self.metadata)
        
def to_o3d_pcd(xyz):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(to_array(xyz))
    return pcd

def to_array(tensor):
    """
    Conver tensor to array
    """
    if(not isinstance(tensor,np.ndarray)):
        if(tensor.device == torch.device('cpu')):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor

def apply_transform( pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts