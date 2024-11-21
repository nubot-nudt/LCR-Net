import os.path as osp
import numpy as np
import torch.utils.data

import open3d as o3d
import torchvision.transforms as transforms

from experiments.lcrnet.datasets.utils.kitti import load_kitti_gt_pose, load_kitti_gt_pair_overlap_loop



def make_dataset_kitti(txt_path, mode, pre_load=False,feature_dict_dir=None):
        if mode == 'train':
            seq_list = [3,4,5,6,7,8,9]
            # seq_list = [3,4]
            # seq_list = [3,4]
        elif mode == 'val':
            seq_list = [2]
            # seq_list = range(10)
        elif mode == 'test':
            # seq_list = [3,4]
            seq_list = [0]
            # seq_list = [0,2,3,4,5,6,7,8,9]
        elif mode == 'all':
            # seq_list = [3,4]
            seq_list = [0,2,3,4,5,6,7,8,9]
        else:
            raise Exception('Invalid mode.')

        
        datasets = []
        poses = []

        # for seq in seq_list:
        #     dataset = load_kitti_gt_pair_txt(txt_path, seq, mode)
        #     datasets += (dataset)

        if mode == 'train':
            for seq in seq_list:
                dataset = load_kitti_gt_pair_overlap_loop(txt_path, seq)
                datasets += (dataset)
        else:
            for seq in seq_list:
                dataset = load_kitti_gt_pose(txt_path, seq)
                datasets += (dataset)

        feats_dict=None
        if pre_load:
            feats_dict = {}

            for seq in seq_list:
                points = []
                feats = []

                data_path = osp.join('/mnt/Mount/Dataset/KITTI_odometry/semantic-kitti-labels/dataset/sequences/%02d/poses.txt' % seq)
                idx=0
                with open(data_path, 'r') as f:
                    for _ in f:
                        feats_c, points_c = read_feature(feature_dict_dir, seq, idx)

                        feats.append(feats_c)
                        points.append(points_c)
                        # dataset.append(data)
                        # poses.append(pose.float().numpy())
                        idx+=1
                feats_dict['%d'%seq] = (feats)
                # feats_dict['%d_points'%seq] = (feats)
                
                print('load seq ', seq)

        return datasets, poses, feats_dict

def read_feature(feature_dict_dir, seq, idx):
    file_name=osp.join(feature_dict_dir, '%d'%(seq), '%d_%d.npz'%(seq, idx))
    data_dict=np.load(file_name, allow_pickle=True)


    points=data_dict['nms_shifted_points_c']
    feats=data_dict['nms_shifted_feats_c']
    return feats, points



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

        self.feature_dict_dir = osp.join(self.dataset_root, 'feature(kpconv4)')

        self.feature_dict={}
        self.pre_load=pre_load
        self.metadata, self.poses, self.feature_dict = make_dataset_kitti(self.dataset_root,subset,pre_load, self.feature_dict_dir)
        


        self.pos_num=pos_num
        self.neg_num=neg_num

        self.ground_segmentation=ground_segmentation



    def _load_point_cloud(self, file_name):
        points = np.load(file_name)
        if self.point_limit is not None and points.shape[0] > self.point_limit:
            indices = np.random.permutation(points.shape[0])[: self.point_limit]
            points = points[indices]
        return points

    def load_feature_dict(self, seq_id, idx):
        if self.pre_load:
            feats_c = self.feature_dict['%d'%seq_id][idx]
            # points_c = self.feature_dict['%d_points'%seq_id][idx]
        else:
            feats_c, _ = read_feature(self.feature_dict_dir, seq_id, idx)
        return feats_c
    

    def __getitem__(self, index):
        data_dict = {}
        metadata = self.metadata[index]


        if self.subset == 'all':
            ''' For pre-extracting features'''

            pos_num=self.pos_num
            neg_num=self.neg_num

            data_dict['idx'] = idx = int(metadata['idx'])
            data_dict['seq_id'] = int(metadata['seq_id'])

            if self.ground_segmentation:
                anc_points = self._load_point_cloud(osp.join(self.dataset_root, 'downsampled_xyzi_noground','%02d'%data_dict['seq_id'], '%06d.npy' % (idx)))[:,:3]
            else:
                anc_points = self._load_point_cloud(osp.join(self.dataset_root, 'downsampled_xyzi','%02d'%data_dict['seq_id'], '%06d.npy' % (idx)))[:,:3]

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


            '''offline train: load feature'''
            positive_idx = (np.random.choice(metadata['positive_idxs'],pos_num,replace=True))
            negative_idx = (np.random.choice(metadata['negative_idxs'],neg_num,replace=True))
            

            pos_feats=[]
            for i in range(pos_num):
                pos_feature_dict = self.load_feature_dict(data_dict['seq_id'], int(positive_idx[i]))
                pos_feats.append(pos_feature_dict)


            neg_feats=[]
            for i in range(neg_num):
                neg_feature_dict = self.load_feature_dict(data_dict['seq_id'], int(negative_idx[i]))
                neg_feats.append(neg_feature_dict)


            anc_feature_dict = self.load_feature_dict(data_dict['seq_id'], int(anchor_idx))
            anc_feats = anc_feature_dict
            

            pos_lengths = np.int64([points.shape[0] for points in pos_feats])
            # pos_points = np.concatenate(pos_points, axis=0)
            neg_lengths = np.int64([points.shape[0] for points in neg_feats])
            # neg_points = np.concatenate(neg_points, axis=0)
            anc_lengths = np.int64([anc_feats.shape[0]])
            pos_feats = np.concatenate(pos_feats, axis=0)
            neg_feats = np.concatenate(neg_feats, axis=0)
            # Use regular expression to extract numbers from a string



            data_dict['pos_num'] = pos_num
            data_dict['neg_num'] = neg_num
            data_dict['anc_lengths'] = anc_lengths
            data_dict['pos_lengths'] = pos_lengths
            data_dict['neg_lengths'] = neg_lengths
            data_dict['anc_feats'] = anc_feats.astype(np.float32)
            data_dict['pos_feats'] = pos_feats.astype(np.float32)
            data_dict['neg_feats'] = neg_feats.astype(np.float32)
          
        else:
            
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