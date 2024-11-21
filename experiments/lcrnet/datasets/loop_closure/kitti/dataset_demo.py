import os.path as osp
import numpy as np
import torch.utils.data

from experiments.lcrnet.datasets.utils.kitti import load_kitti_gt_pose,load_kitti_gt_pair_registration, load_kitti_gt_pair_distance_loop

def make_dataset_kitti(txt_path, seq=None):
        seq_list = [0]

        if seq is not None:
            seq_list = seq

        datasets = []
        for seq in seq_list:
            datasets += (load_kitti_gt_pair_registration(txt_path, seq))

        return datasets

class OdometryKittiDatasetDemo(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root,
        pos_frame,
        anc_frame,
        gt_text_root=None,
        seq=None,
        point_limit=None
    ):
        super(OdometryKittiDatasetDemo, self).__init__()

        self.point_limit = point_limit
        self.pos_frame = pos_frame
        self.anc_frame = anc_frame
        self.dataset_root = dataset_root
        self.metadata = None

        if gt_text_root is not None:
            self.metadata = make_dataset_kitti(gt_text_root, seq)


    def _load_point_cloud(self, file_name):
        points = np.load(file_name)
        if self.point_limit is not None and points.shape[0] > self.point_limit:
            indices = np.random.permutation(points.shape[0])[: self.point_limit]
            points = points[indices]
        return points

    def __getitem__(self, index):
        data_dict = {}


        if self.metadata is not None:
            metadata = self.metadata[index]
            anc_frame = metadata['frame1']
            pos_frame = metadata['frame0']
            seq  = metadata['seq_id']
            dataset_root = osp.join(self.dataset_root, '%02d'%seq)
            data_dict['transform'] = metadata['transform'].astype(np.float32)
        else:
            pos_frame = self.pos_frame[index]
            anc_frame = self.anc_frame[index]
            data_dict['transform'] = None

        pos_points = self._load_point_cloud(osp.join(dataset_root, '%06d.npy' % (pos_frame)))[:,:3]
        anc_points = self._load_point_cloud(osp.join(dataset_root, '%06d.npy' % (anc_frame)))[:,:3]
        
        data_dict['anc_idx'] = anc_frame
        data_dict['pos_idx'] = pos_frame
        data_dict['ref_points'] = pos_points.astype(np.float32)
        data_dict['src_points'] = anc_points.astype(np.float32)
        data_dict['ref_feats'] = np.ones((pos_points.shape[0], 1), dtype=np.float32)
        data_dict['src_feats'] = np.ones((anc_points.shape[0], 1), dtype=np.float32)

        return data_dict

    def __len__(self):
        if self.metadata is not None:
            return len(self.metadata)
        return len(self.pos_frame)