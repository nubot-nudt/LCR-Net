import numpy as np
import torch
import os.path as osp

import os, glob


def load_mulran_gt_pair_registration(txt_root, seq):
    dataset = []

    with open(osp.join(txt_root, seq), 'r') as f:
        lines_list = f.readlines()
        for i, line_str in enumerate(lines_list):

            line_splitted = line_str.split()
            anc_idx = int(line_splitted[0])
            pos_idx = int(line_splitted[1])
            trans = np.array([float(x) for x in line_splitted[2:]])
            trans = np.reshape(trans, (3, 4))    
            trans = np.vstack([trans, [0, 0, 0, 1]])

            data = {'seq_id': seq, 'frame0':  pos_idx, 'frame1': anc_idx, 'transform': trans}
            # data = {'seq': seq, 'anc_idx': anc_idx, 'pos_idx': pos_idx}
            dataset.append(data)
    # dataset.pop(0)
    return dataset


def load_mulran_gt_pose(dataset_root, seq, only_poses=False):
    '''
    :param dataset_root:
    :param seq
    :return: [{anc_idx: *, pos_idx: *, seq: *}]                
     '''
    dataset = []

    # data_path = osp.join(txt_root + '/semantic-kitti-labels/dataset/sequences/%02d/poses.txt' % seq)
    data_path = osp.join(dataset_root, 'mulran/%s/sensor_data/poses_in_kitti_format.txt' % seq )

    fnames = glob.glob(dataset_root + '/mulran/%s/sensor_data/Ouster/*.bin' % seq)
    inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

    idx=0
    poses=[]
    with open(data_path, 'r') as f:
        for x in f:
            x = x.strip().split()
            x = [float(v) for v in x]
            pose = torch.zeros((4, 4), dtype=torch.float64)
            pose[0, 0:4] = torch.tensor(x[0:4])
            pose[1, 0:4] = torch.tensor(x[4:8])
            pose[2, 0:4] = torch.tensor(x[8:12])
            pose[3, 3] = 1.0
            pose = pose 
            data={'seq_id': seq, 'idx': inames[idx], 'pose':pose.float().numpy()}
            dataset.append(data)
            poses.append(pose.float().numpy())
            idx+=1
    if only_poses:
        return np.array(poses)
    # poses = dataset
    return dataset   



def read_points(file_name, point_limit=None):
    points = np.load(file_name)
    if point_limit is not None and points.shape[0] > point_limit:
        indices = np.random.permutation(points.shape[0])[: point_limit]
        points = points[indices]
    return points