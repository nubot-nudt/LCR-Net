import numpy as np
import torch
import os.path as osp

def load_gt_pair_registration(txt_root, seq):
    dataset = []

    with open(osp.join(txt_root, '%04d'%seq), 'r') as f:
        lines_list = f.readlines()
        for i, line_str in enumerate(lines_list):

            line_splitted = line_str.split()
            anc_idx = int(line_splitted[0])
            pos_idx = int(line_splitted[1])
            trans = np.array([float(x) for x in line_splitted[2:]])
            trans = np.reshape(trans, (3, 4))    
            trans = np.vstack([trans, [0, 0, 0, 1]])

            data = {'seq_id': seq, 'frame0':  pos_idx, 'frame1': anc_idx, 'transform': trans}
            dataset.append(data)
    return dataset

def load_gt_pose(dataset_root, seq, only_poses=False):
    '''
    :param dataset_root:
    :param seq
    :return: [{anc_idx: *, pos_idx: *, seq: *}]                
     '''
    dataset = []

    seq_str ='2013_05_28_drive_%04d_sync'%seq

    cam0_to_velo = get_cam2velo(dataset_root, seq_str)
    cam0_to_velo = torch.tensor(cam0_to_velo)

    data_path = osp.join(dataset_root, 'data_poses', seq_str, 'cam0_to_world.txt')
    idx=0
    poses=[]
    idxs=[]
    with open(data_path, 'r') as f:
        for x in f:
            x = x.strip().split()
            x = [float(v) for v in x]
            pose = torch.zeros((4, 4), dtype=torch.float64)
            pose[0, 0:4] = torch.tensor(x[1:5])
            pose[1, 0:4] = torch.tensor(x[5:9])
            pose[2, 0:4] = torch.tensor(x[9:13])
            pose[3, 3] = 1.0
            pose = pose @ cam0_to_velo.inverse()
            data={'seq_id': seq, 'idx': np.int64(x[0]), 'pose':pose.float().numpy()}
            dataset.append(data)
            poses.append(pose.float().numpy())
            idxs.append(np.int64(x[0]))
            idx+=1
    if only_poses:
        return np.array(poses), np.array(idxs)
    # poses = dataset
    return dataset



def get_cam2velo(dataset_root, seq):
    data_path = dataset_root + '/calibration/calib_cam_to_velo.txt' 

    with open(data_path, 'r') as f:
        for line in f.readlines():
            data = np.array([float(x) for x in line.split()])

        cam0_to_velo = np.reshape(data, (3, 4))
        cam0_to_velo = np.vstack([cam0_to_velo, [0, 0, 0, 1]])
    return cam0_to_velo

def read_points(file_name, point_limit=None):
    points = np.load(file_name)
    if point_limit is not None and points.shape[0] > point_limit:
        indices = np.random.permutation(points.shape[0])[: point_limit]
        points = points[indices]
    return points