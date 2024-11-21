import numpy as np
import torch
import os.path as osp

def load_kitti_gt_pair_registration(txt_root, seq):
    '''
    :param txt_root:
    :param seq
    :return: [{anc_idx: *, pos_idx: *, seq: *}]                
    '''
    dataset = []

    with open(osp.join(txt_root, '%02d'%seq), 'r') as f:
        lines_list = f.readlines()
        for i, line_str in enumerate(lines_list):


            line_splitted = line_str.split()
            anc_idx = int(line_splitted[0])
            pos_idx = int(line_splitted[1])
            trans = np.array([float(x) for x in line_splitted[2:]])
            trans = np.reshape(trans, (3, 4))    
            trans = np.vstack([trans, [0, 0, 0, 1]])

            # if anc_idx!=10 and anc_idx!=572 and anc_idx!=853:
            #     continue

            data = {'seq_id': seq, 'frame0':  pos_idx, 'frame1': anc_idx, 'transform': trans}
            # data = {'seq': seq, 'anc_idx': anc_idx, 'pos_idx': pos_idx}
            dataset.append(data)
    # dataset.pop(0)
    return dataset

import glob
def load_kitti_gt_pair_overlap_loop(dataset_root, sequence):
    '''
    :param txt_root:
    :param seq
    :return: [{anc_idx: *, pos_idx: *, seq: *}]                
    '''
    dataset = []
    fnames = glob.glob(osp.join(dataset_root, 'overlap/overlap-based_gt_pairs', '%d*.npz'%sequence))
    for file_name in fnames:
        # file_name=osp.join(dataset_root, 'overlap/overlap-based_gt_pairs/' + f'{sequence}_{i}.npz')
        data_dict=np.load(file_name)

        data = {'seq_id': data_dict['seq_id'], 'anchor_idx': data_dict['anc_idx'], 'positive_idxs':  data_dict['pos_idxs'], 'negative_idxs': data_dict['neg_idxs'], 'neg_num': data_dict['neg_num']}
        # if data_dict['neg_num']>0:
        #     continue

        dataset.append(data)


    # dataset.pop(0)
    return dataset

def load_kitti_gt_pair_distance_loop(npz_root, seq):
    '''
    :param txt_root:
    :param seq
    :return: [{anc_idx: *, pos_idx: *, seq: *}]                
    '''
    dataset = []
    ground_truth_file_name = osp.join(npz_root, '%02d.npz'%seq)

    ground_truth = np.load(ground_truth_file_name, allow_pickle='True')

    dataset = ground_truth['data']


    return dataset


def load_kitti_gt_pose(dataset_root, seq, only_poses=False):
    '''
    :param dataset_root:
    :param seq
    :return: [{anc_idx: *, pos_idx: *, seq: *}]                
     '''
    dataset = []

    velo_to_cam0 = get_velo2cam(dataset_root, seq)
    velo_to_cam0 = torch.tensor(velo_to_cam0)

    data_path = osp.join(dataset_root, 'semantic-kitti-labels/dataset/sequences/%02d/poses.txt' % seq)
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
            pose = velo_to_cam0.inverse() @ (pose @ velo_to_cam0)
            data={'seq_id': seq, 'idx': idx, 'pose':pose.float().numpy()}
            dataset.append(data)
            poses.append(pose.float().numpy())
            idx+=1
    if only_poses:
        return np.array(poses)
    # poses = dataset
    return dataset   



def get_velo2cam(dataset_root, seq):
    data_path = dataset_root + '/calib/sequences/%02d/calib.txt' % seq

    with open(data_path, 'r') as f:
        for line in f.readlines():
            _, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                calib = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
            calib = np.reshape(calib, (3, 4))    
            calib = np.vstack([calib, [0, 0, 0, 1]])
    return calib

def read_points(file_name, point_limit=None):
    points = np.load(file_name)
    if point_limit is not None and points.shape[0] > point_limit:
        indices = np.random.permutation(points.shape[0])[: point_limit]
        points = points[indices]
    return points