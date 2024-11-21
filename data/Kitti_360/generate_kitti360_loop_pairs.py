import os.path as osp

import numpy as np

import faiss
from experiments.lcrnet.config_reg import make_cfg

from experiments.lcrnet.datasets.utils.kitti360 import load_gt_pose


def generate_kitti360_loop_pairs_distance_npz(seq, root): 
    '''generate loop ground truth based on distance'''
    dataset = load_gt_pose(root, seq, True)
    poses = np.stack(dataset).copy()

    file_name = osp.join(root, '%02d'%seq)

    # test_pair_idxs = []
    index = faiss.IndexFlatL2(3)
    index.add(poses[:50, :3, 3].copy())
    data_npz=[]
    for i in range(100, len(dataset)):
        current_pose = poses[i:i+1, :3, 3].copy()
        index.add(poses[i-50:i-49, :3, 3].copy())
        # lims, D, I = index.range_search(current_pose, 10.**2)
        lims, D, I = index.range_search(current_pose, 4.**2)
        relative_poses=[]
        for j in range(lims[0], lims[1]):
            relative_pose = np.linalg.inv(poses[I[j]]) @ poses[i]
            # if j == 0:
            #     num_frames_with_loop += 1
            #     yaw_diff = RT.npto_XYZRPY(np.linalg.inv(poses[I[j]]) @ poses[i])[-1]
            #     yaw_diff = yaw_diff % (2 * np.pi)
            #     if 0.79 <= yaw_diff <= 5.5:
            #         num_frames_with_reverse_loop += 1
            relative_poses.append(relative_pose)

        if I.shape[0]>0:
            relative_poses = np.array(relative_poses)
            data = {'seq_id': seq, 'anc_idx': i, 'pos_idx': I, 'pose': relative_poses}
                # test_pair_idxs.append(data)
            data_npz.append(data)


    np.savez_compressed(
        file_name,
        data=data_npz)


def prepare_kitii360_dataset(cfg):
    root = cfg.data.dataset_360_root

    for i in [2,9]:
    
        generate_kitti360_loop_pairs_distance_npz(i, root)






def main():

    cfg = make_cfg()
    prepare_kitii360_dataset(cfg)


if __name__ == '__main__':
    main()
