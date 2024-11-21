import os.path as osp

import numpy as np
import faiss

from experiments.lcrnet.datasets.utils.ford import load_ford_gt_pose

from experiments.lcrnet.config_reg import make_cfg


def transfer_distance(seq, root): 
    '''generate loop ground truth based on distance'''
    dataset = load_ford_gt_pose(root, seq, True)
    poses = np.stack(dataset).copy()

    f = open(osp.join(root, '%02d'%seq),'a')

    # test_pair_idxs = []
    index = faiss.IndexFlatL2(3)
    index.add(poses[:50, :3, 3].copy())
    for i in range(100, len(dataset)):
        current_pose = poses[i:i+1, :3, 3].copy()
        index.add(poses[i-50:i-49, :3, 3].copy())
        # lims, D, I = index.range_search(current_pose, 10.**2)
        lims, D, I = index.range_search(current_pose, 4.**2)
        for j in range(lims[0], lims[1]):
            relative_pose = np.linalg.inv(poses[i]) @ poses[I[j]]
            # if j == 0:
            #     num_frames_with_loop += 1
            #     yaw_diff = RT.npto_XYZRPY(np.linalg.inv(poses[I[j]]) @ poses[i])[-1]
            #     yaw_diff = yaw_diff % (2 * np.pi)
            #     if 0.79 <= yaw_diff <= 5.5:
            #         num_frames_with_reverse_loop += 1
            # data = {'seq_id': seq, 'idx': i, 'pose': relative_pose, 'anc_idx': I[j], 'pos_idx': i}
            # test_pair_idxs.append(data)


            M2 = relative_pose.reshape(-1)[:12]

            f.write(f'{I[j]} {i} {M2[0]:.6f} {M2[1]:.6f} {M2[2]:.6f} {M2[3]:.6f} {M2[4]:.6f} {M2[5]:.6f} {M2[6]:.6f} {M2[7]:.6f} {M2[8]:.6f} {M2[9]:.6f} {M2[10]:.6f} {M2[11]:.6f} \n')





def eval_one_epoch(cfg):
    root = cfg.data.dataset_ford_root

    for i in [0]:
    
        # transfer2(i)
        transfer_distance(i, root)

    # transfer_distance_npz(2)





def main():

    cfg = make_cfg()
    eval_one_epoch(cfg)


if __name__ == '__main__':
    main()
