import numpy as np
from experiments.lcrnet.datasets.utils.kitti import load_kitti_gt_pose



def distance(dataset_root, seq, idx1, idx2):
        '''
        calculate distance between scan idx1 and idx2
        '''
        pose = load_kitti_gt_pose(dataset_root, seq)
        anc_pose = pose[idx1]['pose'][:3,3]

        
        if type(idx2).__name__ == 'ndarray':
            dis_a = []
            for i in range(len(idx2)):
                idx=int(idx2[i])
                pos_pose = pose[idx]['pose'][:3,3]

                dis = np.linalg.norm(anc_pose-pos_pose)
                dis_a.append(dis)
                # print(dis)
            dis = np.mean(dis_a)
        elif type(idx2).__name__ == 'int':
            pos_pose = pose[idx2]['pose'][:3,3]
            dis = np.linalg.norm(anc_pose-pos_pose)
        else:
            print('type error')
            return
                            

        print('distance ', dis)
        return