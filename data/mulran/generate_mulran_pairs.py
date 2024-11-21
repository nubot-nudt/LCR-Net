import numpy as np
import os, glob
from utils.utils.open3d import to_o3d_pcd
import open3d as o3d
from torch.utils.data import Dataset

from experiments.lcrnet.config_reg import make_cfg




class SparseDataset(Dataset):
    """Sparse correspondences dataset.  
    Reads images from files and creates pairs. It generates keypoints, 
    descriptors and ground truth matches which will be used in training."""

    def __init__(self, opt):
        # self.DATA_FILES = ['riveside01','sejong01']
        self.DATA_FILES = ['sejong01']
        
        self.root = opt.data.mulran_root

        self.IS_ODOMETRY = True
        self.kitti_cache={}

    def __len__(self):
        return len(self.dataset)
    
    def get_video_odometry(self, drive, indices=None, ext='.txt', return_all=False):
        if self.IS_ODOMETRY:
            # data_path = self.root + '/mulran/kaist/kaist%02d/poses_in_kitti_format.txt' % drive
            data_path = self.root + '/{}/sensor_data/poses_in_kitti_format.txt' .format(drive) 
            if data_path not in self.kitti_cache:
                self.kitti_cache[data_path] = np.genfromtxt(data_path)
            if return_all:
                return self.kitti_cache[data_path]
            else:
                return self.kitti_cache[data_path][indices]
        
    def odometry_to_positions(self, odometry):
        if self.IS_ODOMETRY:
            T_w_cam0 = odometry.reshape(3, 4)
            T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
            return T_w_cam0

    def _get_velodyne_fn(self, drive, t):
        if self.IS_ODOMETRY:
            fname = self.root + '/%s/sensor_data/Ouster/%d.bin' % (drive, t)
        return fname

    def get_position_transform(self, pos0, pos1, invert=False):
        T0 = self.pos_transform(pos0)
        T1 = self.pos_transform(pos1)
        return (np.dot(T1, np.linalg.inv(T0)).T if not invert else np.dot(
            np.linalg.inv(T1), T0).T)
    
    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    def prepare_mulran_pairs(self):
        '''
        sch
        10m以内最远的scan作为pair
        舍弃中间的scan
        @todo 随机的1-10m？ 5-15m？
        '''
        for seq in self.DATA_FILES:

            fnames = glob.glob(self.root + '/%s/sensor_data/Ouster/*.bin' % seq)
            assert len(fnames) > 0, f"Make sure that the path {self.root} has data {seq}"

            inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            # get one-to-one distance by comparing the translation vector
            all_odo = self.get_video_odometry(seq, return_all=True)
            all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
            Ts = all_pos[:, :3, 3]
            pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
            pdist = np.sqrt(pdist.sum(-1)) 

            ######################################
            # D3Feat script to generate test pairs
            more_than_10 = pdist > 10
            curr_time = 0
            f = open(os.path.join(self.root,'raw10_for_odom',seq),'a')
            while curr_time in range(len(inames)):
                # dis=5+np.random.rand(1)*10
                # more_than_10 = pdist > dis
                next_time = np.where(more_than_10[curr_time][curr_time:curr_time + 100])[0]
                if len(next_time) == 0:
                    curr_time += 1
                else:
                    next_time = next_time[0] + curr_time - 1
                
                if next_time in range(len(inames)):

                    all_odometry = self.get_video_odometry(seq, [curr_time, next_time])
                    positions = [self.odometry_to_positions(odometry) for odometry in all_odometry]


                    M = (positions[0].T @ np.linalg.inv(positions[1].T)).T


                    '''perform  ICP'''
                    # fname0 = self._get_velodyne_fn(seq, inames[curr_time])
                    # fname1 = self._get_velodyne_fn(seq, inames[next_time])
                    # xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
                    # xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)
                    # xyz0 = xyzr0[:, :3]
                    # xyz1 = xyzr1[:, :3]
                    # xyz0_t = self.apply_transform(xyz0, M)

                    # pcd0 = to_o3d_pcd(xyz0_t)
                    # pcd1 = to_o3d_pcd(xyz1)
                    # o3d.visualization.draw_geometries([pcd0.paint_uniform_color([1, 0, 0])+pcd1.paint_uniform_color([0, 1, 0])])
                    # reg = o3d.pipelines.registration.registration_icp(pcd0, pcd1, 0.2, np.eye(4),
                    #                                         o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    #                                         o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
                    # # pcd0.transform(reg.transformation)
                    # # M3 = torch.einsum('ki,ij->kj', torch.tensor(reg.transformation, dtype=torch.float, device='cpu'), T_gt).numpy()
                    # M2 = reg.transformation @  M

                    '''visualization'''
                    # xyz0_t = self.apply_transform(xyz0, M2)
                    # pcd01 = to_o3d_pcd(xyz0_t)
                    # pcd1 = to_o3d_pcd(xyz1)
                    # # o3d.visualization.draw_geometries([pcd0.paint_uniform_color([1, 0, 0])+pcd1.paint_uniform_color([0, 1, 0])])
                    # xyz0_t2 = self.apply_transform(xyz0, M @ reg.transformation)
                    # pcd02 = to_o3d_pcd(xyz0_t2)
                    # o3d.visualization.draw_geometries([pcd0.paint_uniform_color([1, 0, 0])+pcd1.paint_uniform_color([0, 1, 0])])
                    # o3d.visualization.draw_geometries([pcd01.paint_uniform_color([1, 0, 0])+pcd1.paint_uniform_color([0, 1, 0])])
                    # o3d.visualization.draw_geometries([pcd02.paint_uniform_color([1, 0, 0])+pcd1.paint_uniform_color([0, 1, 0])])

                    # M2=M2.reshape(-1)[:12]
                    M2=M.reshape(-1)[:12]

                    # with open(os.path.join(self.root,'icp3','%02d'% seq),'a') as f:
                    #     f.write(f'{curr_time} {next_time} {M2[0]:.6f} \n')
                    # f.close()
                    f.write(f'{inames[curr_time]} {inames[next_time]} {M2[0]:.6f} {M2[1]:.6f} {M2[2]:.6f} {M2[3]:.6f} {M2[4]:.6f} {M2[5]:.6f} {M2[6]:.6f} {M2[7]:.6f} {M2[8]:.6f} {M2[9]:.6f} {M2[10]:.6f} {M2[11]:.6f} \n')

                    curr_time = next_time + 1

                    print(curr_time)
            f.close()



if __name__ == '__main__':
    cfg = make_cfg()


    train_set = SparseDataset(cfg)
    train_set.prepare_mulran_pairs()
    