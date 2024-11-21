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
        self.DATA_FILES = [0,2,3,4,5,6,7,9,10]
        
        self.root_path = opt.data.dataset_360_root

        self.IS_ODOMETRY = True
        self.kitti_cache={}

    def __len__(self):
        return len(self.dataset)
    
    def get_video_odometry(self, drive, indices=None, ext='.txt', return_all=False):
        if self.IS_ODOMETRY:
            data_path = self.root_path + '/data_poses/2013_05_28_drive_%04d_sync/cam0_to_world.txt' % drive
            if data_path not in self.kitti_cache:
                self.kitti_cache[data_path] = np.genfromtxt(data_path)
            if return_all:
                return self.kitti_cache[data_path]
            else:
                return self.kitti_cache[data_path][indices]
    
    def get_cam2velo(self, drive):
        data_path = self.root_path + '/calibration/calib_cam_to_velo.txt' 

        calib = np.genfromtxt(data_path)
        
        calib = calib.reshape(3, 4)
        calib = np.vstack((calib, [0, 0, 0, 1]))
        return calib
        
    def odometry_to_positions(self, odometry):
        scan_idx = int(odometry[0])

        T_w_cam0 = odometry[1:].reshape(4, 4)
        return scan_idx, T_w_cam0

    def _get_velodyne_fn(self, drive, t):
        if self.IS_ODOMETRY:
            fname = self.root_path + '/data_3d_raw/2013_05_28_drive_%04d_sync/velodyne_points/data/%010d.bin' % (drive, t)
        return fname

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    def prepare_kitti360_pairs(self):
        '''
        '''
        for seq in self.DATA_FILES:
            all_odo = self.get_video_odometry(seq, return_all=True)

            all_pos = np.array([self.odometry_to_positions(odo)[1] for odo in all_odo])
            # scan_idx = np.array([self.odometry_to_positions(odo)[0] for odo in all_odo])

            cam2velo = self.get_cam2velo(seq)
            Ts = all_pos[:, :3, 3]
            pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
            pdist = np.sqrt(pdist.sum(-1)) 

            inames = range(all_pos.shape[0])

            ######################################
            # D3Feat script to generate test pairs
            more_than_10 = pdist > 10

            curr_time = inames[0]
            f = open(os.path.join(self.root_path,'raw10_for_odom','%02d'% seq),'a')
            while curr_time in inames:
                next_time = np.where(more_than_10[curr_time][curr_time:curr_time + 100])[0]
                if len(next_time) == 0:
                    curr_time += 1
                else:
                    next_time = next_time[0] + curr_time - 1

                if next_time in inames:
                    

                    all_odometry = self.get_video_odometry(seq, [curr_time, next_time])
                    positions = [self.odometry_to_positions(odometry)[1] for odometry in all_odometry]
                    scan_idx = [self.odometry_to_positions(odometry)[0] for odometry in all_odometry]

                    fname0 = self._get_velodyne_fn(seq, scan_idx[0])
                    fname1 = self._get_velodyne_fn(seq, scan_idx[1])
                    # XYZ and reflectance
                    xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
                    xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)

                    xyz0 = xyzr0[:, :3]
                    xyz1 = xyzr1[:, :3]
                
                    M = (cam2velo @ np.linalg.inv(positions[1]) @ positions[0]
                            @ np.linalg.inv(cam2velo))
                    xyz0_t = self.apply_transform(xyz0, M)


                    pcd0 = to_o3d_pcd(xyz0_t)
                    pcd1 = to_o3d_pcd(xyz1)
                    # o3d.visualization.draw_geometries([pcd0.paint_uniform_color([1, 0, 0])+pcd1.paint_uniform_color([0, 1, 0])])
                    reg = o3d.pipelines.registration.registration_icp(pcd0, pcd1, 0.2, np.eye(4),
                                                            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
                    # pcd0.transform(reg.transformation)
                    # M3 = torch.einsum('ki,ij->kj', torch.tensor(reg.transformation, dtype=torch.float, device='cpu'), T_gt).numpy()
                    # M2 = M @ reg.transformation
                    M2 = reg.transformation @ M
                    

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
                    
                    
                    M2=M.reshape(-1)[:12]
                    f.write(f'{scan_idx[0]} {scan_idx[1]} {M2[0]:.6f} {M2[1]:.6f} {M2[2]:.6f} {M2[3]:.6f} {M2[4]:.6f} {M2[5]:.6f} {M2[6]:.6f} {M2[7]:.6f} {M2[8]:.6f} {M2[9]:.6f} {M2[10]:.6f} {M2[11]:.6f} \n')

                    curr_time = next_time + 1
                    print(seq,' ',scan_idx[0])


if __name__ == '__main__':

    cfg = make_cfg()

    dataset = SparseDataset(cfg)
    dataset.prepare_kitti360_pairs()
    