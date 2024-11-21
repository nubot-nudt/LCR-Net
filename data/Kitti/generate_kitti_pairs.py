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
        # self.DATA_FILES = [0,1,2,3,4,5,6,7,8,9,10]

        # self.DATA_FILES = [8,9,10]
        self.DATA_FILES = [10]
        
        self.root = opt.data.dataset_root

        self.IS_ODOMETRY = True
        self.kitti_cache={}

    def __len__(self):
        return len(self.dataset)
    
    def get_video_odometry(self, drive, indices=None, ext='.txt', return_all=False):
        if self.IS_ODOMETRY:
            data_path = self.root + '/poses/%02d.txt' % drive
            if data_path not in self.kitti_cache:
                self.kitti_cache[data_path] = np.genfromtxt(data_path)
            if return_all:
                return self.kitti_cache[data_path]
            else:
                return self.kitti_cache[data_path][indices]
    
    def get_cam2velo(self, drive):
        data_path = self.root + '/calib/sequences/%02d/calib.txt' % drive

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
        
    def odometry_to_positions(self, odometry):
        if self.IS_ODOMETRY:
            T_w_cam0 = odometry.reshape(3, 4)
            T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
            return T_w_cam0

    def _get_velodyne_fn(self, drive, t):
        if self.IS_ODOMETRY:
            fname = self.root + '/sequences/%02d/velodyne/%06d.bin' % (drive, t)
        return fname

    
    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    def prepare_kitti_pairs(self, thres):
        print('process thres=%d'%thres)
        for seq in self.DATA_FILES:
  
            fnames = glob.glob(self.root + '/sequences/%02d/velodyne/*.bin' % seq)
            assert len(fnames) > 0, f"Make sure that the path {self.root} has data {seq}"

            inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            # get one-to-one distance by comparing the translation vector
            all_odo = self.get_video_odometry(seq, return_all=True)
            all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
            cam2velo = self.get_cam2velo(seq)
            Ts = all_pos[:, :3, 3]
            pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
            pdist = np.sqrt(pdist.sum(-1)) 

            ######################################
            # D3Feat script to generate test pairs
            more_than_10 = pdist > thres
            # more_than_10 = pdist > 20
            curr_time = inames[0]

            icp_dir = os.path.join(self.root,'icp%d'%thres)
            if not os.path.exists(icp_dir):
                os.makedirs(icp_dir)
            
            f = open(os.path.join(icp_dir, '%02d'%seq),'a')

            while curr_time in inames:
                next_time = np.where(more_than_10[curr_time][curr_time:curr_time + 100])[0]
                if len(next_time) == 0:
                    curr_time += 1
                else:
                    next_time = next_time[0] + curr_time - 1

                if next_time in inames:

                    cam2velo = self.get_cam2velo(seq)
                    # cam2velo = np.linalg.inv(velo2cam)
                    all_odometry = self.get_video_odometry(seq, [curr_time, next_time])
                    positions = [self.odometry_to_positions(odometry) for odometry in all_odometry]

                    fname0 = self._get_velodyne_fn(seq, curr_time)
                    fname1 = self._get_velodyne_fn(seq, next_time)

                    # XYZ and reflectance
                    xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
                    xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)

                    xyz0 = xyzr0[:, :3]
                    xyz1 = xyzr1[:, :3]
                    
                    M = np.linalg.inv(cam2velo)@ np.linalg.inv(positions[1])@ positions[0]@(cam2velo)

                    xyz0_t = self.apply_transform(xyz0, M)

                    pcd0 = to_o3d_pcd(xyz0_t)
                    pcd1 = to_o3d_pcd(xyz1)

                    vis_animation = True
                    if vis_animation:
                        vis = o3d.visualization.Visualizer()
                        vis.create_window()
                        vis.add_geometry(pcd0)
                        vis.add_geometry(pcd1)

                    

                    o3d.visualization.draw_geometries([pcd0.paint_uniform_color([1, 0, 0])+pcd1.paint_uniform_color([0, 1, 0])])
                    reg = o3d.pipelines.registration.registration_icp(pcd0, pcd1, 0.5, np.eye(4),
                                                            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000))
                    M2 = reg.transformation @  M


                    # xyz0_t = self.apply_transform(xyz0_t, reg.transformation)
                    xyz0_t = self.apply_transform(xyz0, M2)
                    pcd01 = to_o3d_pcd(xyz0_t)
                    o3d.visualization.draw_geometries([pcd01.paint_uniform_color([1, 0, 0])+pcd1.paint_uniform_color([0, 1, 0])])

                    # # False register from orignial FCGF
                    # xyz0_t2 = self.apply_transform(xyz0, M @ reg.transformation)
                    # pcd02 = to_o3d_pcd(xyz0_t2)
                    # o3d.visualization.draw_geometries([pcd02.paint_uniform_color([1, 0, 0])+pcd1.paint_uniform_color([0, 1, 0])])
                    # o3d.visualization.draw_geometries([pcd0.paint_uniform_color([1, 0, 0])+pcd1.paint_uniform_color([0, 1, 0])])
                    # o3d.visualization.draw_geometries([pcd01.paint_uniform_color([1, 0, 0])+pcd1.paint_uniform_color([0, 1, 0])])

                    # with open(os.path.join(self.root,'icp3','%02d'% seq),'a') as f:
                    #     f.write(f'{curr_time} {next_time} {M2[0]:.6f} \n')
                    # f.close()

                    M2=M2.reshape(-1)[:12]
                    curr_time = next_time + 1
                    print(curr_time)
                    
            f.close()



if __name__ == '__main__':

    # torch.multiprocessing.set_start_method('spawn')


    cfg = make_cfg()
    train_set = SparseDataset(cfg)
    
    train_set.prepare_kitti_pairs(10)