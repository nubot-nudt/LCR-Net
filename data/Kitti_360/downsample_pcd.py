import os
import os.path as osp
import open3d as o3d
import numpy as np
import glob
from tqdm import tqdm
from pathlib import Path
from experiments.lcrnet.config_reg import make_cfg

def main():
    cfg = make_cfg()
    root = cfg.data.dataset_360_root
    for i in range(7,11):
        seq_id = '{:04d}'.format(i)
        seq_path = '2013_05_28_drive_{}_sync'.format(seq_id)
        file_names = glob.glob(osp.join(root, 'data_3d_raw', seq_path, 'velodyne_points', 'data', '*.bin'))
        for file_name in tqdm(file_names):
            frame = file_name.split('/')[-1][:-4]
            new_file_name = osp.join(root,'downsampled_xyzi', seq_id, frame + '.npy')
            points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
            num=points.shape[0]
            xyz = points[:, :3]
            intensity = points[:, 3]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(intensity.reshape(num,1).repeat(3,axis=1))
            pcd = pcd.voxel_down_sample(0.3)
            xyz = np.array(pcd.points).astype(np.float32)
            intensity = np.array(pcd.colors).astype(np.float32)[:,0]

            xyzi = np.concatenate([xyz,intensity.reshape(xyz.shape[0],1)],axis=1)
            
            file_out_path = Path(osp.join(root,'downsampled_xyzi', seq_id))
            file_out_path.mkdir(exist_ok=True, parents=True)
            np.save(new_file_name, xyzi)


if __name__ == '__main__':
    main()
