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
    root = cfg.data.apollo_root
    for i in range(1,5):
        seq_id = '{:02d}'.format(i)
        file_out_path = Path(osp.join(root,'downsampled_xyzi', seq_id))
        file_out_path.mkdir(exist_ok=True, parents=True)
        file_names = glob.glob(osp.join(root,'kitti_format/MapData/ColumbiaPark/2018-09-21', seq_id, 'velodyne', '*.bin'))
        for file_name in tqdm(file_names):
            frame = file_name.split('/')[-1][:-4]
            new_file_name = osp.join(root,'downsampled_xyzi', seq_id, frame + '.npy')
            if os.path.exists(new_file_name):
                continue
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
            
            
            np.save(new_file_name, xyzi)


if __name__ == '__main__':
    main()
