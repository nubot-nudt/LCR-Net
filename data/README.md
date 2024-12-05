# Preparation before training and test

This repo currently offers support for KITTI, KITTI360, Apollo, MulRan, and Ford campus datasets for continuous registration testing; KITTI and KITTI360 for closed-loop registration testing; and KITTI and Ford campus for loop detection testing.

Change output path `_C.output_root` to your preferred path. Logs, trained snapshots, processed features will be saved in this path.

## Dataset (KITTI)

Download the data from the [Kitti official website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).

```text
--(your dataset path)--Kitti--sequences--00--velodyne--000000.bin
                           |              |         |--...
                           |              |...
                           |--calib--sequences--00--calib.txt
                                             |   |--times.txt
                                             |--...
```

Change `_C.data.dataset_root` in `experiments/lcrnet/config_model.py` to your dataset path

### Downsample

Downsample the point cloud:

```
python data/Kitti/downsample_pcd_kitti.py
```

### Generate ground truth continuous pairs

Generate ground truth text for registration testing:

```
python data/Kitti/generate_kitti_pairs.py
```

You can also use our generated ones in  `assets/data/registration/icp10`.

Put the files in `(your dataset path)/Kitti/icp10` directory.

### Generate ground truth closed loop pairs for registration

Generate ground truth text for loop closing  testing:

```
python data/Kitti/generate_kitti_loop_pairs.py
```

You can also use our generated ones in  `assets/data/registration/loop_distance4` and  `assets/data/registration/loop_overlap0.3`.

Put the files in `(your dataset path)/Kitti/loop_distance4` and `(your dataset path)/Kitti/loop_overlap0.3` directory.

### Ground truth loop pairs for loop detection

Please follow [overlapnet](https://github.com/PRBonn/OverlapNet/) to generate the ground truth files. Also, you can use our generated one in  `assets/data/loop_detection/overlap`.

Put the files in `(your dataset path)/Kitti/overlap`, and unzip `overlap-based_gt_pairs.zip`.



## Dataset (KITTI360)

TO BE RELEASED



## Dataset (Apollo)

TO BE RELEASED



## Dataset (Ford campus)

TO BE RELEASED



## Dataset (Mulran)

TO BE RELEASED

