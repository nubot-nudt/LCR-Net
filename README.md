# Fast and Accurate Deep Loop Closing and Relocalization for Reliable LiDAR SLAM

**This work has been accepted by IEEE Transactions on Robotics (TRO) :tada: [[pdf](https://arxiv.org/pdf/2309.08086.pdf)] [[video](tbd)].**

We propose LCR-Net to tackle both LiDAR-based loop detection and registration. 
It exploits novel feature extraction and a 3D rotary attention mechanism to precisely estimate similarities and 6-DoF poses between pairs of LiDAR scans. 

LCR-Net has been integrated into SLAM systems, serving both as a loop closing module and relocalization module. This integration ensures robust and accurate online LiDAR SLAM in outdoor driving environments. 

<img src="https://github.com/nubot-nudt/LCR-Net/blob/main/assets/demo.gif" alt="demo" style="zoom:150%;" />

## Publication

If you use our implementation in your academic work, please cite the corresponding [paper](https://ieeexplore.ieee.org/document/10494918):

```
@ARTICLE{shi2024tro,
  author={Shi, Chenghao and Chen, Xieyuanli and Xiao, Junhao and Dai, Bin and Lu, Huimin},
  journal={IEEE Transactions on Robotics}, 
  title={Fast and Accurate Deep Loop Closing and Relocalization for Reliable LiDAR SLAM}, 
  year={2024},
  volume={40},
  number={},
  pages={2620-2640},
  doi={10.1109/TRO.2024.3386363}}
```

The 3D-RoFormer follows [RDMNet](https://github.com/nubot-nudt/RDMNet), please cite:

```
@article{shi2023tits,
   author={Shi, Chenghao and Chen, Xieyuanli and Lu, Huimin and Deng, Wenbang and Xiao, Junhao and Dai, Bin},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={RDMNet: Reliable Dense Matching Based Point Cloud Registration for Autonomous Driving}, 
  year={2023},
  volume={24},
  number={10},
  pages={11372-11383},
  doi={10.1109/TITS.2023.3286464}}

```



## Installation

```
python setup.py build develop
```

The code has been tested with {Ubuntu 20.04, torch 1.8.0, cuda 11.4, and python 3.8.8}/{Ubuntu 22.04, torch 2.4.1, cuda 12.4, and python 3.8.19}



## Quick demo

For quick test, we provide 4 pre-trained models [here](https://1drv.ms/u/c/c1d2e94819d2dee5/ERfCV8Gd8PpBlWhlMb9DD80BlTywaYLdaNaJzgcQ1buSkw?e=9PMG5C).

For mainland China friends, you may use a faster [link](https://pan.baidu.com/s/1df7DMv9X4W3v7hCtAaj89w), extract code: m8bh.

Below is an notation for each model.

> ./weights/best-model-lc.tar    # Dense point matching head trained on close-loop point cloud pairs, denoted as $LCR-Net^\dagger$ in our paper.
> ./weights/best-model-reg.tar # Dense point matching head trained on continuous point cloud pairs, denoted as $LCR-Net^\diamond$​ in our paper.
>
> ./weights/best-model-ld.tar  # Global description head with backbone pre-trained on continuous point cloud pairs, denoted as $LCR-Net^\diamond$ in our paper.
>
> ./weights/best-model-mixed.tar # Full LCR-Net trained on mixed point cloud pairs, denoted as $LCR-Net$ in our paper.

Unzip weights and put them in `weights` directory, then run:

```
python3 demo/demo.py 
```

This script returns the dissimilarity between the two scans and calculates their relative pose:

> Test pos_idx: 3854 and anc_idx: 958
>
> L2 feature distance: 0.809192
>
> Estimated transformation:
> [[ 3.8640183e-01 -9.2232913e-01 -1.6882520e-03 -5.1863933e+00]
> [ 9.2216253e-01  3.8629568e-01  1.9789029e-02  5.1413069e+00]
> [-1.7599827e-02 -9.2033548e-03  9.9980271e-01 -8.8044703e-02]
> [ 0.0000000e+00  0.0000000e+00  0.0000000e+00  1.0000000e+00]]

Use `--vis=True` to enable visualization, and you will get the following in sequence:

![Picture1](https://github.com/nubot-nudt/LCR-Net/blob/main/assets/demo.png)

Carefully examining figures b) and d), you can find that as long as there are geometrically salient object near the nodes, they are likely to be moved to the object, thereby enhancing the ability to recognize and match the same object as a whole.

We also provide a case for two scans (4481,958) that have no correlation. The network will output a significant dissimilarity.



## Preparation before training and testing

This repo currently offers support for KITTI, KITTI360, Apollo, MulRan, and Ford campus datasets for continuous registration testing; KITTI and KITTI360 for closed-loop registration testing; and KITTI and Ford campus for loop detection testing. Following [instructions]( data/README.md)  to prepare data and ground truth files.



## Training and testing

### Training

We employ a two-stage training approach, which initially trains the network for the registration task, followed by training the global description head for the candidate retrieval.

To begin with the initial training phase for the registration task, run the following command:

```sh
python3 experiments/registration/trainval_mixed.py
```

Saving the best model in the registration task, and then you can train the global description head. Run the following command:

```sh
python3 experiments/loop_detection/trainval_loop_detection.py --snapshot=path/to/best-model.tar 
```

Saving the best model in the loop_detection task, and then use this [script](experiments/lcrnet/merge_model.py) to merge the two snapshots:

```sh
python3 experiments/lcrnet/merge_model.py --model_reg=path/to/best-model-reg.tar  --model_reg=path/to/best-model-ld.tar 
```

### Testing

##### To reproduce the results in our loop detection experiments, run the following command:

```sh
python3 experiments/loop_detection/test_loop_detection.py --snapshot=./weights/best-model-mixed.tar --dataset=kitti
```

This script generates and saves global descriptors.  Use `--dataset` to specify the dataset; currently, loop detection testing supports the KITTI and Ford campus datasets. Use `--snapshot=./weights/best-model-reg.tar` to reproduce the results of $LCR-Net^\diamond$. 

To evaluate the loop detection performance, run:

```sh
python3 experiments/loop_detection/eval_loop_detection_overlap_dataset.py --dataset=kitti
```

##### To reproduce the results in our loop closing experiments, run the following command:

```sh
python3 experiments/registration/test_loop_closure.py --snapshot=./weights/best-model-mixed.tar
```

This script returns registration results and middle outputs. Use `--seq` to specify the test sequence, `--lc_text_root` to specify ground truth root, `--dataset` to select dataset. Currently, the loop closing test supports the KITTI and KITTI360 datasets.

```sh
python3 experiments/registration/test_loop_closure.py --snapshot=./weights/best-model-mixed.tar --dataset=kitti --seq=8 --lc_text_root=/loop_distance4
```

Use `--snapshot=./weights/best-model-lc.tar` to reproduce the results of $LCR-Net^\dagger$.  

To evaluate the loop closing performance, run:

```sh
python3 experiments/registration/eval.py --dataset=kitti
```

Use`--method=ransac` to specify RANSAC as the pose estimation algorithm.

##### To reproduce the results in our continuous registration experiments, run the following command:

```sh
python3 experiments/registration/test_registration.py --snapshot=./weights/best-model-mixed.tar --dataset=kitti --reg_text_root=/icp10
```

This script returns registration results and middle outputs. Use `--reg_text_root` to specify ground truth root, `--dataset` to select datasets. Currently, the loop closing test supports the KITTI, KITTI360, Apollo, Ford campus, and Mulran datasets.

Use `--snapshot=./weights/best-model-reg.tar` to reproduce the results of $LCR-Net^\diamond$. To evaluate the loop closing performance, run:

```sh
python3 experiments/registration/eval.py --dataset=kitti --subset=continuous
```

Use`--method=ransac` to specify RANSAC as the pose estimation algorithm. 



## Results

All the results presented below are evaluated using the same model $LCR-Net$ trained **only on KITTI**.

### Candidate Retrieval

| Datasets (overlap>0.3) | AUC   | F1max | Recall@1 | Recall@1% |
| ---------------------- | ----- | ----- | -------- | --------- |
| KITTI Seq.00           | 0.958 | 0.922 | 0.937    | 0.993     |
| Ford campus            | 0.972 | 0.920 | 0.932    | 0.987     |

<img src="https://github.com/nubot-nudt/LCR-Net/blob/main/assets/pr.png" alt="pr" />

### Closed loop registration

| Datasets (overlap>0.3) | RR    | RYE(°) | RTE(cm) |
| ---------------------- | ----- | ------ | ------- |
| KITTI Seq.00           | 100   | 0.10   | 5       |
| KITTI Seq.08           | 100   | 0.34   | 8       |
| KITTI-360 Seq.09       | 100   | 0.14   | 10      |
| KITTI-360 Seq.02       | 98.64 | 0.27   | 21      |

LCRNet is **the only method achieves a 100% registration recall on the test data** (except KITTI-360 Seq.02). We find that the majority of the failed samples in KITTI-360 Seq.02 are attributable to issues with the ground truth data.

### Continuous registration

| Datasets (distance>10m) | RR    | RRE(°) | RTE(cm) |
| ----------------------- | ----- | ------ | ------- |
| KITTI                   | 99.82 | 0.19   | 3.9     |
| KITTI-360               | 99.94 | 0.24   | 5.8     |
| Apollo                  | 100   | 0.09   | 3.4     |
| Ford campus             | 100   | 0.16   | 6.6     |
| Mulran                  | 98.22 | 0.17   | 7.4     |



## Testing on your own data

To test on your own data, it is recommended to implement a `Dataset` as in `experiments/lcrnet/datasets/loop_closure/kitti/dataset_demo.py`, and generate your ground truth text referring to `generate_kitti_loop_pairs.py` or `generate_kitti_pairs.py` in the `data/Kitti` directory.  

Use `--dataset_root` to specify the root directory for the dataset, `--gt_text_root` to specify the root directory for the ground truth data, `--seq` to specify the sequence. Run the following command:

```sh
python3 demo/demo.py --dataset_root=/your/Dataset/root --gt_text_root=/your/ground/truth/text/root --seq=0
```



## Inference for practice use

In practical applications, when ground truth is unavailable, you might only be interested in obtaining the network's similarity estimates or pose estimates. We have provided corresponding scripts in the `experiments/inference` directory.

**To generate global descriptors for a set of data, please refer to the following example:**

```sh
python3 experiments/inference/infer_loop_detection_descriptor_generation.py --dataset=kitti --seq=8
```

Afterward, utilize the following command to output the most similar candidate frames for each individual frame:

```sh
python3 experiments/inference/infer_loop_detection_find_top1.py --dataset=kitti --seq=8
```



**To generate pose estimates between any two frames**, please refer to the [file](assets/data/08) to set the index of the data you wish to test, and then follow the example below to generate the pose estimates:

```sh
python3 experiments/inference/infer_registration.py --dataset=kitti --seq=8
```




## Acknowledgements

 We thank the following authors for open sourcing their methods.

- [GeoTransformer](https://github.com/qinzheng93/GeoTransformer)

- [D3Feat](https://github.com/XuyangBai/D3Feat.pytorch)
- [PREDATOR](https://github.com/prs-eth/OverlapPredator)
- [CoFiNet](https://github.com/haoyu94/Coarse-to-fine-correspondences)
- [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork)



## License

This project is free software made available under the MIT License. For details see the LICENSE file.
