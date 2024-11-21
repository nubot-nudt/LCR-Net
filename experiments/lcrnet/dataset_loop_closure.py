from experiments.lcrnet.datasets.loop_closure.kitti.dataset import OdometryKittiDataset
from experiments.lcrnet.datasets.loop_closure.kitti.dataset_demo import OdometryKittiDatasetDemo as OdometryKittiDatasetDemo
from experiments.lcrnet.datasets.loop_closure.kitti_360.dataset import OdometryKitti360Dataset


from experiments.lcrnet.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
    build_dataloader_stack_mode,
)


def train_valid_data_loader(cfg, distributed, dataset):
    dataset_init = loop_closure_dataset_initialization()
    train_dataset = getattr(dataset_init, '%s_dataset_initialization' % dataset)(
                cfg, dataset, 'train'
            )

    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )


    train_loader = build_dataloader_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        distributed=distributed,
        precompute_data=True,
        pin_memory=False
    )

    valid_dataset = getattr(dataset_init, '%s_dataset_initialization' % dataset)(
                cfg, dataset, 'val'
            )
    valid_loader = build_dataloader_stack_mode(
        valid_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=True,
        distributed=distributed,
        precompute_data=True
    )

    return train_loader, valid_loader, neighbor_limits

def test_loop_closure_data_loader(cfg, distributed, dataset):
    dataset_init = loop_closure_dataset_initialization()

    test_dataset = getattr(dataset_init, '%s_dataset_initialization' % dataset)(
                cfg, dataset, 'test'
            )
    neighbor_limits = calibrate_neighbors_stack_mode(
        test_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )
    
    test_loader = build_dataloader_stack_mode(
        test_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        distributed=distributed,
        precompute_data=True
    )

    return test_loader, neighbor_limits


def infer_loop_closure_data_loader(cfg, distributed, dataset, mode='infer'):
    dataset_init = loop_closure_dataset_initialization()

    infer_dataset = getattr(dataset_init, '%s_dataset_initialization' % dataset)(
                cfg, dataset, mode
            )
    neighbor_limits = calibrate_neighbors_stack_mode(
        infer_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )

    infer_loader = build_dataloader_stack_mode(
        infer_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        distributed=distributed,
        precompute_data=True
    )

    return infer_loader, neighbor_limits


class loop_closure_dataset_initialization:
    def __init__(self):
        super().__init__()

    def kitti_dataset_initialization(self, cfg, dataset, mode):
        if mode == 'train' or mode == 'val':
            dataset = OdometryKittiDataset(
                cfg.data.dataset_root,
                cfg.train.lc_text_root,
                mode,
                point_limit=cfg.train.point_limit,
                use_augmentation=cfg.train.use_augmentation,
                augmentation_noise=cfg.train.augmentation_noise,
                augmentation_min_scale=cfg.train.augmentation_min_scale,
                augmentation_max_scale=cfg.train.augmentation_max_scale,
                augmentation_shift=cfg.train.augmentation_shift,
                augmentation_rotation=cfg.train.augmentation_rotation,
                ground_segmentation=cfg.train.ground_segmentation,
            )
        elif mode == 'demo':
            dataset = OdometryKittiDatasetDemo(
                cfg.dataset_root,
                cfg.pos_frame,
                cfg.anc_frame,
                cfg.gt_text_root,
                cfg.seq,
                point_limit=cfg.test.point_limit,
            )
        else:
            dataset = OdometryKittiDataset(
                cfg.data.dataset_root,
                cfg.test.lc_text_root,
                mode,
                seq=cfg.seq,
                point_limit=cfg.test.point_limit,
                use_augmentation=False,
                ground_segmentation=cfg.test.ground_segmentation,
            )
        return dataset

    def kitti360_dataset_initialization(self, cfg, dataset, mode):
        if mode == 'train':
            dataset = OdometryKitti360Dataset(
                cfg.data.dataset_360_root,
                cfg.test.lc_text_root,
                mode,
                point_limit=cfg.train.point_limit,
                use_augmentation=cfg.train.use_augmentation,
                augmentation_noise=cfg.train.augmentation_noise,
                augmentation_min_scale=cfg.train.augmentation_min_scale,
                augmentation_max_scale=cfg.train.augmentation_max_scale,
                augmentation_shift=cfg.train.augmentation_shift,
                augmentation_rotation=cfg.train.augmentation_rotation,
            )
        else:
            dataset = OdometryKitti360Dataset(
                cfg.data.dataset_360_root,
                cfg.test.lc_text_root,
                mode,
                seq=cfg.seq,
                point_limit=cfg.test.point_limit,
                use_augmentation=False,
            )
        return dataset
