from experiments.lcrnet.datasets.registration.kitti.dataset import OdometryKittiPairDataset
from experiments.lcrnet.datasets.registration.ford.dataset import OdometryFordPairDataset
from experiments.lcrnet.datasets.registration.kitti_360.dataset import OdometryKitti360PairDataset
from experiments.lcrnet.datasets.registration.apollo.dataset import OdometryApolloPairDataset
from experiments.lcrnet.datasets.registration.mulran.dataset import OdometrymulranPairDataset

from experiments.lcrnet.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
    build_dataloader_stack_mode,
)


def train_valid_data_loader(cfg, distributed, dataset):
    dataset_init = registration_dataset_initialization()
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
        shuffle=False,
        distributed=distributed,
    )

    return train_loader, valid_loader, neighbor_limits


def test_data_loader(cfg, distributed, dataset):
    dataset_init = registration_dataset_initialization()
    train_dataset = getattr(dataset_init, '%s_dataset_initialization' % dataset)(
                cfg, dataset, 'test'
            )
    
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )


    test_dataset = getattr(dataset_init, '%s_dataset_initialization' % dataset)(
                cfg, dataset, 'test'
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
    )

    return test_loader, neighbor_limits


class registration_dataset_initialization:
    def __init__(self):
        super().__init__()

    def kitti_dataset_initialization(self, cfg, dataset, mode):
        if mode == 'train':
            dataset = OdometryKittiPairDataset(
                cfg.data.dataset_root,
                cfg.train.reg_text_root,
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
        else:
            dataset = OdometryKittiPairDataset(
                cfg.data.dataset_root,
                cfg.test.reg_text_root,
                mode,
                point_limit=cfg.test.point_limit,
                use_augmentation=False,
                ground_segmentation=cfg.test.ground_segmentation,
            )
        return dataset
    
    def ford_dataset_initialization(self, cfg, dataset, mode):
        if mode == 'train':
            dataset = OdometryFordPairDataset(
                cfg.data.dataset_ford_root,
                cfg.train.reg_text_root,
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
        else:
            dataset = OdometryFordPairDataset(
                cfg.data.dataset_ford_root,
                cfg.test.reg_text_root,
                mode,
                point_limit=cfg.test.point_limit,
                use_augmentation=False,
                ground_segmentation=cfg.test.ground_segmentation,
            )
        return dataset

    def kitti360_dataset_initialization(self, cfg, dataset, mode):
        if mode == 'train':
            dataset = OdometryKitti360PairDataset(
                cfg.data.dataset_360_root,
                cfg.train.reg_text_root,
                'train',
                point_limit=cfg.train.point_limit,
                use_augmentation=cfg.train.use_augmentation,
                augmentation_noise=cfg.train.augmentation_noise,
                augmentation_min_scale=cfg.train.augmentation_min_scale,
                augmentation_max_scale=cfg.train.augmentation_max_scale,
                augmentation_shift=cfg.train.augmentation_shift,
                augmentation_rotation=cfg.train.augmentation_rotation,
            )
        else:
            dataset = OdometryKitti360PairDataset(
                cfg.data.dataset_360_root,
                cfg.test.reg_text_root,
                mode,
                point_limit=cfg.test.point_limit,
                use_augmentation=False,
            )
        return dataset

    def apollo_dataset_initialization(self, cfg, dataset, mode):
        if mode == 'train':
            dataset = OdometryApolloPairDataset(
                cfg.data.apollo_root,
                cfg.train.reg_text_root,
                'train',
                point_limit=cfg.train.point_limit,
                use_augmentation=cfg.train.use_augmentation,
                augmentation_noise=cfg.train.augmentation_noise,
                augmentation_min_scale=cfg.train.augmentation_min_scale,
                augmentation_max_scale=cfg.train.augmentation_max_scale,
                augmentation_shift=cfg.train.augmentation_shift,
                augmentation_rotation=cfg.train.augmentation_rotation,
            )
        else:
            dataset = OdometryApolloPairDataset(
                cfg.data.apollo_root,
                cfg.test.reg_text_root,
                mode,
                point_limit=cfg.test.point_limit,
                use_augmentation=False,
            )
        return dataset

    def mulran_dataset_initialization(self, cfg, dataset, mode):
        if mode == 'train':
            dataset = OdometrymulranPairDataset(
                cfg.data.mulran_root,
                cfg.train.reg_text_root,
                'train',
                point_limit=cfg.train.point_limit,
                use_augmentation=cfg.train.use_augmentation,
                augmentation_noise=cfg.train.augmentation_noise,
                augmentation_min_scale=cfg.train.augmentation_min_scale,
                augmentation_max_scale=cfg.train.augmentation_max_scale,
                augmentation_shift=cfg.train.augmentation_shift,
                augmentation_rotation=cfg.train.augmentation_rotation,
            )
        else:
            dataset = OdometrymulranPairDataset(
                cfg.data.mulran_root,
                cfg.test.reg_text_root,
                mode,
                point_limit=cfg.test.point_limit,
                use_augmentation=False,
            )
        return dataset