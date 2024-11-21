from experiments.lcrnet.data import (
    calibrate_neighbors_stack_mode,
    build_dataloader_stack_mode,
    test_loop_detection_collate_fn_stack_mode_online,
    train_loop_detection_collate_fn_stack_mode,
    train_loop_detection_collate_fn_stack_mode_online,
    train_loop_detection_collate_fn_stack_mode_halfonline
)



def train_valid_data_loader(cfg, distributed, dataset):
    dataset_init = loop_detection_dataset_initialization()
    train_dataset = getattr(dataset_init, '%s_dataset_initialization' % dataset)(
                cfg, dataset, 'train'
            )

    if cfg.train_mode == 'online':
        stack_fn = train_loop_detection_collate_fn_stack_mode_online
    elif cfg.train_mode == 'offline':
        stack_fn = train_loop_detection_collate_fn_stack_mode
    elif cfg.train_mode == 'half':
        stack_fn = train_loop_detection_collate_fn_stack_mode_halfonline

    neighbor_limits=[64 ,65, 74, 80]

    train_loader = build_dataloader_stack_mode(
        train_dataset,
        stack_fn,
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
        stack_fn,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        distributed=distributed,
        precompute_data=False
    )

    return train_loader, valid_loader, neighbor_limits

def test_loop_detection_data_loader(cfg, distributed, dataset, mode='test'):
    dataset_init = loop_detection_dataset_initialization()

    stack_fn = test_loop_detection_collate_fn_stack_mode_online

    test_dataset = getattr(dataset_init, '%s_dataset_initialization' % dataset)(
                    cfg, dataset, mode
                )

    precompute_data=True
    if precompute_data:
        neighbor_limits = calibrate_neighbors_stack_mode(
            test_dataset,
            stack_fn,
            cfg.backbone.num_stages,
            cfg.backbone.init_voxel_size,
            cfg.backbone.init_radius,
        )
    else:
        neighbor_limits=[64 ,65, 74, 80]

    test_loader = build_dataloader_stack_mode(
        test_dataset,
        stack_fn,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        distributed=distributed,
        precompute_data=True,
    )

    return test_loader, neighbor_limits


from experiments.lcrnet.datasets.loop_detection.ford.dataset_overlap import FordDataset

class loop_detection_dataset_initialization:
    def __init__(self):
        super().__init__()

    def kitti_dataset_initialization(self, cfg, dataset, mode):

        if mode == 'test' or mode == 'infer' or mode == 'all':
            from experiments.lcrnet.datasets.loop_detection.kitti.dataset_overlap_online import OdometryKittiPairDataset
        elif cfg.train_mode == 'online':
            from experiments.lcrnet.datasets.loop_detection.kitti.dataset_overlap_online import OdometryKittiPairDataset
            # from geotransformer.datasets.loop_detection.kitti.dataset_distance_online import OdometryKittiPairDataset
        elif cfg.train_mode == 'offline':
            from experiments.lcrnet.datasets.loop_detection.kitti.dataset_overlap import OdometryKittiPairDataset
        elif cfg.train_mode == 'half':
            from experiments.lcrnet.datasets.loop_detection.kitti.dataset_overlap_half_online import OdometryKittiPairDataset

        if mode == 'train':
            dataset = OdometryKittiPairDataset(
                cfg.data.dataset_root,
                mode,
                point_limit=cfg.train.point_limit,
                use_augmentation=cfg.train.use_augmentation,
                augmentation_noise=cfg.train.augmentation_noise,
                augmentation_min_scale=cfg.train.augmentation_min_scale,
                augmentation_max_scale=cfg.train.augmentation_max_scale,
                augmentation_shift=cfg.train.augmentation_shift,
                augmentation_rotation=cfg.train.augmentation_rotation,
                pos_num=cfg.train.pos_num,
                neg_num=cfg.train.neg_num,
                ground_segmentation=cfg.train.ground_segmentation,
                pre_load=cfg.train.pre_load,
            )
        else:
            dataset = OdometryKittiPairDataset(
                cfg.data.dataset_root,
                mode,
                point_limit=cfg.test.point_limit,
                use_augmentation=False,
                pre_load=False,
                ground_segmentation=cfg.train.ground_segmentation,
            )
        return dataset
    
    def ford_dataset_initialization(self, cfg, dataset, mode):
        if mode == 'train':
            dataset = FordDataset(
                cfg.data.dataset_ford_root,
                mode,
                point_limit=cfg.train.point_limit,
                use_augmentation=cfg.train.use_augmentation,
                augmentation_noise=cfg.train.augmentation_noise,
                augmentation_min_scale=cfg.train.augmentation_min_scale,
                augmentation_max_scale=cfg.train.augmentation_max_scale,
                augmentation_shift=cfg.train.augmentation_shift,
                augmentation_rotation=cfg.train.augmentation_rotation,
                pos_num=cfg.train.pos_num,
                neg_num=cfg.train.neg_num,
                ground_segmentation=cfg.train.ground_segmentation,
                pre_load=cfg.train.pre_load,
            )
        else:
            dataset = FordDataset(
                cfg.data.dataset_ford_root,
                mode,
                point_limit=cfg.test.point_limit,
                use_augmentation=False,
                pre_load=False,
                ground_segmentation=cfg.train.ground_segmentation,
            )
        return dataset