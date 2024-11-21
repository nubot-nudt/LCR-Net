import sys
import json
import argparse
import glob
import os.path as osp
import time

import numpy as np
import torch
from tqdm import tqdm

from experiments.lcrnet.config_reg import make_cfg
from utils.engine import Logger
from experiments.lcrnet.modules.registration import weighted_procrustes
from utils.utils.summary_board import SummaryBoard
from utils.utils.open3d import registration_with_ransac_from_correspondences
from utils.utils.registration import (
    evaluate_sparse_correspondences,
    evaluate_correspondences,
    compute_registration_error,
)



def eval_one_epoch(args, cfg, logger):

    
    features_root = cfg.feature_dir

    coarse_matching_meter = SummaryBoard()
    coarse_matching_meter.register_meter('num')
    coarse_matching_meter.register_meter('precision')
    coarse_matching_meter.register_meter('recall')
    coarse_matching_meter.register_meter('hit_ratio')
    coarse_matching_meter.register_meter('PMR>0')
    coarse_matching_meter.register_meter('node_OV')

    fine_matching_meter = SummaryBoard()
    fine_matching_meter.register_meter('recall')
    fine_matching_meter.register_meter('inlier_ratio')
    fine_matching_meter.register_meter('inlier_ratio_0.3')
    fine_matching_meter.register_meter('inlier_ratio_0.1')
    fine_matching_meter.register_meter('f_recall')
    fine_matching_meter.register_meter('f_hit_ratio')
    fine_matching_meter.register_meter('overlap')
    fine_matching_meter.register_meter('num_corr')

    registration_meter = SummaryBoard()
    registration_meter.register_meter('recall')
    registration_meter.register_meter('rre')
    registration_meter.register_meter('rte')
    registration_meter.register_meter('rx')
    registration_meter.register_meter('ry')
    registration_meter.register_meter('rz')

    overlap_meter = SummaryBoard()
    overlap_meter.register_meter('n2p_p_mean')
    overlap_meter.register_meter('n2p_n_mean')
    overlap_meter.register_meter('n2p_p_std')
    overlap_meter.register_meter('n2p_n_std')

    fail_case = []
    file_names = glob.glob(osp.join(features_root, '*.npz'))
    
    num_test_pairs = len(file_names)

    pbar = tqdm(enumerate(file_names), total=num_test_pairs)
    for i, file_name in pbar:
        try:
            seq_id, anc_frame, pos_frame = [int(x) for x in osp.splitext(osp.basename(file_name))[0].split('_')]
        except:
            seq_id, anc_frame, pos_frame = [x for x in osp.splitext(osp.basename(file_name))[0].split('_')]
            anc_frame=int(anc_frame)
            pos_frame=int(pos_frame)

        # delete bad data
        if seq_id==8 and anc_frame==15 and pos_frame==58:
            continue
        
        data_dict = np.load(file_name)
        try:
            pos_nodes = data_dict['pos_points_c']
            anc_nodes = data_dict['anc_points_c']
            pos_node_corr_indices = data_dict['pos_node_corr_indices']
            anc_node_corr_indices = data_dict['anc_node_corr_indices']
            pos_corr_points = data_dict['pos_corr_points']
            anc_corr_points = data_dict['anc_corr_points']
        except:
            pos_nodes = data_dict['ref_points_c']
            anc_nodes = data_dict['src_points_c']
            pos_node_corr_indices = data_dict['ref_node_corr_indices']
            anc_node_corr_indices = data_dict['src_node_corr_indices']
            pos_corr_points = data_dict['ref_corr_points']
            anc_corr_points = data_dict['src_corr_points']

        corr_scores = data_dict['corr_scores']
        gt_node_corr_overlaps = data_dict['gt_node_corr_overlaps']
        gt_node_corr_indices = data_dict['gt_node_corr_indices']
        gt_transform = data_dict['transform']

        # pos_points=data_dict['pos_points'],
        # anc_points=data_dict['anc_points'],

        # overlaps=data_dict['overlaps'],

        if args.num_corr is not None and corr_scores.shape[0] > args.num_corr:
            sel_indices = np.argsort(-corr_scores)[: args.num_corr]
            pos_corr_points = pos_corr_points[sel_indices]
            anc_corr_points = anc_corr_points[sel_indices]
            corr_scores = corr_scores[sel_indices]


        message = '{}/{}, seq_id: {}, id0: {}, id1: {}'.format(i + 1, num_test_pairs, seq_id, anc_frame, pos_frame)

        # 1. evaluate correspondences
        # 1.1 evaluate coarse correspondences
        coarse_matching_result_dict = evaluate_sparse_correspondences(
            pos_nodes,
            anc_nodes,
            pos_node_corr_indices,
            anc_node_corr_indices,
            gt_node_corr_indices,
        )        

        coarse_precision = coarse_matching_result_dict['precision']
        recall = coarse_matching_result_dict['recall']
        hit_ratio = coarse_matching_result_dict['hit_ratio']

        coarse_matching_meter.update('num', pos_node_corr_indices.shape[0])
        coarse_matching_meter.update('precision', coarse_precision)
        coarse_matching_meter.update('recall', recall)
        coarse_matching_meter.update('hit_ratio', hit_ratio)
        coarse_matching_meter.update('PMR>0', float(coarse_precision > 0))


        # 1.2 evaluate fine correspondences
        fine_matching_result_dict = evaluate_correspondences(
            pos_corr_points,
            anc_corr_points,
            gt_transform,
            positive_radius=cfg.eval.acceptance_radius,
        )
        inlier_ratio = fine_matching_result_dict['inlier_ratio']
        overlap = fine_matching_result_dict['overlap']

        f_recall=0
        f_hit_ratio=0


        fine_matching_meter.update('inlier_ratio', inlier_ratio)
        fine_matching_meter.update('inlier_ratio_0.3', fine_matching_result_dict['inlier_ratio_0.3'])
        fine_matching_meter.update('inlier_ratio_0.1', fine_matching_result_dict['inlier_ratio_0.1'])
        fine_matching_meter.update('f_recall', f_recall)
        fine_matching_meter.update('f_hit_ratio', f_hit_ratio)
        fine_matching_meter.update('overlap', overlap)
        fine_matching_meter.update('recall', float(inlier_ratio >= cfg.eval.inlier_ratio_threshold))
        fine_matching_meter.update('num_corr', pos_corr_points.shape[0])

        message += ', c_PIR: {:.3f}'.format(coarse_precision)
        message += ', c_RECALL: {:.3f}'.format(recall)
        message += ', c_HIT: {:.3f}'.format(hit_ratio)
        message += ', f_IR: {:.3f}'.format(inlier_ratio)
        message += ', f_RECALL: {:.3f}'.format(f_recall)
        message += ', f_HIT: {:.3f}'.format(f_hit_ratio)
        # message += ', f_OV: {:.3f}'.format(overlap)
        # message += ', f_RS: {:.3f}'.format(fine_matching_result_dict['residual'])
        message += ', f_NU: {}'.format(fine_matching_result_dict['num_corr'])

        # 2. evaluate registration
        if args.method == 'lgr':
            try:
                est_transform = data_dict['estimated_transform']
            except:
                est_transform = data_dict['estimated_transform_lgr']
        elif args.method == 'ransac':
            est_transform = registration_with_ransac_from_correspondences(
                anc_corr_points,
                pos_corr_points,
                distance_threshold=cfg.ransac.distance_threshold,
                ransac_n=cfg.ransac.num_points,
                num_iterations=cfg.ransac.num_iterations,
            )
            modified_data = dict(data_dict)
            modified_data['estimated_transform_ransac'] = est_transform
            np.savez(osp.join(features_root,file_name), **modified_data)


        elif args.method == 'svd':
            with torch.no_grad():
                pos_corr_points = torch.from_numpy(pos_corr_points).cuda()
                anc_corr_points = torch.from_numpy(anc_corr_points).cuda()
                corr_scores = torch.from_numpy(corr_scores).cuda()
                est_transform = weighted_procrustes(
                    anc_corr_points, pos_corr_points, corr_scores, return_transform=True
                )
                est_transform = est_transform.detach().cpu().numpy()
        elif args.method == 'teaser':
            import teaserpp_python
            solver_params = teaserpp_python.RobustRegistrationSolver.Params()
            solver_params.cbar2 = 1
            solver_params.noise_bound = 0.01
            solver_params.estimate_scaling = False
            solver_params.rotation_estimation_algorithm = (
                teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
            )
            solver_params.rotation_gnc_factor = 1.4
            solver_params.rotation_max_iterations = 100
            solver_params.rotation_cost_threshold = 1e-12

            solver = teaserpp_python.RobustRegistrationSolver(solver_params)
            solver.solve(anc_corr_points.T, pos_corr_points.T)

            solution = solver.getSolution()

            est_transform = np.zeros((4,4))
            est_transform[:3,:3] = solution.rotation
            est_transform[:3,3] = solution.translation
            est_transform[3,:3] = 1.0
        else:
            raise ValueError(f'Unsupported registration method: {args.method}.')

        rre, rte, rx, ry, rz = compute_registration_error(gt_transform, est_transform)
        accepted = rre < cfg.eval.rre_threshold and rte < cfg.eval.rte_threshold
        if accepted:
            registration_meter.update('rre', rre)
            registration_meter.update('rte', rte)
            registration_meter.update('rx',rx)
            registration_meter.update('ry',ry)
            registration_meter.update('rz',rz)
        else:
            fail_case.append([seq_id,anc_frame, pos_frame, coarse_precision, inlier_ratio, pos_corr_points.shape[0]])
        registration_meter.update('recall', float(accepted))
        
        message += ', r_RRE: {:.3f}'.format(rre)
        message += ', r_RTE: {:.3f}'.format(rte)
        message += ', Fail_case: {}'.format(len(fail_case))
        

        if args.verbose:
            logger.info(message)
        

        pbar.set_description(message)

    message = 'Pairs: {}'.format(num_test_pairs)
    logger.critical(message)

    # 1. print correspondence evaluation results
    message = '  Coarse Matching'
    message += ', NUM: {:.3f}'.format(coarse_matching_meter.mean('num'))
    message += ', PIR: {:.3f}'.format(coarse_matching_meter.mean('precision'))
    message += ', RECALL: {:.3f}'.format(coarse_matching_meter.mean('recall'))
    message += ', HIT_RATIO: {:.3f}'.format(coarse_matching_meter.mean('hit_ratio'))
    message += ', PMR>0: {:.3f}'.format(coarse_matching_meter.mean('PMR>0'))
    
    logger.critical(message)

    message = '  Fine Matching'
    message += ', FMR: {:.4f}'.format(fine_matching_meter.mean('recall'))
    message += ', IR: {:.3f}'.format(fine_matching_meter.mean('inlier_ratio'))
    message += ', RECALL: {:.3f}'.format(fine_matching_meter.mean('f_recall'))
    message += ', HIT_RATIO: {:.3f}'.format(fine_matching_meter.mean('f_hit_ratio'))
    message += ', num_Corr: {:.3f}'.format(fine_matching_meter.mean('num_corr'))
    # message += ', OV: {:.3f}'.format(fine_matching_meter.mean('overlap'))
    # message += ', std: {:.3f}'.format(fine_matching_meter.std('recall'))
    logger.critical(message)

    # 2. print registration evaluation results
    message = '  Registration'
    message += ', RR: {:.4f}'.format(registration_meter.mean("recall"))
    message += ', RRE: {:.3f}'.format(registration_meter.mean("rre"))
    message += ', RTE: {:.3f}'.format(registration_meter.mean("rte"))
    message += ', Rx: {:.3f}'.format(registration_meter.mean("rx"))
    message += ', Ry: {:.3f}'.format(registration_meter.mean("ry"))
    message += ', Rz: {:.3f}'.format(registration_meter.mean("rz"))
    logger.critical(message)

    message = '  Fail case'
    logger.critical(fail_case)



def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['lgr', 'ransac', 'svd', 'ransac_featurematch'], default='lgr', help='registration method')
    parser.add_argument('--num_corr', type=int, default=None, help='number of correspondences for registration')
    parser.add_argument('--verbose', action='store_true', help='verbose mode')
    parser.add_argument('--dataset', type=str, default='kitti', help='')
    parser.add_argument('--subset', choices=['loop', 'continuous'], default='loop', help='')
    return parser

def main():
    parser = make_parser()
    args = parser.parse_args()

    cfg = make_cfg()
    log_file = osp.join(cfg.log_dir, 'eval-{}.log'.format(time.strftime("%Y%m%d-%H%M%S")))
    logger = Logger(log_file=log_file)

    message = 'Command executed: ' + ' '.join(sys.argv)
    logger.info(message)
    message = 'Configs:\n' + json.dumps(cfg, indent=4)
    logger.info(message)

    if args.subset=='loop':
        cfg.feature_dir = cfg.lc_feature_dir+args.dataset
    elif  args.subset=='continuous':
        cfg.feature_dir = cfg.feature_dir+args.dataset
    
    eval_one_epoch(args, cfg, logger)


if __name__ == '__main__':
    main()
