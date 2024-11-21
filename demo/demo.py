import os.path as osp
import time


from utils.engine import SingleTester
from utils.utils.common import ensure_dir, get_log_string,empty_dir
from utils.utils.torch import release_cuda

from experiments.lcrnet.dataset_loop_closure import infer_loop_closure_data_loader 

from experiments.lcrnet.model_family.LCRNet import create_model
from experiments.lcrnet.config_reg import make_cfg
import numpy as np


class Tester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg)


        cfg.pos_frame = self.args.pos_frame
        cfg.anc_frame = self.args.anc_frame
        cfg.dataset = self.args.dataset # default: kitti, no need to change for demo
        cfg.dataset_root = self.args.dataset_root
        cfg.gt_text_root = self.args.gt_text_root
        cfg.vis = self.args.vis
        cfg.seq = self.args.seq

        start_time = time.time()
        data_loader, neighbor_limits = infer_loop_closure_data_loader(cfg, self.distributed, cfg.dataset, 'demo')

        loading_time = time.time() - start_time
        message = f'Data loader created: {loading_time:.3f}s collapsed.'
        self.logger.info(message)
        message = f'Calibrate neighbors: {neighbor_limits}.'
        self.logger.info(message)
        self.register_loader(data_loader)

        cfg.neighbor_limits=neighbor_limits
        # model
        model = create_model(cfg).cuda()
        self.register_model(model)

        # preparation
        # self.output_dir = osp.join(cfg.output_dir)
        # ensure_dir(self.output_dir)
        # empty_dir(self.output_dir)

        self.lgr_time=[]
        self.ransac_time=[]

    def test_step(self, iteration, data_dict):
        output_dict = self.model(data_dict)
        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
        pass

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        pos_idx = data_dict['pos_idx']
        anc_idx=data_dict['anc_idx']
        message = f'pos_idx: {pos_idx}, anc_idx:{anc_idx}'
        # message += ', ' + get_log_string(result_dict=result_dict)
        message += ', nCorr: {}'.format(output_dict['corr_scores'].shape[0])
        return message

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        positive_idx = data_dict['pos_idx']
        anchor_idx = data_dict['anc_idx']

        estimated_transform = release_cuda(output_dict['estimated_transform'])
        pos_feature_global = release_cuda(output_dict['pos_feature_global'])
        anc_feature_global = release_cuda(output_dict['anc_feature_global'])

        feat_dis = np.sqrt(np.sum((pos_feature_global - anc_feature_global) ** 2))

        M2=estimated_transform.reshape(-1)[:12]
        print('Test pos_idx: %i and anc_idx: %i\nL2 feature distance: %f\nEstimated transformation:\n%s'%(positive_idx, anchor_idx, feat_dis, estimated_transform))
        if data_dict['transform'] is None:
            f = open(osp.join('./demo', 'lcr_output'),'a')
            f.write(f'{positive_idx} {anchor_idx} {feat_dis:.2f} {M2[0]:.6f} {M2[1]:.6f} {M2[2]:.6f} {M2[3]:.6f} {M2[4]:.6f} {M2[5]:.6f} {M2[6]:.6f} {M2[7]:.6f} {M2[8]:.6f} {M2[9]:.6f} {M2[10]:.6f} {M2[11]:.6f} \n')
        else:
            seq_id = data_dict['seq_id']

            file_name = osp.join(self.output_dir, f'{seq_id}_{anchor_idx}_{positive_idx}.npz')
            np.savez_compressed(
                file_name,
                pos_points_f=release_cuda(output_dict['pos_points_f']),
                anc_points_f=release_cuda(output_dict['anc_points_f']),
                pos_points_c=release_cuda(output_dict['pos_points_c']),
                anc_points_c=release_cuda(output_dict['anc_points_c']),
                # pos_feats_c=release_cuda(output_dict['pos_feats_c']),
                # anc_feats_c=release_cuda(output_dict['anc_feats_c']),
                pos_node_corr_indices=release_cuda(output_dict['pos_node_corr_indices']),
                anc_node_corr_indices=release_cuda(output_dict['anc_node_corr_indices']),
                pos_corr_points=release_cuda(output_dict['pos_corr_points']),
                anc_corr_points=release_cuda(output_dict['anc_corr_points']),
                corr_scores=release_cuda(output_dict['corr_scores']),
                gt_node_corr_indices=release_cuda(output_dict['gt_node_corr_indices']),
                gt_node_corr_overlaps=release_cuda(output_dict['gt_node_corr_overlaps']),
                estimated_transform=release_cuda(output_dict['estimated_transform']),
                transform=release_cuda(data_dict['transform']),
                pos_feature_global=pos_feature_global,
                anc_feature_global=anc_feature_global,
            )

def main():
    cfg = make_cfg()
    
    tester = Tester(cfg)
    snapshots='./weights/best-model-mixed.tar'
    tester.run(snapshots)



if __name__ == '__main__':
    main()