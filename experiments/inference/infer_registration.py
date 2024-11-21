import os.path as osp
import time


from utils.engine import SingleTester
from utils.utils.common import ensure_dir, get_log_string,empty_dir
from utils.utils.torch import release_cuda

from experiments.lcrnet.dataset_loop_closure import infer_loop_closure_data_loader 

from experiments.lcrnet.model_family.LCRNet_Matching_infer import create_model
from experiments.lcrnet.config_reg import make_cfg


class Tester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg)


        cfg.dataset = self.args.dataset
        cfg.seq = self.args.seq
        cfg.vis = self.args.vis

        cfg.test.lc_text_root = self.args.lc_text_root
        cfg.output_dir = cfg.output_dir + cfg.test.lc_text_root

        start_time = time.time()
        data_loader, neighbor_limits = infer_loop_closure_data_loader(cfg, self.distributed, cfg.dataset)

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
        self.output_dir = osp.join(cfg.output_dir)
        ensure_dir(self.output_dir)
        # empty_dir(self.output_dir)

        self.lgr_time=[]
        self.ransac_time=[]

    def test_step(self, iteration, data_dict):
        output_dict = self.model(data_dict)
        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
        pass

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        seq_id = data_dict['seq_id']
        pos_idx = data_dict['pos_idx']
        anc_idx=data_dict['anc_idx']
        message = f'seq_id: {seq_id}, pos_idx: {pos_idx}, anc_idx: {anc_idx}'
        # message += ', ' + get_log_string(result_dict=result_dict)
        message += ', nCorr: {}'.format(output_dict['corr_scores'].shape[0])
        return message

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        seq_id = data_dict['seq_id']
        positive_idx = data_dict['pos_idx']
        anchor_idx = data_dict['anc_idx']


        f = open(osp.join(self.output_dir, '%s_pose'%seq_id),'a')
        estimated_transform=release_cuda(output_dict['estimated_transform'])
        M2=estimated_transform.reshape(-1)[:12]
        f.write(f'{positive_idx} {anchor_idx} {M2[0]:.6f} {M2[1]:.6f} {M2[2]:.6f} {M2[3]:.6f} {M2[4]:.6f} {M2[5]:.6f} {M2[6]:.6f} {M2[7]:.6f} {M2[8]:.6f} {M2[9]:.6f} {M2[10]:.6f} {M2[11]:.6f} \n')


        del data_dict, output_dict, seq_id
           

def main():

    cfg = make_cfg()
    

    tester = Tester(cfg)
    snapshot='./weights/best-model-reg.tar'
    # snapshot='./weights/best-model-lc.tar'
    # snapshot='./weights/best-model-mixed.tar'

    tester.run(snapshot)



if __name__ == '__main__':
    main()