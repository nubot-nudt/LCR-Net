import glob
import os.path as osp

import numpy as np
from tqdm import tqdm

from experiments.lcrnet.config_ld import make_cfg
import faiss
import matplotlib.pyplot as plt
import os
import argparse


def find_top1(cfg,seq,des_dists,N,thres=0.11):

    top1_with_thre = []
    for idx in tqdm(range(0,N-1)):

        dist = des_dists[des_dists[:,0]==int(idx)]
        if dist.shape[0]==0:
            continue
        
        if dist[dist[:,2]<thres].shape[0]==0:
            continue
        else:
            top1_with_thre.append(dist[dist[:,2]<thres])
    
    path = '%s/result/top1_with_thres_%.2f'%(cfg.data.dataset_root,thres)
    if osp.exists(path) is False:
        os.makedirs(path)

    file_name='%s/%02d.txt'%(path,seq)


    f = open(file_name,'a')
    for i in range(len(top1_with_thre)):
        for j in range(top1_with_thre[i].shape[0]):
            f.write(f'{int(top1_with_thre[i][j][0])} {int(top1_with_thre[i][j][1])} {(top1_with_thre[i][j][2])}  \n')
    f.close()

    print('top1 file saved in %s'%file_name)

    return 



def inference_one_epoch(cfg, eval_seq_list=[2],thres=0.11):
    '''
    0.11 for seq.02
    '''
    features_root = cfg.ld_feature_dir+cfg.dataset

    for seq in eval_seq_list:

        pred_file = features_root+"/predicted_des_L2_dis.npz"
       
        file_names = sorted(
            glob.glob(osp.join(features_root, '%d*.npz'%seq)),
            key=lambda x: [int(i) for i in osp.splitext(osp.basename(x))[0].split('_')],
        )

        num_test_pairs = len(file_names)
        emb_list_map = []
        if not osp.exists(pred_file):
        # if True:
            for i, file_name in tqdm(enumerate(file_names), total=num_test_pairs):

                # seq_id, idx = [int(x) for x in osp.splitext(osp.basename(file_name))[0].split('_')]
                data_dict = np.load(file_name)
                anc_global = data_dict['anc_global']

                emb_list_map.append(anc_global)
            emb_list_map = np.concatenate(emb_list_map)

            emb_list_map = emb_list_map / np.linalg.norm(emb_list_map, axis=1, keepdims=True)
            
            print("Processing pairwise descriptor distance---->")
            row_list = []
            for i in range(101, emb_list_map.shape[0]-1):
                nlist = 1
                k = 50
                d = 256
                quantizer = faiss.IndexFlatL2(d)
                index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
                assert not index.is_trained
                index.train(emb_list_map[:i-100,:])
                assert index.is_trained
                index.add(emb_list_map[:i-100,:])
                plt.clf()
                """Faiss searching"""
                D, I = index.search(emb_list_map[i, :].reshape(1, -1), k)
                for j in range(D.shape[1]):
                    """The nearest 100 frames are not considered."""
                    if (i-I[:,j])<100:
                        continue
                    else:
                        one_row = np.zeros((1,3))
                        one_row[:, 0] = i
                        one_row[:, 1] = I[:,j]
                        one_row[:, 2] = D[:,j]
                        row_list.append(one_row)
                        message = str(i) + "---->" + str(I[:, j]) + "  " + str(D[:, j])
                        # logger.info(message)
                        
            pair_dist = np.array(row_list)
            """Saving for the next test"""
            np.savez_compressed(features_root+"/predicted_des_L2_dis", pair_dist)
            
            pair_dist = np.asarray(pair_dist, dtype='float32')
            pair_dist = pair_dist.reshape((len(pair_dist), 3))
        else:
            pair_dist = np.load(pred_file)['arr_0']
            pair_dist = np.asarray(pair_dist, dtype='float32')
            pair_dist = pair_dist.reshape((len(pair_dist), 3))

        find_top1(cfg,seq,pair_dist, num_test_pairs,thres)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='kitti', help='kitti ford')
    parser.add_argument('--seq', type=int, default=[0], help='')
    return parser
 
def main():
    cfg = make_cfg()
    args = parser()

    cfg.dataset = args.parse_args().dataset
    cfg.seq = args.parse_args().seq
    
    inference_one_epoch(cfg, cfg.seq, 0.22)


if __name__ == '__main__':
    main()
