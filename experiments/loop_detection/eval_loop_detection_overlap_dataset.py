import glob
import os.path as osp
import time

import numpy as np
from tqdm import tqdm

from experiments.lcrnet.config_ld import make_cfg
import faiss
import matplotlib.pyplot as plt
from sklearn import metrics
import argparse

def compute_AP(precision, recall):
    ap = 0.
    for i in range(1, len(precision)):
        ap += (recall[i] - recall[i-1])*precision[i]
    return ap

def compute_F1(precision, recall):
    precision = np.asarray(precision)
    recall = np.asarray(recall)
    F1 = 2*precision*recall/(precision+recall)
    idx = F1.argmax()
    F1 = F1.max()
    
    return F1, idx

def compute_topN(prediction_file_name, ground_truth_file_name, topn):
    # loading overlap predictions
    des_dists = np.load(prediction_file_name)['arr_0']
    des_dists = np.asarray(des_dists, dtype='float32')
    des_dists = des_dists.reshape((len(des_dists), 3))


    # loading ground truth in terms of distance
    try:
        ground_truth = np.load(ground_truth_file_name, allow_pickle='True')['arr_0']
    except:
        ground_truth = np.load(ground_truth_file_name, allow_pickle='True')['data']


    all_have_gt = 0
    tps = 0

    for idx in range(0,len(ground_truth)-1):
        gt_idxes = ground_truth[int(idx)]

        if not gt_idxes.any():
            continue

        all_have_gt += 1
        for t in range(topn):
            if des_dists[des_dists[:,0]==int(idx),:][t, 1] in gt_idxes:
                tps += 1
                break

    recall_topN = tps/all_have_gt
    # print(recall_topN)


    return recall_topN



def compute_PR_overlap(pair_dist, ground_truth_file_name, thre_range=[0,1], interval=0.01):
    precisions = []
    recalls = []

    print('Load descrptor distances predictions with pairs of ', len(pair_dist))
    print(pair_dist.shape)

    try:
        ground_truth = np.load(ground_truth_file_name, allow_pickle='True')['arr_0']
    except:
        ground_truth = np.load(ground_truth_file_name, allow_pickle='True')['data']

    """Change the threshold will lead to different test results"""
    for thres in np.arange(thre_range[0], thre_range[1], interval):
        tps = 0
        fps = 0
        tns = 0
        fns = 0
        """Update the start frame index"""
        for idx in range(150,len(ground_truth)-1):
            gt_idxes = ground_truth[int(idx)]
            reject_flag = False

            if pair_dist[pair_dist[:,0]==int(idx),2][0]>thres:
                reject_flag = True
            if reject_flag:
                if not gt_idxes.any():
                    tns += 1
                else:
                    fns += 1
            else:
                if pair_dist[pair_dist[:,0]==int(idx),1][0] in gt_idxes:
                    tps += 1
                else:
                    fps += 1

        if fps == 0:
            precision = 1
        else:
            precision = float(tps) / (float(tps) + float(fps))
        if fns == 0:
            recall = 1
        else:
            recall = float(tps) / (float(tps) + float(fns))

        precisions.append(precision)
        recalls.append(recall)

        F1 = 2*precision*recall/(precision+recall+1e-12)

        message = "thresh: %.3f "%thres + "   precision: %.3f "%precision + "   recall: %.3f "%recall + "   F1: %.3f "%F1
        print(message)
        if recall == 1:
            break

    return precisions, recalls

"""Ploting and saving AUC."""
def plotPRC(precisions, recalls, F1, recall_list, print_2file=False, dataset='kitti'):
    # initial plot
    plt.clf()

    save_name = './PRC.png'

    recalls, precisions = (list(t) for t in zip(*sorted(zip(recalls, precisions), reverse=True)))
    auc = metrics.auc(recalls, precisions) * 100

    plt.plot(recalls, precisions, linewidth=1.0)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1])
    plt.xlim([0.0, 1])
    plt.title('auc=%.3f '%auc + 'F1=%.3f '%F1 + 'top1=%.3f '%recall_list[0] + 'top%%1=%.3f '%recall_list[1])

    if print_2file:
        plt.savefig(save_name)

    # plt.show()
    return auc


def eval_one_epoch(cfg, seqlist=[0]):
    features_root = cfg.ld_feature_dir+cfg.dataset

    for seq in seqlist:

        pred_file = features_root+"/predicted_des_L2_dis.npz"
        if cfg.dataset=='kitti':
            gt_file = cfg.data.dataset_root + '/overlap/loop_gt_seq00_0.3overlap_inactive.npz'
        elif cfg.dataset=='ford':
            gt_file = cfg.data.dataset_ford_root + '/overlap/loop_gt_seq00_0.3overlap_inactive.npz'

        if not osp.exists(pred_file):
        # if True:

            file_names = sorted(
                glob.glob(osp.join(features_root, '%d*.npz'%seq)),
                key=lambda x: int(osp.splitext(osp.basename(x))[0]),
            )

            emb_list_map = []
            num_test_pairs = len(file_names)
            for i, file_name in tqdm(enumerate(file_names), total=num_test_pairs):

                data_dict = np.load(file_name)
                anc_global = data_dict['anc_global']

                emb_list_map.append(anc_global.astype(np.float32))
            emb_list_map = np.concatenate(emb_list_map)

            # emb_list_map = emb_list_map / np.linalg.norm(emb_list_map, axis=1, keepdims=True)
            
            print("Processing pairwise descriptor distance---->")
            row_list = []
            comps = []
            querys = []
            for i in range(101, emb_list_map.shape[0]-1):
                a = time.time()
                nlist = 1
                k = 50
                d = 256
                quantizer = faiss.IndexFlatL2(d)
                index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
                assert not index.is_trained
                index.train(emb_list_map[:i-100,:])
                assert index.is_trained
                index.add(emb_list_map[:i-100,:])
                b = time.time()
                # plt.clf()
                """Faiss searching"""
                D, I = index.search(emb_list_map[i, :].reshape(1, -1), k)
                c = time.time()
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
                comp = b-a
                query = c-b
                comps.append(comp)
                querys.append(query)
            print(np.array(comps).mean(), np.array(querys).mean())

            pair_dist = np.array(row_list)
            """Saving for the next test"""
            np.savez_compressed(features_root+"/predicted_des_L2_dis", pair_dist)
            
            pair_dist = np.asarray(pair_dist, dtype='float32')
            pair_dist = pair_dist.reshape((len(pair_dist), 3))
        else:
            pair_dist = np.load(pred_file)['arr_0']
            pair_dist = np.asarray(pair_dist, dtype='float32')
            pair_dist = pair_dist.reshape((len(pair_dist), 3))

        # emb_list_map_norm = emb_list_map / np.linalg.norm(emb_list_map, axis=1, keepdims=True)
        # pair_dist = faiss.pairwise_distances(emb_list_map_norm, emb_list_map_norm)

        thre_range=[0,1]
        interval=0.01
        precision_ours_fp, recall_ours_fp = compute_PR_overlap(pair_dist, gt_file, thre_range, interval)
        ap_ours_fp = compute_AP(precision_ours_fp, recall_ours_fp)
        F1, idx = compute_F1(precision_ours_fp, recall_ours_fp)
        message = "F1_score on KITTI test seq: %.3f "%F1
        print(message)
    

        if cfg.dataset=='ford':
            x=38
        elif cfg.dataset=='kitti':
            x=45

        
        topn = np.array([1,x]) # for KITTI 00 top1%
        recall_list = []
        message = ''
        for i in range(0, topn.shape[0]):
            # print("top"+str(topn[i])+": ")
            rec = compute_topN(pred_file, gt_file, topn[i])
            recall_list.append(rec)
            message += "top"+str(topn[i])+" recall: %.3f  "%rec
        print(message)


        message = ''
        auc = plotPRC(precision_ours_fp, recall_ours_fp, F1, recall_list, True, cfg.dataset)
        message += "AUC: %.3f "%auc+"   Average Precision: %.3f "%ap_ours_fp
        print(message)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='kitti', help='kitti ford')
    parser.add_argument('--seq', type=int, nargs='+', default=[0], help='')
    return parser

def main():

    cfg = make_cfg()
    args = parser().parse_args()

    cfg.dataset = args.dataset
    cfg.seq = args.seq
    
    eval_one_epoch(cfg, cfg.seq)


if __name__ == '__main__':
    main()
