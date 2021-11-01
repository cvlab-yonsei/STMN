import numpy as np
import torch
import torch.nn.functional as F
import sys
import pandas as pd
from progressbar import ProgressBar, AnimatedMarker, Percentage
import math
from tqdm import trange

def Video_Cmc(features, ids, cams, query_idx, rank_size):
    """
    features: numpy array of shape (n, d)
    label`s: numpy array of shape (n)
    """
    # Sample query
    data = {'feature':features, 'id':ids, 'cam':cams}

    q_idx = query_idx
    g_idx = [i for i in range(len(ids)) if data['id'][i]!=-1]
    q_data = {k:v[q_idx] for k, v in data.items()}
    g_data = {k:v[g_idx] for k, v in data.items()}
    if len(g_idx) < rank_size: rank_size = len(g_idx)

    CMC, mAP = Cmc(q_data, g_data, rank_size)
    return CMC, mAP

def Cmc(q_data, g_data, rank_size):

    n_query = q_data['feature'].shape[0]
    n_gallery = g_data['feature'].shape[0]

    distmat = np_cdist(q_data['feature'], g_data['feature']) # Reture a n_query*n_gallery array
    index = np.argsort(distmat, axis=1) # from small to large

    num_no_gt = 0 # num of query imgs without groundtruth
    num_r1 = 0
    CMC = np.zeros(n_gallery)
    AP = 0

    for i in range(n_query):
        # groundtruth index
        query_index = np.argwhere(g_data['id']==q_data['id'][i])
        camera_index = np.argwhere(g_data['cam']==q_data['cam'][i])
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        if good_index.size == 0:
            num_no_gt += 1
            continue
        # remove gallery samples that have the same pid and camid with query
        junk_index = np.intersect1d(query_index, camera_index)

        ap_tmp, CMC_tmp = Compute_AP(good_index, junk_index, index[i])
        if CMC_tmp[0]==1:
            num_r1 += 1
        CMC = CMC + CMC_tmp
        AP += ap_tmp

    if num_no_gt > 0:
        print("{} query imgs do not have groundtruth.".format(num_no_gt))

    # print("R1:{}".format(num_r1))

    CMC = CMC / (n_query - num_no_gt)
    mAP = AP / (n_query - num_no_gt)

    return CMC, mAP

def Compute_AP(good_index, junk_index, index):
    ap = 0
    cmc = np.zeros(len(index))

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1.0
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        ap = ap + d_recall*precision

    return ap, cmc

def cdist(feat1, feat2):
    """Cosine distance"""
    feat1 = torch.FloatTensor(feat1)#.cuda()
    feat2 = torch.FloatTensor(feat2)#.cuda()
    feat1 = torch.nn.functional.normalize(feat1, dim=1)
    feat2 = torch.nn.functional.normalize(feat2, dim=1).transpose(0, 1)
    dist = -1 * torch.mm(feat1, feat2)
    return dist.cpu().numpy()

def np_cdist(feat1, feat2):
    """Cosine distance"""
    feat1_u = feat1 / np.linalg.norm(feat1, axis=1, keepdims=True) # n * d -> n
    feat2_u = feat2 / np.linalg.norm(feat2, axis=1, keepdims=True) # n * d -> n
    return -1 * np.dot(feat1_u, feat2_u.T)

def np_norm_eudist(feat1,feat2):
    feat1_u = feat1 / np.linalg.norm(feat1, axis=1, keepdims=True) # n * d -> n
    feat2_u = feat2 / np.linalg.norm(feat2, axis=1, keepdims=True) # n * d -> n
    feat1_sq = np.sum(feat1_M * feat1, axis=1)
    feat2_sq = np.sum(feat2_M * feat2, axis=1)
    return np.sqrt(feat1_sq.reshape(-1,1) + feat2_sq.reshape(1,-1) - 2*np.dot(feat1_M, feat2.T)+ 1e-12)


def sqdist(feat1, feat2, M=None):
    """Mahanalobis/Euclidean distance"""
    if M is None: M = np.eye(feat1.shape[1])
    feat1_M = np.dot(feat1, M)
    feat2_M = np.dot(feat2, M)
    feat1_sq = np.sum(feat1_M * feat1, axis=1)
    feat2_sq = np.sum(feat2_M * feat2, axis=1)
    return feat1_sq.reshape(-1,1) + feat2_sq.reshape(1,-1) - 2*np.dot(feat1_M, feat2.T)

if __name__ == '__main__':
    from scipy.io import loadmat
    q_feature = loadmat(sys.argv[1])['ff']
    q_db_txt = sys.argv[2]
    g_feature = loadmat(sys.argv[3])['ff']
    g_db_txt = sys.argv[4]
    #print(feature.shape)
    CMC, mAP = Self_Cmc(g_feature, g_db_txt, 100)
    #CMC, mAP = Vanilla_Cmc(q_feature, q_db_txt, g_feature, g_db_txt)
    print('r1 precision = %f, mAP = %f' % (CMC[0], mAP))
