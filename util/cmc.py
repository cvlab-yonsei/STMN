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
    g_idx = np.arange(len(ids))
    
    q_data = {k:v[q_idx] for k, v in data.items()}
    g_data = {k:v[g_idx] for k, v in data.items()}
    if len(g_idx) < rank_size: rank_size = len(g_idx)

    CMC, mAP = Cmc(q_data, g_data, rank_size)
    return CMC, mAP

def Cmc(q_data, g_data, rank_size):
    
    n_query = q_data['feature'].shape[0]
    n_gallery = g_data['feature'].shape[0]

    dist = np_cdist(q_data['feature'], g_data['feature']) # Reture a n_query*n_gallery array

    cmc = np.zeros((n_query, rank_size))
    ap = np.zeros(n_query)
    
    widgets = ["Calc. CMC! ", AnimatedMarker(markers='←↖↑↗→↘↓↙'), ' (', Percentage(), ')']
    pbar = ProgressBar(widgets=widgets, max_value=n_query)
    for k in range(n_query):
        
        good_idx = np.where((q_data['id'][k]==g_data['id']) & (q_data['cam'][k]!=g_data['cam']))[0]
        if len(good_idx) == 0: continue
        
        junk_mask1 = (g_data['id'] == -1)
        junk_mask2 = (q_data['id'][k]==g_data['id']) & (q_data['cam'][k]==g_data['cam'])
        junk_idx = np.where(junk_mask1 | junk_mask2)[0]
        
        score = dist[k, :]
        sort_idx = np.argsort(score)
        sort_idx = sort_idx[:rank_size]

        ap[k], cmc[k, :] = Compute_AP(good_idx, junk_idx, sort_idx)
        pbar.update(k)
        
    pbar.finish()
    CMC = np.mean(cmc, axis=0)
    mAP = np.mean(ap)
    return CMC, mAP

def Compute_AP(good_image, junk_image, index):
    
    cmc = np.zeros((len(index),))
    ngood = len(good_image)
    old_recall = 0
    old_precision = 1.
    ap = 0
    intersect_size = 0
    j = 0
    good_now = 0
    njunk = 0
    
    for n in range(len(index)):
        flag = 0
        if np.any(good_image == index[n]):
            cmc[n-njunk:] = 1
            flag = 1 # good image
            good_now += 1
        if np.any(junk_image == index[n]):
            njunk += 1
            continue # junk image
        
        if flag == 1:
            intersect_size += 1
        recall = intersect_size/ngood
        precision = intersect_size/(j+1)
        ap += (recall-old_recall) * (old_precision+precision) / 2
        old_recall = recall
        old_precision = precision
        j += 1
        
        if good_now == ngood:
            return ap, cmc
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
