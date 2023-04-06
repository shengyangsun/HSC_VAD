import os
import time
import random
import math
import numpy as np
from sklearn import metrics
import torch
from torch.nn import functional as F

Avenue_n_clips = {'01': 89, '02': 75, '03': 57, '04': 59, '05': 62, '06': 80, '07': 37, '08': 2, '09': 73, '10': 52, '11': 29, '12': 79, '13': 34, '14': 31, '15': 62, '16': 46, '17': 26, '18': 18, '19': 15, '20': 17, '21': 4}
ShanghaiTech_n_clips = {'01_0014': 16, '01_0015': 27, '01_0016': 21, '01_0025': 37, '01_0026': 31, '01_0027': 25, '01_0028': 28, '01_0029': 19, '01_0030': 25, '01_0051': 21, '01_0052': 21, '01_0053': 28, '01_0054': 36, '01_0055': 19, '01_0056': 33, '01_0063': 12, '01_0064': 18, '01_0073': 18, '01_0076': 16, '01_0129': 15, '01_0130': 21, '01_0131': 18, '01_0132': 16, '01_0133': 13, '01_0134': 27, '01_0135': 25, '01_0136': 33, '01_0138': 19, '01_0139': 13, '01_0140': 15, '01_0141': 19, '01_0162': 12, '01_0163': 16, '01_0177': 19, '02_0128': 28, '02_0161': 21, '02_0164': 22, '03_0031': 33, '03_0032': 25, '03_0033': 19, '03_0035': 24, '03_0036': 28, '03_0039': 30, '03_0041': 28, '03_0059': 27, '03_0060': 24, '03_0061': 15, '04_0001': 34, '04_0003': 58, '04_0004': 54, '04_0010': 31, '04_0011': 19, '04_0012': 22, '04_0013': 22, '04_0046': 33, '04_0050': 21, '05_0017': 27, '05_0018': 30, '05_0019': 40, '05_0020': 40, '05_0021': 25, '05_0022': 21, '05_0023': 48, '05_0024': 27, '06_0144': 15, '06_0145': 13, '06_0147': 16, '06_0150': 16, '06_0153': 13, '06_0155': 16, '07_0005': 25, '07_0006': 24, '07_0007': 30, '07_0008': 28, '07_0009': 19, '07_0047': 37, '07_0048': 15, '07_0049': 30, '08_0044': 19, '08_0058': 21, '08_0077': 28, '08_0078': 13, '08_0079': 15, '08_0080': 18, '08_0156': 21, '08_0157': 19, '08_0158': 21, '08_0159': 16, '08_0178': 16, '08_0179': 21, '09_0057': 22, '10_0037': 27, '10_0038': 15, '10_0042': 27, '10_0074': 37, '10_0075': 31, '11_0176': 21, '12_0142': 37, '12_0143': 16, '12_0148': 19, '12_0149': 15, '12_0151': 18, '12_0152': 22, '12_0154': 24, '12_0173': 13, '12_0174': 21, '12_0175': 16}
ped2_n_clips = {'01': 11, '02': 11, '03': 9, '04': 11, '05': 9, '06': 11, '07': 11, '08': 11, '09': 7, '10': 9, '11': 11, '12': 11}

def get_n_clips(dataset_name):
    if dataset_name == "Avenue":
        return Avenue_n_clips
    if dataset_name == "ShanghaiTech":
        return ShanghaiTech_n_clips
    if dataset_name == "UCSD_ped2":
        return ped2_n_clips

def get_CE_loss(outputs, labs):
    loss = F.cross_entropy(outputs, labs)
    return loss

def set_seeds(seed):
    print('Set seed is {}'.format(seed))
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def cal_auc(scores,labels):
    fpr, tpr, thresholds = metrics.roc_curve(labels,scores,pos_label=1)
    auc = metrics.auc(fpr,tpr)
    return auc

def eval(total_scores,total_labels,logger):
    total_scores = np.array(total_scores)
    total_labels = np.array(total_labels)
    auc = cal_auc(total_scores, total_labels)
    return auc

def gaussian_filter(support, sigma):
    mu = support[len(support) // 2 - 1]
    filter = 1.0 / (sigma * np.sqrt(2 * math.pi)) * np.exp(-0.5 * ((support - mu) / sigma) ** 2)
    return filter

def calc_l2_error(tensor1, tensor2):
    res = torch.sum(torch.pow(tensor1-tensor2, exponent=2), dim=1)
    return res

def min_max_norm(list_scores):
    np_array = np.array(list_scores)
    np_min = np.min(np_array[np_array > 0])
    np_max = np.max(np_array[np_array > 0])
    np_array = (np_array - np_min) / (np_max - np_min)
    np_array[np_array < 0] = 0
    return np_array
