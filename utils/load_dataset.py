import torch
import os
import numpy as np
from torch.utils.data.dataset import Dataset
import random
import h5py
from tqdm import tqdm
from torch.nn import functional

class background_appear_motion_dataset(Dataset):
    def __init__(self, dataset_path, num_cluster=13, removeMotionBranch=False, use_weights=False):
        self.removeMotionBranch = removeMotionBranch
        dataset = np.load(dataset_path, allow_pickle=True).tolist()
        self.backid_cls_sign = dataset['backid_cls_sign']  #labels of back_id, cls_id, sign
        self.appear_index_of_backid_cls = dataset['appear_index_of_backid_cls']
        self.appear_index_backid = dataset['appear_index_backid']
        self.appear_index_cls_in_backid = dataset['appear_index_cls_in_backid']
        self.appear_index_backid_in_all = dataset['appear_index_backid_in_all']
        self.appear_index = dataset['index_of_appear']

        if self.removeMotionBranch == False:
            self.motion_index_cls_in_backid = dataset['motion_index_cls_in_backid']
            self.motion_index_backid_in_all = dataset['motion_index_backid_in_all']
            self.motion_index_of_backid_cls = dataset['motion_index_of_backid_cls']
            self.motion_index_backid = dataset['motion_index_backid']
            self.motion_index = dataset['index_of_motion']

        if 'video_id' in dataset.keys():
            self.video_id = dataset['video_id']

        self.feat_all = dataset['feat_all'] #store all the feats
        self.num_cluster = num_cluster
        self.shuffle_keys()

        self.use_weights = use_weights
        if use_weights == True:
            self.weights_dict = dataset['weights_dict']

    def get_back_indexes(self, back_id, sign):
        if sign == 0:
            return np.array(self.appear_index_backid[back_id])
        return np.array(self.motion_index_backid[back_id])

    def get_index_cls_in_backid(self, back_id, cls_id, sign):
        if sign == 0:
            return np.array(self.appear_index_cls_in_backid[back_id][cls_id])
        return np.array(self.motion_index_cls_in_backid[back_id][cls_id])

    def get_indexes_back_in_all(self, back_id, sign):
        if sign == 0:
            return np.array(self.appear_index_backid_in_all[back_id])
        return np.array(self.motion_index_backid_in_all[back_id])

    def get_all_feats(self, sign):
        if sign == 0:
            return self.feat_all[self.appear_index, :]
        return self.feat_all[self.motion_index, :]

    def get_backid_cls_sign_dict(self, sign):
        if sign == 0:
            return np.array(self.backid_cls_sign)[self.appear_index]
        else:
            return np.array(self.backid_cls_sign)[self.motion_index]

    def get_video_id_array(self, sign):
        if sign == 0:
            return np.array(self.video_id)[self.appear_index]
        else:
            return np.array(self.video_id)[self.motion_index]

    def __len__(self):
        return len(self.backid_cls_sign)

    def shuffle_keys(self):
        self.iters = np.random.permutation(len(self.backid_cls_sign))

    def __getitem__(self, item):

        iter = self.iters[item]
        back_id, cls_id, sign = self.backid_cls_sign[iter]

        label = torch.tensor([int(back_id)])
        cluster_one_hot = functional.one_hot(label, num_classes=self.num_cluster).float()

        if self.use_weights == True:
            return torch.from_numpy(self.feat_all[iter]), cluster_one_hot, int(back_id), int(cls_id), iter, sign, self.weights_dict[cls_id]
        return torch.from_numpy(self.feat_all[iter]), cluster_one_hot, int(back_id), int(cls_id), iter, sign


class regressor_dataset(Dataset):
    def __init__(self, dataset_path):

        self.dataset = np.load(dataset_path, allow_pickle=True).tolist()
        self.process_feats()
        self.shuffle_keys()

    def process_feats(self):
        positive_feat_list = []
        negative_feat_list = []
        for key_type in self.dataset.keys():
            for action_label in self.dataset[key_type].keys():
                if len(self.dataset[key_type][action_label]) == 0:
                    continue
                if key_type == "positive_gen":
                    positive_feat_list.extend(self.dataset[key_type][action_label])
                if key_type == "negative":
                    negative_feat_list.extend(self.dataset[key_type][action_label])
        self.positive_feats = np.array(positive_feat_list)
        self.negative_feats = np.array(negative_feat_list)

    def __len__(self):
        return min(self.positive_feats.shape[0], self.negative_feats.shape[0])

    def shuffle_keys(self):
        self.iters_positive = np.random.permutation(self.positive_feats.shape[0])
        self.iters_negative = np.random.permutation(self.negative_feats.shape[0])

    def __getitem__(self, item):
        iters_positive = self.iters_positive[item]
        iters_negative = self.iters_negative[item]
        return torch.from_numpy(self.positive_feats[iters_positive]), torch.from_numpy(self.negative_feats[iters_negative])

