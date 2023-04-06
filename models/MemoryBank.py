import torch
from torch import nn
import numpy as np

class MemoryBank(nn.Module):
    def __init__(self, num_feats, feats_dim, momentum, dataset_path):
        super(MemoryBank, self).__init__()

        dataset = np.load(dataset_path, allow_pickle=True).tolist()
        bank = torch.zeros((num_feats, feats_dim), dtype=torch.float32, requires_grad=False).cuda()
        self.momentum = momentum
        self.index_of_appear = dataset['index_of_appear']
        self.index_of_motion = dataset['index_of_motion']

        self.register_buffer("bank", bank)

    def update(self, feats, indexes):
        feats = feats / (feats.norm(dim=1).view(-1, 1) + 1e-10)
        feats_old = self.bank[indexes, :]
        feats_new = (1 - self.momentum) * feats_old + self.momentum * feats
        feats = feats_new / (feats_new.norm(dim=1).view(-1, 1) + 1e-10)
        self.bank[indexes, :] = feats

    def get_elements(self, indexes):
        return self.bank[indexes, :]

    def get_all_elements(self, sign):
        if sign == 0:
            return self.bank[self.index_of_appear, :]
        return self.bank[self.index_of_motion, :]



