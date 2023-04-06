import torch
from torch import nn

class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, weight_init=False):
        super(LinearLayer, self).__init__()

        self.layer = nn.Sequential(nn.Linear(input_dim, output_dim),
                                         nn.Softmax(dim=-1))
        if weight_init == True:
            self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        outputs = self.layer(x)
        return outputs