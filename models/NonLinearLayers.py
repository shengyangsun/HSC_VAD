import torch
from torch import nn

class NonlinearLayers(nn.Module):
    def __init__(self, input_dim, output_dim=512, weight_init=False, hidden_dim=[2048]):
        super(NonlinearLayers, self).__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden_dim[0], bias=False),
                                        nn.BatchNorm1d(hidden_dim[0]),
                                        nn.ReLU(inplace=False),
                                        nn.Linear(hidden_dim[0], output_dim, bias=False),
                                        nn.BatchNorm1d(output_dim))
        if weight_init == True:
            self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        outputs = self.layers(x)
        return outputs