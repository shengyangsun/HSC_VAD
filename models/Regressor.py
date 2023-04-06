import torch
from torch import nn

class Regressor(nn.Module):
    def __init__(self, input_feature_dim, dropout_rate=0.6, hidden_dim=512, weight_init=True):
        super(Regressor, self).__init__()
        self.regressor = nn.Sequential(nn.Linear(input_feature_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate),
                                        nn.Linear(hidden_dim, 32), nn.Dropout(dropout_rate),
                                        nn.Linear(32, 1), nn.Sigmoid())
        if weight_init == True:
            self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x=x.view([-1,x.shape[-1]])
        logits=self.regressor(x)
        return logits