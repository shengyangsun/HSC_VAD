import torch
from torch import nn
from models.NonLinearLayers import NonlinearLayers
from models.LinearLayer import LinearLayer

class HSCModel(nn.Module):
    def __init__(self, appear_input_dim, motion_input_dim, output_dim=512, num_cluster=13, removeClusterHead=False, weight_init=False):
        super(HSCModel, self).__init__()
        self.removeClusterHead = removeClusterHead
        self.appear_encoder = NonlinearLayers(input_dim=appear_input_dim, output_dim=output_dim, weight_init=weight_init, hidden_dim=[2614,2048])
        self.motion_encoder = NonlinearLayers(input_dim=motion_input_dim, output_dim=output_dim, weight_init=weight_init, hidden_dim=[2102,1792])
        if self.removeClusterHead == False:
            self.clusterHead = LinearLayer(input_dim=output_dim, output_dim=num_cluster, weight_init=weight_init)

    def forward(self, inputs, sign):
        if sign == 0:
            encoder_outputs = self.appear_encoder(inputs)
        else:
            encoder_outputs = self.motion_encoder(inputs)
        outputs_norm = encoder_outputs / (encoder_outputs.norm(dim=1).view([-1, 1]) + 1e-10)
        if self.removeClusterHead == True:
            return outputs_norm
        cluster_outputs = self.clusterHead(outputs_norm)
        return outputs_norm, cluster_outputs
