import torch
from torch import nn
from models.NonLinearLayers import NonlinearLayers

class DecoderModel(nn.Module):
    def __init__(self, input_dim, appear_output_dim, motion_output_dim, background_output_dim, hidden_dim, weight_init=False):
        super(DecoderModel, self).__init__()
        self.appear_decoder = NonlinearLayers(input_dim=input_dim, output_dim=appear_output_dim, weight_init=weight_init, hidden_dim=[hidden_dim[0]])
        self.motion_decoder = NonlinearLayers(input_dim=input_dim, output_dim=motion_output_dim, weight_init=weight_init, hidden_dim=[hidden_dim[1]])

    def forward(self, inputs, sign):
        if sign == 0:
            decoder_outputs = self.appear_decoder(inputs)
        elif sign == 1:
            decoder_outputs = self.motion_decoder(inputs)
        return decoder_outputs