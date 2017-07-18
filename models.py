import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

from util.modules import Dense



class EncoderDecoder(nn.Module):

    """A generic encoder/decoder level."""

    def __init__(self, n_in, n_units, n_out, n_layers, non_linearity, connection_type='regular', weight_norm=False):
        super(EncoderDecoder, self).__init__()
        self.connection_type = connection_type
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            layer = Dense(n_in, n_units, non_linearity, weight_norm)
            self.layers.append(layer)
            n_in = n_units
        self.mean = nn.Linear(n_in, n_out)
        self.log_var = nn.Linear(n_in, n_out)

    def forward(self, input):
        for layer in self.layers:
            if self.connection_type == 'regular':
                input = layer(input)
            elif self.connection_type == 'residual':

        return self.mean(input), self.log_var(input)



class LatentVariableModel(nn.Module):
    def __init__(self):
        super(LatentVariableModel, self).__init__()

        # define the basic function blocks


    def forward(self):
        pass