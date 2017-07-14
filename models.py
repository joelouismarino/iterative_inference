import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms


class Fully_Connected(nn.Module):

    def __init__(self, n_in, n_out, non_linearity):
        super(Fully_Connected, self).__init__()
        self.linear = nn.Linear(n_in, n_out)
        if non_linearity == 'relu':
            self.non_linearity = nn.ReLU()
        elif non_linearity == 'elu':
            self.non_linearity = nn.ELU()
        elif non_linearity == 'selu':
            self.non_linearity = nn.SELU()
        elif non_linearity == 'tanh':
            self.non_linearity = nn.Tanh()
        elif non_linearity == 'sigmoid':
            self.non_linearity = nn.Sigmoid()
        else:
            raise Exception('Non-linearity ' + str(non_linearity) + ' not found.')

    def forward(self, x):
        return self.non_linearity(self.linear(x))


class Encoder_Decoder(nn.Module):

    """A generic encoder/decoder level."""

    def __init__(self, n_in, n_out, n_layers, non_linearity, output_shape, connection_type='regular'):
        super(Encoder_Decoder, self).__init__()
        self.connection_type = connection_type
        self.layers = nn.ModuleList([])
        for layer in range(n_layers):
            self.layers.append(Fully_Connected())

    def forward(self, input):





class LatentVariableModel(nn.Module):
    def __init__(self):
        super(LatentVariableModel, self).__init__()

        # define the basic function blocks


    def forward(self):
        pass