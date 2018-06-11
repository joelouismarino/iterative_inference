import torch
import torch.nn as nn
from torch.nn import init
from layer import Layer


class FullyConnectedLayer(Layer):
    """
    Fully-connected neural network layer.

    Args:
        layer_config (dict): dictionary containing layer configuration parameters
                             (see _construct method for keywords)
    """
    def __init__(self, layer_config):
        super(FullyConnectedLayer, self).__init__(layer_config)
        self._construct(**layer_config)

    def _construct(self, n_in, n_out, non_linearity=None, batch_norm=False,
                   weight_norm=False, dropout=0., initialize='glorot_normal'):
        """
        Method to construct the layer from the layer_config dictionary parameters.
        """
        self.linear = nn.Linear(n_in, n_out)
        self.bn = lambda x: x
        if batch_norm:
            self.bn = nn.BatchNorm1d(layer_config['n_out'], momentum=0.99)
        if weight_norm:
            self.linear = nn.utils.weight_norm(self.linear, name='weight')

        init_gain = 1.
        if non_linearity:
            if non_linearity == 'linear':
                self.non_linearity = lambda x: x
            elif non_linearity == 'relu':
                self.non_linearity = nn.ReLU()
                init_gain = init.calculate_gain('relu')
            elif non_linearity == 'leaky_relu':
                self.non_linearity = nn.LeakyReLU()
            elif non_linearity == 'elu':
                self.non_linearity = nn.ELU()
            elif non_linearity == 'selu':
                self.non_linearity = nn.SELU()
            elif non_linearity == 'tanh':
                self.non_linearity = nn.Tanh()
                init_gain = init.calculate_gain('tanh')
            elif non_linearity == 'sigmoid':
                self.non_linearity = nn.Sigmoid()
            else:
                raise Exception('Non-linearity ' + str(non_linearity) + ' not found.')
        else:
            self.non_linearity = lambda x: x

        self.dropout = lambda x: x
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

        if initialize == 'normal':
            init.normal_(self.linear.weight)
        elif initialize == 'glorot_uniform':
            init.xavier_uniform_(self.linear.weight, gain=init_gain)
        elif initialize == 'glorot_normal':
            init.xavier_normal_(self.linear.weight, gain=init_gain)
        elif initialize == 'kaiming_uniform':
            init.kaiming_uniform_(self.linear.weight)
        elif initialize == 'kaiming_normal':
            init.kaiming_normal_(self.linear.weight)
        elif initialize == 'orthogonal':
            init.orthogonal_(self.linear.weight, gain=init_gain)
        elif initialize == '':
            pass
        else:
            raise Exception('Parameter initialization ' + str(initialize) + ' not found.')

        init.constant_(self.linear.bias, 0.)

        if batch_norm:
            init.normal_(self.bn.weight, 1, 0.02)
            init.constant_(self.bn.bias, 0.)

    def forward(self, input):
        """
        Method to perform forward computation.
        """
        output = self.linear(input)
        output = self.bn(output)
        output = self.non_linearity(output)
        output = self.dropout(output)
        return output
