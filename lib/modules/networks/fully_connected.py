import torch
import torch.nn as nn
from lib.modules.layers import FullyConnectedLayer
from network import Network


class FullyConnectedNetwork(Network):
    """
    Fully-connected neural network, i.e. multi-layered perceptron.

    Args:
        network_config (dict): dictionary containing network configuration parameters,
                               keys should include n_in, n_units, n_layers,
                               non_linearity, connection_type, batch_norm,
                               weight_norm, dropout
    """
    def __init__(self, network_config):
        super(FullyConnectedNetwork, self).__init__(network_config)
        self._construct(**network_config)

    def _construct(self, n_in, n_units, connection_type='sequential',
                   non_linearity=None, batch_norm=False, weight_norm=False, dropout=0.):
        """
        Method to construct the network from the network_config dictionary parameters.
        """
        self.layers = nn.ModuleList([])
        self.gates = nn.ModuleList([])

        connection_types = ['sequential', 'residual', 'highway', 'concat_input', 'concat']
        assert network_config['connection_type'] in connection_types, 'Connection type not found.'
        self.connection_type = network_config['connection_type']

        n_in_orig = n_in
        output_size = 0

        if self.connection_type in ['residual', 'highway']:
            # intial linear layer to embed to correct size
            self.initial_fc = FullyConnectedLayer({'n_in': n_in,
                                                   'n_out': n_units,
                                                   'batch_norm': batch_norm,
                                                   'weight_norm': weight_norm})

        for _ in range(network_config['n_layers']):
            layer = FullyConnectedLayer({'n_in': n_in, 'n_out': n_units,
                                         'non_linearity': non_linearity,
                                         'batch_norm': batch_norm,
                                         'weight_norm': weight_norm,
                                         'dropout': dropout})
            self.layers.append(layer)
            if self.connection_type == 'highway':
                gate = FullyConnectedLayer({'n_in': n_in, 'n_out': n_units,
                                            'non_linearity': 'sigmoid',
                                            'batch_norm': batch_norm,
                                            'weight_norm': weight_norm})
                self.gates.append(gate)
            if self.connection_type in ['sequential', 'residual', 'highway']:
                n_in = n_units
            elif self.connection_type == 'concat_input':
                n_in = n_units + n_in_orig
            elif self.connection_type == 'concat':
                n_in += n_units
            output_size = n_in
        self.n_out = output_size

    def forward(self, input):
        """
        Method for forward computation.
        """
        input_orig = input.clone()
        for layer_num, layer in enumerate(self.layers):
            if self.connection_type == 'sequential':
                input = layer(input)
            elif self.connection_type == 'residual':
                if layer_num == 0:
                    input = self.initial_fc(input) + layer(input)
                else:
                    input = input + layer(input)
            elif self.connection_type == 'highway':
                gate = self.gates[layer_num]
                if layer_num == 0:
                    input = gate(input) * self.initial_fc(input) + (1 - gate(input)) * layer(input)
                else:
                    input = gate(input) * input + (1 - gate(input)) * layer(input)
            elif self.connection_type == 'concat_input':
                input = torch.cat((input_orig, layer(input)), dim=1)
            elif self.connection_type == 'concat':
                input = torch.cat((input, layer(input)), dim=1)
        return input
