import torch
import torch.nn as nn
from torch.autograd import Variable

from distributions import DiagonalGaussian


class Dense(nn.Module):

    """Fully-connected layer with batch normalization, non-linearity, and weight normalization."""

    def __init__(self, n_in, n_out, non_linearity=None, batch_norm=False, weight_norm=False):
        super(Dense, self).__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.bn = None
        if batch_norm:
            self.bn = nn.BatchNorm1d(n_out)
        if weight_norm:
            self.linear = nn.utils.weight_norm(self.linear, name='weight')

        if non_linearity is None:
            self.non_linearity = None
        elif non_linearity == 'relu':
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

    def forward(self, input):
        output = self.linear(input)
        if self.bn:
            output = self.bn(output)
        if self.non_linearity:
            output = self.non_linearity(output)
        return output


class MLP(nn.Module):

    """Multi-layered perceptron."""

    def __init__(self, n_in, n_units, n_out, n_layers, non_linearity=None, connection_type='sequential', batch_norm=False, weight_norm=False):

        super(MLP, self).__init__()
        assert connection_type in ['sequential', 'residual', 'highway', 'concat_input', 'concat'], 'Connection type not found.'
        self.connection_type = connection_type
        self.layers = nn.ModuleList([])
        self.gates = nn.ModuleList([])

        n_in_orig = n_in

        if self.connection_type in ['residual', 'highway']:
            self.initial_dense = Dense(n_in, n_units, batch_norm=batch_norm, weight_norm=weight_norm)

        for _ in range(n_layers):
            layer = Dense(n_in, n_units, non_linearity=non_linearity, batch_norm=batch_norm, weight_norm=weight_norm)
            self.layers.append(layer)

            if self.connection_type == 'highway':
                gate = Dense(n_in, n_units, non_linearity='sigmoid', batch_norm=batch_norm, weight_norm=weight_norm)
                self.gates.append(gate)

            if self.connection_type in ['sequential', 'residual', 'highway']:
                n_in = n_units
            elif self.connection_type == 'concat_input':
                n_in = n_units + n_in_orig
            elif self.connection_type == 'concat':
                n_in += n_units

    def forward(self, input):

        input_orig = input.clone()

        for layer_num, layer in enumerate(self.layers):
            if self.connection_type == 'sequential':
                input = layer(input)

            elif self.connection_type == 'residual':
                if layer_num == 0:
                    input = self.initial_dense(input) + layer(input)
                else:
                    input += layer(input)

            elif self.connection_type == 'highway':
                gate = self.gates[layer_num]
                if layer_num == 0:
                    input = gate * self.initial_dense(input) + (1 - gate) * layer(input)
                else:
                    input = gate * input + (1 - gate) * layer(input)

            elif self.connection_type == 'concat_input':
                input = torch.cat((input_orig, layer(input)), dim=1)

            elif self.connection_type == 'concat':
                input = torch.cat((input, layer(input)), dim=1)

        return input


class LatentLevel(object):

    def __init__(self, batch_size, encoder_arch, decoder_arch, n_latent, n_det_enc, n_det_dec):

        self.batch_size = batch_size
        self.n_latent = n_latent

        self.posterior = DiagonalGaussian(Variable(torch.zeros(self.batch_size, self.n_latent)), Variable(torch.zeros(self.batch_size, self.n_latent)))
        self.prior = None

        self.encoder = MLP(**encoder_arch)
        self.decoder = MLP(**decoder_arch)

        if n_det_enc > 0:
            self.det_enc = nn.Linear(encoder_arch['n_units'], n_det_enc)

        if n_det_dec > 0:
            self.det_dec = nn.Linear()

    def reset_posterior(self):
        self.posterior = DiagonalGaussian(Variable(torch.zeros(self.batch_size, self.n_latent)), Variable(torch.zeros(self.batch_size, self.n_latent)))

    def up(self, ):
        pass

    def down(self, input, generative=False):



        if generative:
            sample = self.prior.sample()
        else:
            sample = self.posterior.sample()



        return self.decoder(sample)










def temp(a, b=1):
    return a + b