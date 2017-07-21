import torch
import torch.nn as nn
from torch.autograd import Variable

from distributions import DiagonalGaussian


class Dense(nn.Module):

    """Fully-connected (dense) layer with optional batch normalization, non-linearity, and weight normalization."""

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


class MultiLayerPerceptron(nn.Module):

    """Multi-layered perceptron."""

    def __init__(self, n_in, n_units, n_layers, non_linearity=None, connection_type='sequential', batch_norm=False, weight_norm=False):

        super(MultiLayerPerceptron, self).__init__()
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


class GaussianVariable(object):

    def __init__(self, batch_size, n_variables, n_input, update_form):

        self.batch_size = batch_size
        self.n_variables = n_variables
        self.update_form = update_form
        assert update_type in ['direct', 'highway'], 'Variable update type not found.'

        self.prior_mean = Dense(n_input[1], self.n_variables)
        self.prior_log_var = Dense(n_input[1], self.n_variables)
        self.posterior_mean = Dense(n_input[0], self.n_variables)
        self.posterior_log_var = Dense(n_input[0], self.n_variables)

        if self.update_form == 'highway':
            self.posterior_mean_gate = Dense(n_up_input, self.n_variables, 'sigmoid')
            self.posterior_log_var_gate = Dense(n_up_input, self.n_variables, 'sigmoid')

        self.prior = self.init_dist()
        self.posterior = self.init_dist()

    def init_dist(self):
        return DiagonalGaussian(Variable(torch.zeros(self.batch_size, self.n_variables)),
                                Variable(torch.zeros(self.batch_size, self.n_variables)))

    def encode(self, input):
        # encode the mean and log variance, update, return sample
        mean, log_var = self.posterior_mean(input), self.posterior_log_var(input)
        if self.update_form == 'highway':
            mean_gate = self.posterior_mean_gate(input)
            log_var_gate = self.posterior_log_var_gate(input)
            mean = mean_gate * self.posterior.mean + (1 - mean_gate) * mean
            log_var = log_var_gate * self.posterior.log_var + (1 - log_var_gate) * log_var
        self.posterior.mean, self.posterior.log_var = mean, log_var
        return self.posterior.sample()

    def decode(self, input, generate=False):
        # decode the mean and log variance, update, return sample
        mean, log_var = self.prior_mean(input), self.prior_log_var(input)
        self.prior.mean, self.prior.log_var = mean, log_var
        sample = self.prior.sample() if generate else self.posterior.sample()
        return sample

    def error(self):
        return self.posterior.sample() - self.prior.mean

    def norm_error(self):
        return self.error() / torch.exp(self.prior.log_var)

    def KL_divergence(self):
        return self.posterior.log_prob(self.posterior.sample()) - self.prior.log_prob(self.posterior.sample())

    def reset(self):
        self.posterior.reset()

    def cuda(self, device_id=0):
        # place all modules on the GPU
        self.prior_mean.cuda(device_id)
        self.prior_log_var.cuda(device_id)
        self.posterior_mean.cuda(device_id)
        self.posterior_log_var.cuda(device_id)
        if self.update_form == 'highway':
            self.posterior_mean_gate.cuda(device_id)
            self.posterior_log_var_gate.cuda(device_id)
        self.prior.cuda(device_id)
        self.posterior.cuda(device_id)


class LatentLevel(object):

    def __init__(self, batch_size, encoder_arch, decoder_arch, n_latent, n_det, encoding_form, variable_input_sizes, variable_update_form):

        self.batch_size = batch_size
        self.n_latent = n_latent
        self.encoding_form = encoding_form

        self.encoder = MultiLayerPerceptron(**encoder_arch)
        self.decoder = MultiLayerPerceptron(**decoder_arch)
        self.latent = GaussianVariable(self.batch_size, self.n_latent, variable_input_sizes, variable_update_form)
        self.deterministic_encoder = Dense(variable_input_sizes[0], n_det[0]) if n_det[0] > 0 else None
        self.deterministic_decoder = Dense(variable_input_sizes[1], n_det[1]) if n_det[1] > 0 else None

    def get_encoding(self, input, in_out):
        encoding = input if in_out == 'in' else None
        if 'posterior' in self.encoding_form and in_out == 'out':
            encoding = input
        if ('top_error' in self.encoding_form and in_out == 'in') or ('bottom_error' in self.encoding_form and in_out == 'out'):
            error = self.latent.error()
            encoding = error if encoding is None else torch.cat((encoding, error), axis=1)
        if ('top_norm_error' in self.encoding_form and in_out == 'in') or ('bottom_norm_error' in self.encoding_form and in_out == 'out'):
            norm_error = self.latent.norm_error()
            encoding = norm_error if encoding is None else torch.cat((encoding, norm_error), axis=1)
        return encoding

    def encode(self, input):
        # encode the input, possibly with errors, concatenate any deterministic units
        encoded = self.encoder(self.get_encoding(input, 'in'))
        output = self.get_encoding(self.latent.encode(encoded), 'out')
        if self.deterministic_encoder:
            det = self.deterministic_encoder(encoded)
            output = torch.cat((det, output), axis=1)
        return output

    def decode(self, input, generate=False):
        # sample the latent variables, concatenate any deterministic units, then pass through the decoder
        sample = self.latent.decode(input, generate=generate)
        if self.deterministic_decoder:
            det = self.deterministic_decoder(input)
            sample = torch.cat((sample, det), axis=1)
        return self.decoder(sample)

    def reset(self):
        self.latent.reset()

    def cuda(self, device_id=0):
        # place all modules on the GPU
        self.encoder.cuda(device_id)
        self.decoder.cuda(device_id)
        self.latent.cuda(device_id)
        self.deterministic_encoder.cuda(device_id)
        self.deterministic_decoder.cuda(device_id)



