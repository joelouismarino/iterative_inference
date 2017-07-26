import torch
import torch.nn as nn
from torch.autograd import Variable

from distributions import DiagonalGaussian


class Dense(nn.Module):

    """Fully-connected (dense) layer with optional batch normalization, non-linearity, weight normalization, and dropout."""

    def __init__(self, n_in, n_out, non_linearity=None, batch_norm=False, weight_norm=False, dropout=0.):
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

        self.dropout = None
        if dropout > 0.:
            self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        output = self.linear(input)
        if self.bn:
            output = self.bn(output)
        if self.non_linearity:
            output = self.non_linearity(output)
        if self.dropout:
            output = self.dropout(output)
        return output

class Conv(nn.Module):

    """Basic convolutional layer with optional batch normalization, non-linearity, weight normalization and dropout."""

    def __init__(self, n_in, filter_size, n_out, non_linearity=None, batch_norm=False, weight_norm=False, dropout=0.):
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(n_in, n_out, filter_size, padding=int(np.ceil(filter_size/2)))
        self.bn = None
        if batch_norm:
            self.bn = nn.BatchNorm2d(n_out)
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

        if dropout > 0.:
            self.dropout = nn.Dropout2d(dropout)

    def forward(self, input):
        output = self.conv(input)
        if self.bn:
            output = self.bn(output)
        if self.non_linearity:
            output = self.non_linearity(output)
        if self.dropout:
            output = self.dropout(output)
        return output

class MultiLayerPerceptron(nn.Module):

    """Multi-layered perceptron."""

    def __init__(self, n_in, n_units, n_layers, non_linearity=None, connection_type='sequential', batch_norm=False, weight_norm=False, dropout=0.):

        super(MultiLayerPerceptron, self).__init__()
        assert connection_type in ['sequential', 'residual', 'highway', 'concat_input', 'concat'], 'Connection type not found.'
        self.connection_type = connection_type
        self.layers = nn.ModuleList([])
        self.gates = nn.ModuleList([])

        n_in_orig = n_in

        if self.connection_type in ['residual', 'highway']:
            self.initial_dense = Dense(n_in, n_units, batch_norm=batch_norm, weight_norm=weight_norm)

        for _ in range(n_layers):
            layer = Dense(n_in, n_units, non_linearity=non_linearity, batch_norm=batch_norm, weight_norm=weight_norm, dropout=dropout)
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

class MultiLayerConvolutional(nn.Module):

    """Multi-layer convolutional network."""

    def __init__(self, n_in, n_filters, filter_size, n_layers, non_linearity=None, connection_type='sequential', batch_norm=False, weight_norm=False, dropout=0.):
        super(MultiLayerConvolutional, self).__init__()
        assert connection_type in ['sequential', 'residual', 'highway', 'concat_input', 'concat'], 'Connection type not found.'
        self.connection_type = connection_type
        self.layers = nn.ModuleList([])
        self.gates = nn.ModuleList([])

        n_in_orig = n_in

        if self.connection_type in ['residual', 'highway']:
            self.initial_conv = Conv(n_in, filter_size, n_filters, batch_norm=batch_norm, weight_norm=weight_norm)

        for _ in range(n_layers):
            layer = Conv(n_in, n_units, non_linearity=non_linearity, batch_norm=batch_norm, weight_norm=weight_norm, dropout=dropout)
            self.layers.append(layer)

            if self.connection_type == 'highway':
                gate = Conv(n_in, filter_size, n_filters, non_linearity='sigmoid', batch_norm=batch_norm, weight_norm=weight_norm)
                self.gates.append(gate)

            if self.connection_type in ['sequential', 'residual', 'highway']:
                n_in = n_filters
            elif self.connection_type == 'concat_input':
                n_in = n_filters + n_in_orig
            elif self.connection_type == 'concat':
                n_in += n_filters

    def forward(self, input):

        input_orig = input.clone()

        for layer_num, layer in enumerate(self.layers):
            if self.connection_type == 'sequential':
                input = layer(input)

            elif self.connection_type == 'residual':
                if layer_num == 0:
                    input = self.initial_conv(input) + layer(input)
                else:
                    input += layer(input)

            elif self.connection_type == 'highway':
                gate = self.gates[layer_num]
                if layer_num == 0:
                    input = gate * self.initial_conv(input) + (1 - gate) * layer(input)
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
        assert update_form in ['direct', 'highway'], 'Variable update type not found.'
        self.update_form = update_form

        self.prior_mean = Dense(n_input[1], self.n_variables)
        self.prior_log_var = Dense(n_input[1], self.n_variables)
        self.posterior_mean = Dense(n_input[0], self.n_variables)
        self.posterior_log_var = Dense(n_input[0], self.n_variables)

        if self.update_form == 'highway':
            self.posterior_mean_gate = Dense(n_input[0], self.n_variables, 'sigmoid')
            self.posterior_log_var_gate = Dense(n_input[0], self.n_variables, 'sigmoid')

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
        return self.error() / (torch.exp(self.prior.log_var) + 1e-7)

    def KL_divergence(self):
        return self.posterior.log_prob(self.posterior.sample()) - self.prior.log_prob(self.posterior.sample())

    def reset(self):
        self.reset_mean()
        self.reset_log_var()

    def reset_mean(self):
        self.posterior.reset_mean()

    def reset_log_var(self):
        self.posterior.reset_log_var()

    def trainable_mean(self, trainable=True):
        self.posterior.mean_trainable(trainable)

    def trainable_log_var(self, trainable=True):
        self.posterior.log_var_trainable(trainable)

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

    def parameters(self):
        return self.encoder_parameters() + self.decoder_parameters()

    def encoder_parameters(self):
        encoder_params = []
        encoder_params.extend(list(self.posterior_mean.parameters()))
        encoder_params.extend(list(self.posterior_log_var.parameters()))
        if self.update_form == 'highway':
            encoder_params.extend(list(self.posterior_mean_gate.parameters()))
            encoder_params.extend(list(self.posterior_log_var_gate.parameters()))
        return encoder_params

    def decoder_parameters(self):
        decoder_params = []
        decoder_params.extend(list(self.prior_mean.parameters()))
        decoder_params.extend(list(self.prior_log_var.parameters()))
        return decoder_params

    def state_parameters(self):
        return self.posterior.state_parameters()

"""
class ConvGaussianVariable(object):

    def __init__(self, batch_size, n_variable_channels, filter_size, n_input, update_form):

        self.batch_size = batch_size
        self.n_variable_channels = n_variable_channels
        assert update_form in ['direct', 'highway'], 'Variable update type not found.'
        self.update_form = update_form

        self.prior_mean = Conv(n_input[1], filter_size, self.n_variable_channels)
        self.prior_log_var = Dense(n_input[1], filter_size, self.n_variable_channels)
        self.posterior_mean = Dense(n_input[0], filter_size, self.n_variable_channels)
        self.posterior_log_var = Dense(n_input[0], filter_size, self.n_variable_channels)

        if self.update_form == 'highway':
            self.posterior_mean_gate = Conv(n_input[0], filter_size, self.n_variable_channels, 'sigmoid')
            self.posterior_log_var_gate = Dense(n_input[0], filter_size, self.n_variable_channels, 'sigmoid')

        self.prior = self.init_dist()
        self.posterior = self.init_dist()

    def init_dist(self):
        return DiagonalGaussian(Variable(torch.zeros(self.batch_size, self.n_variable_channels, , )),
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
        return self.error() / (torch.exp(self.prior.log_var) + 1e-7)

    def KL_divergence(self):
        return self.posterior.log_prob(self.posterior.sample()) - self.prior.log_prob(self.posterior.sample())

    def reset(self):
        self.reset_mean()
        self.reset_log_var()

    def reset_mean(self):
        self.posterior.reset_mean()

    def reset_log_var(self):
        self.posterior.reset_log_var()

    def trainable_mean(self, trainable=True):
        self.posterior.mean_trainable(trainable)

    def trainable_log_var(self, trainable=True):
        self.posterior.log_var_trainable(trainable)

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

    def parameters(self):
        pass

    def encoder_parameters(self):
        pass

    def decoder_parameters(self):
        pass

    def state_parameters(self):
        return self.posterior.state_parameters()


"""

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
            output = torch.cat((det, output), 1)
        return output

    def decode(self, input, generate=False):
        # sample the latent variables, concatenate any deterministic units, then pass through the decoder
        sample = self.latent.decode(input, generate=generate)
        if self.deterministic_decoder:
            det = self.deterministic_decoder(input)
            sample = torch.cat((sample, det), 1)
        return self.decoder(sample)

    def KL_divergence(self):
        return self.latent.KL_divergence()

    def reset(self):
        self.latent.reset()

    def cuda(self, device_id=0):
        # place all modules on the GPU
        self.encoder.cuda(device_id)
        self.decoder.cuda(device_id)
        self.latent.cuda(device_id)
        self.deterministic_encoder.cuda(device_id)
        self.deterministic_decoder.cuda(device_id)

    def parameters(self):
        return self.encoder_parameters() + self.decoder_parameters()

    def encoder_parameters(self):
        encoder_params = []
        encoder_params.extend(list(self.encoder.parameters()))
        encoder_params.extend(list(self.deterministic_encoder.parameters()))
        encoder_params.extend(list(self.latent.encoder_parameters()))
        return encoder_params

    def decoder_parameters(self):
        decoder_params = []
        decoder_params.extend(list(self.decoder.parameters()))
        decoder_params.extend(list(self.deterministic_decoder.parameters()))
        decoder_params.extend(list(self.latent.decoder_parameters()))
        return decoder_params

    def state_parameters(self):
        return self.latent.state_parameters()