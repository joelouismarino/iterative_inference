import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

from util.distributions import DiagonalGaussian, Bernoulli
from util.modules import Dense, MultiLayerPerceptron, GaussianVariable, LatentLevel

# todo: add support for multiple samples to encode, decode
class LatentVariableModel(object):

    def __init__(self, train_config, arch):

        self.encoding_form = arch.encoding_form
        self.batch_size = train_config.batch_size
        self.kl_min = train_config.kl_min
        self.top_size = arch.top_size
        assert train_config.output_distribution in ['bernoulli', 'gaussian'], 'Output distribution not recognized.'
        self.output_distribution = train_config.output_distribution
        self.levels = []
        self.construct_model(arch)
        self.output_dist = None
        if self.output_distribution == 'bernoulli':
            self.mean_output = Dense(arch.n_units_dec[0], arch.n_latent[0], non_linearity='sigmoid', weight_norm=arch.weight_norm_dec)
        if self.output_distribution == 'gaussian':
            self.mean_output = Dense(arch.n_units_dec[0], arch.n_latent[0], weight_norm=arch.weight_norm_dec)
            self.log_var_output = Dense(arch.n_units_dec[0], arch.n_latent[0], weight_norm=arch.weight_norm_dec)
        self._cuda_device = None
        if train_config['cuda_device'] is not None:
            self.cuda(train_config['cuda_device'])

    def construct_model(self, arch):
        # construct the model

        encoding_form = arch.encoding_form
        variable_update_form = arch.variable_update_form
        constant_prior_variances = arch.constant_prior_variances

        encoder_arch = {}
        encoder_arch['non_linearity'] = arch.non_linearity_enc
        encoder_arch['connection_type'] = arch.connection_type_enc
        encoder_arch['batch_norm'] = arch.batch_norm_enc
        encoder_arch['weight_norm'] = arch.weight_norm_enc

        decoder_arch = {}
        decoder_arch['non_linearity'] = arch.non_linearity_dec
        decoder_arch['connection_type'] = arch.connection_type_dec
        decoder_arch['batch_norm'] = arch.batch_norm_dec
        decoder_arch['weight_norm'] = arch.weight_norm_dec

        for level in range(len(arch.n_latent)):

            encoder_arch['n_in'] = arch.n_latent[level] + arch.n_det_enc[level]
            encoder_arch['n_units'] = arch.n_units_enc[level]
            encoder_arch['n_layers'] = arch.n_layers_enc[level]

            decoder_arch['n_in'] = arch.n_latent[level] + arch.n_det_dec[level+1]
            decoder_arch['n_units'] = arch.n_units_dec[level]
            decoder_arch['n_layers'] = arch.n_layers_dec[level]

            n_latent = arch.n_latent[level]
            n_det = [arch.n_det_enc[level], arch.n_det_dec[level]]
            variable_input_sizes = [arch.num_units_enc[level], arch.num_units_dec[level]]

            latent_level = LatentLevel(self.batch_size, encoder_arch, decoder_arch, n_latent, n_det,
                                       encoding_form, constant_prior_variances, variable_input_sizes, variable_update_form)

            self.levels.append(latent_level)

    def get_input_encoding(self, input):
        if 'bottom_error' in self.encoding_form or 'bottom_norm_error' in self.encoding_form:
            assert self.output_dist is not None, 'Cannot encode error. Output distribution is None.'
        encoding = None
        if 'posterior' in self.encoding_form:
            encoding = input
        if 'bottom_error' in self.encoding_form:
            error = input - self.output_dist.mean
            encoding = torch.cat((encoding, error)) if encoding else error
        if 'bottom_norm_error' in self.encoding_form:
            error = input - self.output_dist.mean
            if self.output_distribution == 'gaussian':
                norm_error = error / torch.exp(self.output_dist.log_var)
            elif self.output_distribution == 'bernoulli':
                pass
            encoding = torch.cat((encoding, norm_error)) if encoding else norm_error
        return encoding

    def encode(self, input):
        # encode the input into a posterior estimate
        h = self.get_input_encoding(input)
        if self._cuda_device is not None:
            h = h.cuda(self._cuda_device)
        for latent_level in self.levels:
            h = latent_level.encode(h)

    def decode(self, generate=False):
        # decode the posterior or prior estimate
        h = Variable(torch.zeros(self.batch_size, self.top_size))
        if self._cuda_device is not None:
            h = h.cuda(self._cuda_device)
        for latent_level in self.levels:
            h = latent_level.decode(h, generate)

        if self.output_distribution == 'bernoulli':
            output_mean = self.mean_output(h)
            self.output_dist = Bernoulli(output_mean)
        elif self.output_distribution == 'gaussian':
            output_mean = self.mean_output(h)
            output_log_var = self.log_var_output(h)
            self.output_dist = DiagonalGaussian(output_mean, output_log_var)
        return self.output_dist

    def kl_divergences(self):
        # returns a list containing kl divergences at each level, for all examples
        kl = []
        for latent_level in self.levels:
            kl.append(torch.clamp(latent_level.kl_divergence(), min=self.kl_min).sum(1))
        return kl

    def conditional_likelihoods(self, input):
        # returns the conditional likelihoods, for all examples
        return self.output_dist.log_prob(sample=input).sum(1)

    def ELBO(self, input):
        # returns the ELBO, averaged across examples
        loss = self.conditional_likelihoods(input)
        kl = self.kl_divergences()
        for level in range(len(kl)):
            loss += kl[level]
        return loss.mean(0)

    def reset(self):
        # reset the posterior estimate
        for latent_level in self.levels:
            latent_level.reset()

    def parameters(self):
        return self.encoder_parameters() + self.decoder_parameters()

    def encoder_parameters(self):
        # return a list containing all parameters in the encoder
        params = []
        for level in self.levels:
            params.extend(level.encoder_parameters())
        return params

    def decoder_parameters(self):
        # return a list containing all parameters in the decoder
        params = []
        for level in self.levels:
            params.extend(level.decoder_parameters())
        params.extend(list(self.mean_output.parameters))
        if self.output_distribution == 'gaussian':
            params.extend(list(self.log_var_output.parameters()))
        return params

    def state_parameters(self):
        # return a list containing all of the parameters of the posterior
        states = []
        for latent_level in self.levels:
            states.extend(list(latent_level.state_parameters()))
        return states

    def cuda(self, device_id=0):
        # place the model on the GPU
        self._cuda_device = device_id
        for latent_level in self.levels:
            latent_level.cuda(device_id)
        self.mean_output.cuda(device_id)
        if self.output_distribution == 'gaussian':
            self.log_var_output.cuda(device_id)

    def save(self):
        pass