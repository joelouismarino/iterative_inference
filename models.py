import torch
import torch.utils.data
from torch.autograd import Variable
import numpy as np

from util.logs import load_model_checkpoint
from util.distributions import DiagonalGaussian, Bernoulli
from util.modules import Dense, MultiLayerPerceptron, GaussianVariable, LatentLevel

# todo: add support for multiple samples to encode, decode
# todo: allow for printing out the model architecture
# todo: implement cpu() method


def get_model(train_config, arch, data_size):
    if train_config['resume_experiment'] != '' and train_config['resume_experiment'] is not None:
        return load_model_checkpoint()
    else:
        return LatentVariableModel(train_config, arch, data_size)


class LatentVariableModel(object):

    def __init__(self, train_config, arch, data_size):

        self.encoding_form = arch['encoding_form']
        self.constant_variances = arch['constant_prior_variances']
        self.batch_size = train_config['batch_size']
        self.kl_min = train_config['kl_min']

        self.concat_variables = arch['concat_variables']

        self.top_size = arch['top_size']
        arch['n_units_dec'].append(arch['top_size'])
        arch['n_layers_dec'].append(0)

        self.input_size = np.prod(data_size).astype(int)

        assert train_config['output_distribution'] in ['bernoulli', 'gaussian'], 'Output distribution not recognized.'
        self.output_distribution = train_config['output_distribution']

        self.levels = [None for _ in range(len(arch['n_latent']))]
        self.construct_model(arch)

        self.output_dist = None
        if self.output_distribution == 'bernoulli':
            self.output_dist = Bernoulli(None)
            self.mean_output = Dense(self.levels[0].decoder.n_out, self.input_size, non_linearity='sigmoid', weight_norm=arch['weight_norm_dec'])
        if self.output_distribution == 'gaussian':
            self.output_dist = DiagonalGaussian(None, None)
            self.mean_output = Dense(self.levels[0].decoder.n_out, self.input_size, weight_norm=arch['weight_norm_dec'])
            if self.constant_variances:
                self.trainable_log_var = Variable(torch.zeros(self.input_size), requires_grad=True)
                self.log_var_output = self.trainable_log_var.unsqueeze(0).repeat(self.batch_size, 1)
            else:
                self.log_var_output = Dense(self.levels[0].decoder.n_out, self.input_size, weight_norm=arch['weight_norm_dec'])

        self._cuda_device = None
        if train_config['cuda_device'] is not None:
            self.cuda(train_config['cuda_device'])

    def construct_model(self, arch):
        # construct the model from the architecture dictionary

        encoding_form = arch['encoding_form']
        variable_update_form = arch['variable_update_form']
        const_prior_var = arch['constant_prior_variances']
        learn_top_prior = arch['learn_top_prior']

        encoder_arch = {}
        encoder_arch['non_linearity'] = arch['non_linearity_enc']
        encoder_arch['connection_type'] = arch['connection_type_enc']
        encoder_arch['batch_norm'] = arch['batch_norm_enc']
        encoder_arch['weight_norm'] = arch['weight_norm_enc']
        encoder_arch['dropout'] = arch['dropout_enc']

        decoder_arch = {}
        decoder_arch['non_linearity'] = arch['non_linearity_dec']
        decoder_arch['connection_type'] = arch['connection_type_dec']
        decoder_arch['batch_norm'] = arch['batch_norm_dec']
        decoder_arch['weight_norm'] = arch['weight_norm_dec']
        encoder_arch['dropout'] = arch['dropout_dec']

        for level in range(len(arch['n_latent'])):

            encoder_arch['n_in'] = self.get_input_encoding_size(level, arch)
            if arch['concat_variables']:
                for lower_level in range(level):
                    encoder_arch['n_in'] += get_input_encoding_size(lower_level, arch)
            encoder_arch['n_units'] = arch['n_units_enc'][level]
            encoder_arch['n_layers'] = arch['n_layers_enc'][level]

            decoder_arch['n_in'] = arch['n_latent'][level] + arch['n_det_dec'][level]
            if arch['concat_variables']:
                for higher_level in range(level+1, len(arch['n_latent'])):
                    decoder_arch['n_in'] += arch['n_latent'][higher_level]
            decoder_arch['n_units'] = arch['n_units_dec'][level]
            decoder_arch['n_layers'] = arch['n_layers_dec'][level]

            n_latent = arch['n_latent'][level]
            n_det = [arch['n_det_enc'][level], arch['n_det_dec'][level]]

            if level == len(arch['n_latent']) - 1:
                prior_input_size = self.top_size
            else:
                prior_input_size = self.get_mlp_output_size(arch['n_latent'][level+1] + arch['n_det_dec'][level+1],
                                                            arch['n_layers_dec'][level+1], arch['n_units_dec'][level+1],
                                                            decoder_arch['connection_type'])

            posterior_input_size = self.get_mlp_output_size(encoder_arch['n_in'], encoder_arch['n_layers'],
                                                            encoder_arch['n_units'], encoder_arch['connection_type'])

            variable_input_sizes = [posterior_input_size, prior_input_size]

            learn_prior = True
            if level == len(arch['n_latent'])-1:
                learn_prior = learn_top_prior

            latent_level = LatentLevel(self.batch_size, encoder_arch, decoder_arch, n_latent, n_det,
                                       encoding_form, const_prior_var, variable_input_sizes, variable_update_form, learn_prior)

            self.levels[level] = latent_level

    def get_mlp_output_size(self, n_in, n_layers, n_units, connection_type):

        if connection_type in ['sequential', 'residual', 'highway']:
            return n_units
        elif connection_type == 'concat_input':
            return n_units + n_in
        else:
            output_size = n_in
            for _ in range(n_layers):
                output_size += n_units
            return output_size

    def get_input_encoding_size(self, level_num, arch):
        if level_num == 0:
            latent_size = self.input_size
            det_size = 0
        else:
            latent_size = arch['n_latent'][level_num-1]
            det_size = arch['n_det_enc'][level_num-1]
        encoding_size = det_size

        if 'posterior' in self.encoding_form:
            encoding_size += latent_size
        if 'bottom_error' in self.encoding_form:
            encoding_size += latent_size
        if 'bottom_norm_error' in self.encoding_form:
            encoding_size += latent_size
        if 'top_error' in self.encoding_form:
            encoding_size += arch['n_latent'][level_num]
        if 'top_norm_error' in self.encoding_form:
            encoding_size += arch['n_latent'][level_num]

        return encoding_size

    def get_input_encoding(self, input):
        if 'bottom_error' in self.encoding_form or 'bottom_norm_error' in self.encoding_form:
            assert self.output_dist is not None, 'Cannot encode error. Output distribution is None.'
        encoding = None
        if 'posterior' in self.encoding_form:
            encoding = input - 0.5
        if 'bottom_error' in self.encoding_form:
            error = input - self.output_dist.mean.detach()
            encoding = torch.cat((encoding, error)) if encoding else error
        if 'bottom_norm_error' in self.encoding_form:
            error = input - self.output_dist.mean.detach()
            if self.output_distribution == 'gaussian':
                norm_error = error / torch.exp(self.output_dist.log_var.detach())
            elif self.output_distribution == 'bernoulli':
                pass
            encoding = torch.cat((encoding, norm_error)) if encoding else norm_error
        return encoding

    def encode(self, input):
        # encode the input into a posterior estimate
        if self._cuda_device is not None:
            input = input.cuda(self._cuda_device)
        h = self.get_input_encoding(input.view(-1, self.input_size))
        for latent_level in self.levels:
            if self.concat_variables:
                h = torch.cat([h, latent_level.encode(h)], dim=1)
            else:
                h = latent_level.encode(h)

    def decode(self, generate=False):
        # decode the posterior or prior estimate
        h = Variable(torch.zeros(self.batch_size, self.top_size))
        if self._cuda_device is not None:
            h = h.cuda(self._cuda_device)
        for latent_level in self.levels[::-1]:
            if self.concat_variables:
                h = torch.cat([h, latent_level.decode(h, generate)], dim=1)
            else:
                h = latent_level.decode(h, generate)

        output_mean = self.mean_output(h)
        self.output_dist.mean = output_mean

        if self.output_distribution == 'gaussian':
            if self.constant_variances:
                output_log_var = self.log_var_output
            else:
                output_log_var = self.log_var_output(h)
            self.output_dist.log_var = output_log_var
        return self.output_dist

    def kl_divergences(self, averaged=False):
        # returns a list containing kl divergences at each level
        kl = []
        for latent_level in self.levels:
            kl.append(torch.clamp(latent_level.kl_divergence(), min=self.kl_min).sum(1))
        if averaged:
            [level_kl.mean(0) for level_kl in kl]
        else:
            return kl

    def conditional_log_likelihoods(self, input, averaged=False):
        # returns the conditional likelihoods
        if self._cuda_device is not None:
            input = input.cuda(self._cuda_device)
        input = input.view(self.batch_size, self.input_size)
        if averaged:
            return self.output_dist.log_prob(sample=input).sum(1).mean(0)
        else:
            return self.output_dist.log_prob(sample=input).sum(1)

    def elbo(self, input, averaged=False):
        # returns the ELBO
        cond_like = self.conditional_log_likelihoods(input)
        kl = sum(self.kl_divergences())
        lower_bound = cond_like - kl
        if averaged:
            return lower_bound.mean(0)
        else:
            return lower_bound

    def losses(self, input, averaged=False):
        # returns all losses
        cond_log_like = self.conditional_log_likelihoods(input)
        kl = self.kl_divergences()
        lower_bound = cond_log_like - sum(kl)
        if averaged:
            return lower_bound.mean(0), cond_log_like.mean(0), [level_kl.mean(0) for level_kl in kl]
        else:
            return lower_bound, cond_log_like, kl

    def reset(self):
        # reset the posterior estimate
        for latent_level in self.levels:
            latent_level.reset()

    def trainable_state(self):
        # make the posterior estimate trainable
        for latent_level in self.levels:
            latent_level.trainable_state()

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
        params.extend(list(self.mean_output.parameters()))
        if self.output_distribution == 'gaussian':
            if self.constant_variances:
                params.append(self.trainable_log_var)
            else:
                params.extend(list(self.log_var_output.parameters()))
        return params

    def state_parameters(self):
        # return a list containing all of the parameters of the posterior (state)
        states = []
        for latent_level in self.levels:
            states.extend(list(latent_level.state_parameters()))
        return states

    def eval(self):
        for latent_level in self.levels:
            latent_level.eval()
        self.mean_output.eval()
        if self.output_distribution == 'gaussian':
            if self.constant_variances:
                self.trainable_log_var.eval()
            else:
                self.log_var_output.eval()

    def train(self):
        for latent_level in self.levels:
            latent_level.train()
        self.mean_output.train()
        if self.output_distribution == 'gaussian':
            if self.constant_variances:
                self.trainable_log_var.train()
            else:
                self.log_var_output.train()

    def cuda(self, device_id=0):
        # place the model on the GPU
        self._cuda_device = device_id
        for latent_level in self.levels:
            latent_level.cuda(device_id)
        self.mean_output.cuda(device_id)
        if self.output_distribution == 'gaussian':
            if self.constant_variances:
                self.trainable_log_var = self.trainable_log_var.cuda(device_id)
                self.log_var_output = self.trainable_log_var.unsqueeze(0).repeat(self.batch_size, 1)
            else:
                self.log_var_output.cuda(device_id)

    def cpu(self):
        # place the model on the CPU
        pass