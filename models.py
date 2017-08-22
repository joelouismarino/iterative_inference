import torch
import torch.utils.data
from torch.autograd import Variable
import numpy as np

from util.logs import load_model_checkpoint
from util.distributions import DiagonalGaussian, Bernoulli
from util.modules import Dense, MultiLayerPerceptron, DenseGaussianVariable, DenseLatentLevel

# todo: add support for multiple samples to encode, decode
# todo: allow for printing out the model architecture
# todo: implement random re-initialization


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
        self.input_size = np.prod(data_size).astype(int)
        assert train_config['output_distribution'] in ['bernoulli', 'gaussian'], 'Output distribution not recognized.'
        self.output_distribution = train_config['output_distribution']

        # construct the model
        self.levels = [None for _ in range(len(arch['n_latent']))]
        self.output_decoder = self.output_dist = self.mean_output = self.log_var_output = self.trainable_log_var = None
        self.__construct__(arch)

        self._cuda_device = None
        if train_config['cuda_device'] is not None:
            self.cuda(train_config['cuda_device'])

    def __construct__(self, arch):
        """Construct the model from the architecture dictionary."""

        # these are the same across all latent levels
        encoding_form = arch['encoding_form']
        variable_update_form = arch['variable_update_form']
        const_prior_var = arch['constant_prior_variances']

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
        decoder_arch['dropout'] = arch['dropout_dec']

        # construct a DenseLatentLevel for each level of latent variables
        for level in range(len(arch['n_latent'])):

            # get specifications for this level's encoder and decoder
            encoder_arch['n_in'] = self.encoder_input_size(level, arch)
            encoder_arch['n_units'] = arch['n_units_enc'][level]
            encoder_arch['n_layers'] = arch['n_layers_enc'][level]

            decoder_arch['n_in'] = self.decoder_input_size(level, arch)
            decoder_arch['n_units'] = arch['n_units_dec'][level+1]
            decoder_arch['n_layers'] = arch['n_layers_dec'][level+1]

            n_latent = arch['n_latent'][level]
            n_det = [arch['n_det_enc'][level], arch['n_det_dec'][level]]

            learn_prior = True if arch['learn_top_prior'] else (level != len(arch['n_latent'])-1)

            self.levels[level] = DenseLatentLevel(self.batch_size, encoder_arch, decoder_arch, n_latent, n_det,
                                                  encoding_form, const_prior_var, variable_update_form, learn_prior)

        # construct the output decoder
        decoder_arch['n_in'] = self.decoder_input_size(-1, arch)
        decoder_arch['n_units'] = arch['n_units_dec'][0]
        decoder_arch['n_layers'] = arch['n_layers_dec'][0]
        self.output_decoder = MultiLayerPerceptron(**decoder_arch)

        # construct the output distribution
        if self.output_distribution == 'bernoulli':
            self.output_dist = Bernoulli(None)
            self.mean_output = Dense(arch['n_units_dec'][0], self.input_size, non_linearity='sigmoid', weight_norm=arch['weight_norm_dec'])
        elif self.output_distribution == 'gaussian':
            self.output_dist = DiagonalGaussian(None, None)
            self.mean_output = Dense(arch['n_units_dec'][0], self.input_size, non_linearity='sigmoid', weight_norm=arch['weight_norm_dec'])
            if self.constant_variances:
                self.trainable_log_var = Variable(torch.zeros(self.input_size), requires_grad=True)
            else:
                self.log_var_output = Dense(arch['n_units_dec'][0], self.input_size, weight_norm=arch['weight_norm_dec'])

    def encoder_input_size(self, level_num, arch):
        """Calculates the size of the encoding input to a level."""

        def _encoding_size(_self, _level_num, _arch, lower_level=False):

            if _level_num == 0:
                latent_size = _self.input_size
                det_size = 0
            else:
                latent_size = _arch['n_latent'][_level_num-1]
                det_size = _arch['n_det_enc'][_level_num-1]
            encoding_size = det_size

            if 'posterior' in _self.encoding_form:
                encoding_size += latent_size
            if 'bottom_error' in _self.encoding_form:
                encoding_size += latent_size
            if 'bottom_norm_error' in _self.encoding_form:
                encoding_size += latent_size
            if 'top_error' in _self.encoding_form and not lower_level:
                encoding_size += _arch['n_latent'][_level_num]
            if 'top_norm_error' in _self.encoding_form and not lower_level:
                encoding_size += _arch['n_latent'][_level_num]

            return encoding_size

        encoder_size = _encoding_size(self, level_num, arch)
        if self.concat_variables:
            for level in range(level_num):
                encoder_size += _encoding_size(self, level, arch, lower_level=True)
        return encoder_size

    def decoder_input_size(self, level_num, arch):
        """Calculates the size of the decoding input to a level."""
        if level_num == len(arch['n_latent'])-1:
            return self.top_size

        decoder_size = arch['n_latent'][level_num+1] + arch['n_det_dec'][level_num+1]
        if self.concat_variables:
            for level in range(level_num+2, len(arch['n_latent'])):
                decoder_size += (arch['n_latent'][level] + arch['n_det_dec'][level])
        return decoder_size

    def get_input_encoding(self, input):
        """Gets the encoding at the bottom level."""
        if 'bottom_error' in self.encoding_form or 'bottom_norm_error' in self.encoding_form:
            assert self.output_dist is not None, 'Cannot encode error. Output distribution is None.'
        encoding = None
        if 'posterior' in self.encoding_form:
            encoding = input - 0.5
        if 'bottom_error' in self.encoding_form:
            error = input - self.output_dist.mean.detach()
            encoding = error if encoding is None else torch.cat((encoding, error), dim=1)
        if 'bottom_norm_error' in self.encoding_form:
            error = input - self.output_dist.mean.detach()
            norm_error = None
            if self.output_distribution == 'gaussian':
                norm_error = error / torch.exp(self.output_dist.log_var.detach())
            elif self.output_distribution == 'bernoulli':
                mean = self.output_dist.mean.detach()
                norm_error = error * torch.exp(- torch.log(mean + 1e-5) - torch.log(1 - mean + 1e-5))
            encoding = norm_error if encoding is None else torch.cat((encoding, norm_error), dim=1)
        return encoding

    def encode(self, input):
        """Encodes the input into an updated posterior estimate."""
        if self._cuda_device is not None:
            input = input.cuda(self._cuda_device)
        h = self.get_input_encoding(input.view(-1, self.input_size))
        for latent_level in self.levels:
            if self.concat_variables:
                h = torch.cat([h, latent_level.encode(h)], dim=1)
            else:
                h = latent_level.encode(h)

    def decode(self, generate=False):
        """Decodes the posterior (prior) estimate to get a reconstruction (sample)."""
        h = Variable(torch.zeros(self.batch_size, self.top_size))
        if self._cuda_device is not None:
            h = h.cuda(self._cuda_device)
        concat = False
        for latent_level in self.levels[::-1]:
            if self.concat_variables and concat:
                h = torch.cat([h, latent_level.decode(h, generate)], dim=1)
            else:
                h = latent_level.decode(h, generate)
            concat = True

        h = self.output_decoder(h)
        self.output_dist.mean = self.mean_output(h)

        if self.output_distribution == 'gaussian':
            if self.constant_variances:
                self.output_dist.log_var = self.trainable_log_var.unsqueeze(0).repeat(self.batch_size, 1)
            else:
                self.output_dist.log_var = self.log_var_output(h)
        return self.output_dist

    def kl_divergences(self, averaged=False):
        """Returns a list containing kl divergences at each level."""
        kl = []
        for latent_level in self.levels:
            kl.append(torch.clamp(latent_level.kl_divergence(), min=self.kl_min).sum(1))
        if averaged:
            [level_kl.mean(0) for level_kl in kl]
        else:
            return kl

    def conditional_log_likelihoods(self, input, averaged=False):
        """Returns the conditional likelihood."""
        if self._cuda_device is not None:
            input = input.cuda(self._cuda_device)
        input = input.view(-1, self.input_size)
        if averaged:
            return self.output_dist.log_prob(sample=input).sum(1).mean(0)
        else:
            return self.output_dist.log_prob(sample=input).sum(1)

    def elbo(self, input, averaged=False):
        """Returns the ELBO."""
        cond_like = self.conditional_log_likelihoods(input)
        kl = sum(self.kl_divergences())
        lower_bound = cond_like - kl
        if averaged:
            return lower_bound.mean(0)
        else:
            return lower_bound

    def losses(self, input, averaged=False):
        """Returns all losses."""
        cond_log_like = self.conditional_log_likelihoods(input)
        kl = self.kl_divergences()
        lower_bound = cond_log_like - sum(kl)
        if averaged:
            return lower_bound.mean(0), cond_log_like.mean(0), [level_kl.mean(0) for level_kl in kl]
        else:
            return lower_bound, cond_log_like, kl

    def reset_state(self):
        """Resets the posterior estimate."""
        for latent_level in self.levels:
            latent_level.reset()

    def trainable_state(self):
        """Makes the posterior estimate trainable."""
        for latent_level in self.levels:
            latent_level.trainable_state()

    def parameters(self):
        """Returns a list containing all parameters."""
        return self.encoder_parameters() + self.decoder_parameters()

    def encoder_parameters(self):
        """Returns a list containing all parameters in the encoder."""
        params = []
        for level in self.levels:
            params.extend(level.encoder_parameters())
        return params

    def decoder_parameters(self):
        """Returns a list containing all parameters in the decoder."""
        params = []
        for level in self.levels:
            params.extend(level.decoder_parameters())
        params.extend(list(self.output_decoder.parameters()))
        params.extend(list(self.mean_output.parameters()))
        if self.output_distribution == 'gaussian':
            if self.constant_variances:
                params.append(self.trainable_log_var)
            else:
                params.extend(list(self.log_var_output.parameters()))
        return params

    def state_parameters(self):
        """Returns a list containing the posterior estimate (state)."""
        states = []
        for latent_level in self.levels:
            states.extend(list(latent_level.state_parameters()))
        return states

    def eval(self):
        """Puts the model into eval mode (affects batch_norm and dropout)."""
        for latent_level in self.levels:
            latent_level.eval()
        self.output_decoder.eval()
        self.mean_output.eval()
        if self.output_distribution == 'gaussian':
            if not self.constant_variances:
                self.log_var_output.eval()

    def train(self):
        """Puts the model into train mode (affects batch_norm and dropout)."""
        for latent_level in self.levels:
            latent_level.train()
        self.output_decoder.train()
        self.mean_output.train()
        if self.output_distribution == 'gaussian':
            if not self.constant_variances:
                self.log_var_output.train()

    def random_re_init(self, re_init_fraction=0.05):
        """Randomly re-initializes a fraction of all of the weights in the model."""
        for level in self.levels:
            level.random_re_init(re_init_fraction)
        self.output_decoder.random_re_init(re_init_fraction)
        self.mean_output.random_re_init(re_init_fraction)
        if output_distribution == 'gaussian':
            if not self.constant_variances:
                self.log_var_output.random_re_init(re_init_fraction)

    def cuda(self, device_id=0):
        """Places the model on the GPU."""
        self._cuda_device = device_id
        for latent_level in self.levels:
            latent_level.cuda(device_id)
        self.output_decoder = self.output_decoder.cuda(device_id)
        self.mean_output = self.mean_output.cuda(device_id)
        self.output_dist.cuda(device_id)
        if self.output_distribution == 'gaussian':
            if self.constant_variances:
                self.trainable_log_var = Variable(self.trainable_log_var.data.cuda(device_id), requires_grad=True)
                self.log_var_output = self.trainable_log_var.unsqueeze(0).repeat(self.batch_size, 1)
            else:
                self.log_var_output = self.log_var_output.cuda(device_id)

    def cpu(self):
        """Places the model on the CPU."""
        self._cuda_device = None
        for latent_level in self.levels:
            latent_level.cpu()
        self.output_decoder = self.output_decoder.cpu()
        self.mean_output = self.mean_output.cpu()
        self.output_dist.cpu()
        if self.output_distribution == 'gaussian':
            if self.constant_variances:
                self.trainable_log_var = self.trainable_log_var.cpu()
                self.log_var_output = self.trainable_log_var.unsqueeze(0).repeat(self.batch_size, 1)
            else:
                self.log_var_output = self.log_var_output.cpu()
