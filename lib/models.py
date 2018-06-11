import torch
from torch.autograd import Variable
import torch.optim as opt
import numpy as np

from util.logs import load_model_checkpoint
from distributions import DiagonalGaussian, Bernoulli
from modules import Dense, MultiLayerPerceptron, DenseGaussianVariable, DenseLatentLevel, RecurrentLatentLevel


def get_model(train_config, arch, data_loader):
    if train_config['resume_experiment'] != '' and train_config['resume_experiment'] is not None:
        model = load_model_checkpoint(cuda_device=train_config['cuda_device'])
        model.cuda(train_config['cuda_device'])
        return model
    elif arch['model_form'] == 'dense':
        return DenseLatentVariableModel(train_config, arch, data_loader)
    elif arch['model_form'] == 'conv':
        return ConvLatentVariableModel(train_config, arch, data_loader)


class DenseLatentVariableModel(object):

    def __init__(self, train_config, arch, data_loader):

        self.encoding_form = arch['encoding_form']
        self.constant_variances = arch['constant_prior_variances']
        self.single_output_variance = arch['single_output_variance']
        self.posterior_form = arch['posterior_form']
        self.batch_size = train_config['batch_size']
        self.n_training_samples = train_config['n_samples']
        self.kl_min = train_config['kl_min']
        self.concat_variables = arch['concat_variables']
        self.top_size = arch['top_size']
        self.input_size = np.prod(tuple(next(iter(data_loader))[0].size()[1:])).astype(int)
        assert train_config['output_distribution'] in ['bernoulli', 'gaussian'], 'Output distribution not recognized.'
        self.output_distribution = train_config['output_distribution']
        self.reconstruction = None
        self.kl_weight = 1.

        # construct the model
        self.levels = [None for _ in range(len(arch['n_latent']))]
        self.output_decoder = self.output_dist = self.mean_output = self.log_var_output = self.trainable_log_var = None
        self.state_optimizer = None
        self.__construct__(arch)

        self.whitening_matrix = self.inverse_whitening_matrix = self.data_mean = None
        if arch['whiten_input']:
            self.whitening_matrix, self.inverse_whitening_matrix, self.data_mean = self.calculate_whitening_matrix(data_loader)

        self._cuda_device = None
        if train_config['cuda_device'] is not None:
            self.cuda(train_config['cuda_device'])

    def __construct__(self, arch):
        """
        Construct the model from the architecture dictionary.
        :param arch: architecture dictionary
        :return None
        """

        # these are the same across all latent levels
        encoding_form = arch['encoding_form']
        variable_update_form = arch['variable_update_form']
        const_prior_var = arch['constant_prior_variances']
        posterior_form = arch['posterior_form']

        latent_level_type = RecurrentLatentLevel if arch['encoder_type'] == 'recurrent' else DenseLatentLevel

        encoder_arch = None
        if arch['encoder_type'] == 'inference_model':
            encoder_arch = dict()
            encoder_arch['non_linearity'] = arch['non_linearity_enc']
            encoder_arch['connection_type'] = arch['connection_type_enc']
            encoder_arch['batch_norm'] = arch['batch_norm_enc']
            encoder_arch['weight_norm'] = arch['weight_norm_enc']
            encoder_arch['dropout'] = arch['dropout_enc']

        decoder_arch = dict()
        decoder_arch['non_linearity'] = arch['non_linearity_dec']
        decoder_arch['connection_type'] = arch['connection_type_dec']
        decoder_arch['batch_norm'] = arch['batch_norm_dec']
        decoder_arch['weight_norm'] = arch['weight_norm_dec']
        decoder_arch['dropout'] = arch['dropout_dec']

        # construct a DenseLatentLevel for each level of latent variables
        for level in range(len(arch['n_latent'])):
            # get specifications for this level's encoder and decoder

            if arch['encoder_type'] == 'inference_model':
                encoder_arch['n_in'] = self.encoder_input_size(level, arch)
                encoder_arch['n_units'] = arch['n_units_enc'][level]
                encoder_arch['n_layers'] = arch['n_layers_enc'][level]

            decoder_arch['n_in'] = self.decoder_input_size(level, arch)
            decoder_arch['n_units'] = arch['n_units_dec'][level+1]
            decoder_arch['n_layers'] = arch['n_layers_dec'][level+1]

            n_latent = arch['n_latent'][level]
            n_det = [arch['n_det_enc'][level], arch['n_det_dec'][level]]

            learn_prior = True if arch['learn_top_prior'] else (level != len(arch['n_latent'])-1)

            self.levels[level] = latent_level_type(self.batch_size, encoder_arch, decoder_arch, n_latent, n_det,
                                                  encoding_form, const_prior_var, variable_update_form, posterior_form, learn_prior)

        # construct the output decoder
        decoder_arch['n_in'] = self.decoder_input_size(-1, arch)
        decoder_arch['n_units'] = arch['n_units_dec'][0]
        decoder_arch['n_layers'] = arch['n_layers_dec'][0]
        self.output_decoder = MultiLayerPerceptron(**decoder_arch)

        # construct the output distribution
        if self.output_distribution == 'bernoulli':
            self.output_dist = Bernoulli(self.input_size, None)
            self.mean_output = Dense(arch['n_units_dec'][0], self.input_size, non_linearity='sigmoid', weight_norm=arch['weight_norm_dec'])
        elif self.output_distribution == 'gaussian':
            self.output_dist = DiagonalGaussian(self.input_size, None, None)
            non_lin = 'linear' if arch['whiten_input'] else 'sigmoid'
            self.mean_output = Dense(arch['n_units_dec'][0], self.input_size, non_linearity=non_lin, weight_norm=arch['weight_norm_dec'])
            if self.constant_variances:
                if arch['single_output_variance']:
                    self.trainable_log_var = Variable(torch.zeros(1), requires_grad=True)
                else:
                    self.trainable_log_var = Variable(torch.normal(torch.zeros(self.input_size), 0.25), requires_grad=True)
            else:
                self.log_var_output = Dense(arch['n_units_dec'][0], self.input_size, weight_norm=arch['weight_norm_dec'])

        # make the state trainable if encoder_type is EM
        if arch['encoder_type'] in ['em', 'EM']:
            self.trainable_state()

    def encoder_input_size(self, level_num, arch):
        """
        Calculates the size of the encoding input to a level.
        If we're encoding the gradient, then the encoding size
        is the size of the latent variables (x 2 if Gaussian variable).
        Otherwise, the encoding size depends on how many errors/variables
        we're encoding.
        :param level_num: the index of the level we're calculating the
                          encoding size for
        :param arch: architecture dictionary
        :return: the size of this level's encoder's input
        """

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
            if 'mean' in _self.encoding_form and not lower_level:
                encoding_size += _arch['n_latent'][_level_num]
            if 'l2_norm_mean' in _self.encoding_form and not lower_level:
                encoding_size += _arch['n_latent'][_level_num]
            if 'layer_norm_mean' in _self.encoding_form and not lower_level:
                encoding_size += _arch['n_latent'][_level_num]
            if 'mean_gradient' in _self.encoding_form and not lower_level:
                encoding_size += _arch['n_latent'][_level_num]
            if 'l2_norm_mean_gradient' in _self.encoding_form and not lower_level:
                encoding_size += _arch['n_latent'][_level_num]
            if 'layer_norm_mean_gradient' in _self.encoding_form and not lower_level:
                encoding_size += _arch['n_latent'][_level_num]
            if 'log_var_gradient' in _self.encoding_form and not lower_level:
                encoding_size += _arch['n_latent'][_level_num]
            if 'l2_norm_log_var_gradient' in _self.encoding_form and not lower_level:
                encoding_size += _arch['n_latent'][_level_num]
            if 'layer_norm_log_var_gradient' in _self.encoding_form and not lower_level:
                encoding_size += _arch['n_latent'][_level_num]
            if 'log_var' in _self.encoding_form and not lower_level:
                encoding_size += _arch['n_latent'][_level_num]
            if 'l2_norm_log_var' in _self.encoding_form and not lower_level:
                encoding_size += _arch['n_latent'][_level_num]
            if 'layer_norm_log_var' in _self.encoding_form and not lower_level:
                encoding_size += _arch['n_latent'][_level_num]
            if 'var' in _self.encoding_form and not lower_level:
                encoding_size += _arch['n_latent'][_level_num]
            if 'bottom_error' in _self.encoding_form:
                encoding_size += latent_size
            if 'l2_norm_bottom_error' in _self.encoding_form:
                encoding_size += latent_size
            if 'layer_norm_bottom_error' in _self.encoding_form:
                encoding_size += latent_size
            if 'bottom_norm_error' in _self.encoding_form:
                encoding_size += latent_size
            if 'l2_norm_bottom_norm_error' in _self.encoding_form:
                encoding_size += latent_size
            if 'layer_norm_bottom_norm_error' in _self.encoding_form:
                encoding_size += latent_size
            if 'top_error' in _self.encoding_form and not lower_level:
                encoding_size += _arch['n_latent'][_level_num]
            if 'l2_norm_top_error' in _self.encoding_form and not lower_level:
                encoding_size += _arch['n_latent'][_level_num]
            if 'layer_norm_top_error' in _self.encoding_form and not lower_level:
                encoding_size += _arch['n_latent'][_level_num]
            if 'top_norm_error' in _self.encoding_form and not lower_level:
                encoding_size += _arch['n_latent'][_level_num]
            if 'l2_norm_top_norm_error' in _self.encoding_form and not lower_level:
                encoding_size += _arch['n_latent'][_level_num]
            if 'layer_norm_top_norm_error' in _self.encoding_form and not lower_level:
                encoding_size += _arch['n_latent'][_level_num]
            if 'gradient' in _self.encoding_form and not lower_level:
                encoding_size += _arch['n_latent'][_level_num]
                if self.posterior_form == 'gaussian':
                    encoding_size += _arch['n_latent'][_level_num]
            if 'l2_norm_gradient' in _self.encoding_form and not lower_level:
                encoding_size += _arch['n_latent'][_level_num]
                if self.posterior_form == 'gaussian':
                    encoding_size += _arch['n_latent'][_level_num]
            if 'log_gradient' in _self.encoding_form and not lower_level:
                encoding_size += _arch['n_latent'][_level_num]
                if self.posterior_form == 'gaussian':
                    encoding_size += _arch['n_latent'][_level_num]
            if 'scaled_log_gradient' in _self.encoding_form and not lower_level:
                encoding_size += _arch['n_latent'][_level_num]
                if self.posterior_form == 'gaussian':
                    encoding_size += _arch['n_latent'][_level_num]
            if 'sign_gradient' in _self.encoding_form and not lower_level:
                encoding_size += _arch['n_latent'][_level_num]
                if self.posterior_form == 'gaussian':
                    encoding_size += _arch['n_latent'][_level_num]

            return encoding_size

        encoder_size = _encoding_size(self, level_num, arch)
        if 'gradient' not in self.encoding_form:
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

    def calculate_whitening_matrix(self, data_loader, n_examples=50000):
        """Calculates and returns the whitening matrix for the training data."""
        print 'Calculating whitening matrix...'
        n = 0
        batch_size = next(iter(data_loader))[0].size()[0]
        train_data = torch.zeros(batch_size * int(n_examples / batch_size), self.input_size)
        for batch, _ in data_loader:
            if n >= train_data.shape[0]:
                break
            train_data[n:n+batch.shape[0]] = batch.view(batch_size, -1)
            n += batch_size
        train_data = train_data.numpy()
        data_mean = np.mean(train_data, axis=0)
        train_centered = train_data - data_mean
        Sigma = np.dot(train_centered.T, train_centered) / train_centered.shape[0]
        U, Lambda, _ = np.linalg.svd(Sigma)
        ZCA_matrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(Lambda + 1e-5)), U.T))
        ZCA_matrix_inv = np.linalg.inv(ZCA_matrix)
        print 'Whitening matrix calculated.'
        return Variable(torch.from_numpy(ZCA_matrix)), Variable(torch.from_numpy(ZCA_matrix_inv)), Variable(torch.from_numpy(data_mean))

    def process_input(self, input):
        """
        Whitens or scales the input.
        :param input: the input data
        """
        if self.whitening_matrix is not None:
            return torch.mm(input - self.data_mean, self.whitening_matrix)
        else:
            return input / 255.

    def process_output(self, mean):
        """
        Colors or un-scales the output.
        :param mean: the mean of the output distribution
        :return the unnormalized or unscaled mean
        """
        if self.whitening_matrix is not None:
            return self.data_mean + torch.mm(mean, self.inverse_whitening_matrix)
        else:
            return 255. * mean

    def get_input_encoding(self, input):
        """
        Encoding at the bottom level.
        :param input: the input data
        :return the encoding of the data
        """
        if 'bottom_error' in self.encoding_form or 'bottom_norm_error' in self.encoding_form:
            assert self.output_dist is not None, 'Cannot encode error. Output distribution is None.'
        encoding = None
        if 'posterior' in self.encoding_form:
            encoding = input if self.whitening_matrix is not None else input - 0.5
        if 'bottom_error' in self.encoding_form:
            error = input - self.output_dist.mean.detach().mean(dim=1)
            encoding = error if encoding is None else torch.cat((encoding, error), dim=1)
        if 'norm_bottom_error' in self.encoding_form:
            error = input - self.output_dist.mean.detach().mean(dim=1)
            norm_error = error / torch.norm(error, 2, 1, True)
            encoding = norm_error if encoding is None else torch.cat((encoding, norm_error), dim=1)
        if 'log_bottom_error' in self.encoding_form:
            log_error = torch.log(torch.abs(input - self.output_dist.mean.detach().mean(dim=1)))
            encoding = log_error if encoding is None else torch.cat((encoding, log_error), dim=1)
        if 'sign_bottom_error' in self.encoding_form:
            sign_error = torch.sign(input - self.output_dist.mean.detach())
            encoding = sign_error if encoding is None else torch.cat((encoding, sign_error), dim=1)
        if 'bottom_norm_error' in self.encoding_form:
            error = input - self.output_dist.mean.detach().mean(dim=1)
            norm_error = None
            if self.output_distribution == 'gaussian':
                norm_error = error / torch.exp(self.output_dist.log_var.detach().mean(dim=1))
            elif self.output_distribution == 'bernoulli':
                mean = self.output_dist.mean.detach().mean(dim=1)
                norm_error = error * torch.exp(- torch.log(mean + 1e-5) - torch.log(1 - mean + 1e-5))
            encoding = norm_error if encoding is None else torch.cat((encoding, norm_error), dim=1)
        if 'norm_bottom_norm_error' in self.encoding_form:
            error = input - self.output_dist.mean.detach().mean(dim=1)
            norm_error = None
            if self.output_distribution == 'gaussian':
                norm_error = error / torch.exp(self.output_dist.log_var.detach().mean(dim=1))
            elif self.output_distribution == 'bernoulli':
                mean = self.output_dist.mean.detach().mean(dim=1)
                norm_error = error * torch.exp(- torch.log(mean + 1e-5) - torch.log(1 - mean + 1e-5))
            norm_norm_error = norm_error / torch.norm(norm_error, 2, 1, True)
            encoding = norm_norm_error if encoding is None else torch.cat((encoding, norm_norm_error), dim=1)
        return encoding

    def encode(self, input):
        """
        Encodes the input into an updated posterior estimate.
        :param input: the data input
        :return None
        """
        if self.state_optimizer is None:
            if self._cuda_device is not None:
                input = input.cuda(self._cuda_device)
            input = self.process_input(input.view(-1, self.input_size))

            h = self.get_input_encoding(input)
            for latent_level in self.levels:
                if self.concat_variables:
                    h = torch.cat([h, latent_level.encode(h)], dim=1)
                else:
                    h = latent_level.encode(h)

    def decode(self, n_samples=0, generate=False):
        """
        Decodes the posterior (prior) estimate to get a reconstruction (sample).
        :param n_samples: number of samples to decode
        :param generate: flag to generate or reconstruct the data
        :return output distribution of reconstruction/sample
        """
        if n_samples == 0:
            n_samples = self.n_training_samples
        h = Variable(torch.zeros(self.batch_size, n_samples, self.top_size))
        if self._cuda_device is not None:
            h = h.cuda(self._cuda_device)
        concat = False
        for latent_level in self.levels[::-1]:
            if self.concat_variables and concat:
                h = torch.cat([h, latent_level.decode(h, n_samples, generate)], dim=2)
            else:
                h = latent_level.decode(h, n_samples, generate)
            concat = True

        h = h.view(-1, h.size()[2])
        h = self.output_decoder(h)
        # self.output_dist.mean = self.process_output(self.mean_output(h)) / 255.
        mean_out = self.mean_output(h)
        mean_out = mean_out.view(self.batch_size, n_samples, self.input_size)
        self.output_dist.mean = mean_out

        if self.output_distribution == 'gaussian':
            if self.constant_variances:
                if self.single_output_variance:
                    self.output_dist.log_var = torch.clamp(self.trainable_log_var * Variable(torch.ones(self.batch_size, n_samples, self.input_size).cuda(self._cuda_device)), -7, 15)
                else:
                    self.output_dist.log_var = torch.clamp(self.trainable_log_var.view(1, 1, -1).repeat(self.batch_size, n_samples, 1), -7., 15)
            else:
                log_var_out = self.log_var_output(h)
                log_var_out = log_var_out.view(self.batch_size, n_samples, self.input_size)
                self.output_dist.log_var = torch.clamp(log_var_out, -7., 15)

        self.reconstruction = 255. * self.output_dist.mean[:, 0, :]
        # self.reconstruction = self.process_output(self.output_dist.mean)
        return self.output_dist

    def kl_divergences(self, averaged=False):
        """
        Returns a list containing kl divergences at each level.
        :param averaged: whether to average across the batch dimension
        :return list of KL divergences at each level
        """
        kl = []
        for latent_level in range(len(self.levels)-1):
            # for latent_level in range(len(self.levels)):
            kl.append(torch.clamp(self.levels[latent_level].kl_divergence(), min=self.kl_min).sum(dim=2))
        kl.append(self.levels[-1].latent.analytical_kl().sum(dim=2))
        if averaged:
            return [level_kl.mean() for level_kl in kl]
        else:
            return kl

    def conditional_log_likelihoods(self, input, averaged=False):
        """
        Returns the conditional likelihood.
        :param input: the input data
        :param averaged: whether to average across the batch dimension
        :return the conditional log likelihood
        """
        if self._cuda_device is not None:
            input = input.cuda(self._cuda_device)
        input = input.view(-1, 1, self.input_size) / 255.
        # input = self.process_input(input.view(-1, self.input_size))
        n_samples = self.output_dist.mean.data.shape[1]
        input = input.repeat(1, n_samples, 1)
        log_prob = self.output_dist.log_prob(sample=input)
        if self.output_distribution == 'gaussian':
            log_prob = log_prob - np.log(256.)
        log_prob = log_prob.sum(dim=2)
        if averaged:
            return log_prob.mean()
        else:
            return log_prob

    def elbo(self, input, averaged=False):
        """
        Returns the ELBO.
        :param input: the input data
        :param averaged: whether to average across the batch dimension
        :return the ELBO
        """
        cond_like = self.conditional_log_likelihoods(input)
        kl = sum(self.kl_divergences())
        lower_bound = (cond_like - self.kl_weight * kl).mean(dim=1)  # average across sample dimension
        if averaged:
            return lower_bound.mean()
        else:
            return lower_bound

    def losses(self, input, averaged=False):
        """
        Returns all losses.
        :param input: the input data
        :param averaged: whether to average across the batch dimension
        """
        cll = self.conditional_log_likelihoods(input)
        cond_log_like = cll.mean(dim=1)
        kld = self.kl_divergences()
        kl_div = [kl.mean(dim=1) for kl in kld]
        lower_bound = (cll - self.kl_weight * sum(kld)).mean(dim=1)
        if averaged:
            return lower_bound.mean(), cond_log_like.mean(), [level_kl.mean() for level_kl in kl_div]
        else:
            return lower_bound, cond_log_like, kl_div

    def state_gradients(self):
        """
        Get the gradients for the approximate posterior parameters.
        :return: dictionary containing keys for each level with lists of gradients
        """
        state_grads = {}
        for level_num, latent_level in enumerate(self.levels):
            state_grads[level_num] = latent_level.state_gradients()
        return state_grads

    def reset_state(self, mean=None, log_var=None, from_prior=True):
        """Resets the posterior estimate."""
        for latent_level in self.levels:
            latent_level.reset(mean=mean, log_var=log_var, from_prior=from_prior)

    def trainable_state(self):
        """Makes the posterior estimate trainable."""
        for latent_level in self.levels:
            latent_level.trainable_state()

    def not_trainable_state(self):
        """Makes the posterior estimate not trainable."""
        for latent_level in self.levels:
            latent_level.not_trainable_state()

    def parameters(self):
        """Returns a list containing all parameters."""
        return self.encoder_parameters() + self.decoder_parameters() + self.state_parameters()

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
        if self.whitening_matrix is not None:
            self.whitening_matrix = self.whitening_matrix.cuda(device_id)
            self.inverse_whitening_matrix = self.inverse_whitening_matrix.cuda(device_id)
            self.data_mean = self.data_mean.cuda(device_id)

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
        if self.whitening_matrix is not None:
            self.whitening_matrix = self.whitening_matrix.cpu()
            self.inverse_whitening_matrix = self.inverse_whitening_matrix.cpu()
            self.data_mean = self.data_mean.cpu()


class ConvLatentVariableModel(object):

    def __init__(self, train_config, arch, data_loader):
        raise NotImplementedError
