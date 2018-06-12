import torch
import torch.nn as nn
from lib.distributions import Normal
from latent_variable import LatentVariable
from lib.modules.layers import FullyConnectedLayer
from lib.modules.networks import FullyConnectedNetwork


class FullyConnectedLatentVariable(LatentVariable):
    """
    A fully-connected Gaussian latent variable.

    Args:
        latent_config (dict): dictionary containing variable configuration
                              parameters
    """
    def __init__(self, latent_config):
        super(FullyConnectedLatentVariable, self).__init__(latent_config)
        self._construct(**latent_config)

    def _construct(self, n_in, n_variables, inference_procedure):
        """
        Method to construct the latent variable from the latent_config dictionary
        """
        self.inference_procedure = inference_procedure
        self.inf_mean_output = FullyConnectedLayer({'n_in': n_in[0],
                                                    'n_out': n_variables})
        self.inf_log_var_output = FullyConnectedLayer({'n_in': n_in[0],
                                                       'n_out': n_variables})
        if 'gradient' in self.inference_procedure or 'error' in self.inference_procedure:
            self.approx_post_mean_gate = FullyConnectedLayer({'n_in': n_in[0],
                                                              'n_out': n_variables,
                                                              'non_linearity': 'sigmoid'})
            self.approx_post_log_var_gate = FullyConnectedLayer({'n_in': n_in[0],
                                                                 'n_out': n_variables,
                                                                 'non_linearity': 'sigmoid'})
            # self.close_gates()
        # prior inputs
        self.prior_mean = self.prior_log_var = None
        if n_in[1] is not None:
            self.prior_mean = FullyConnectedLayer({'n_in': n_in[1],
                                                   'n_out': n_variables})
            self.prior_log_var = FullyConnectedLayer({'n_in': n_in[1],
                                                      'n_out': n_variables})
        # distributions
        self.approx_post = Normal(n_variables=n_variables)
        self.prior = Normal(n_variables=n_variables)
        self.approx_post.re_init()
        self.prior.re_init(sample_dim=True)

    def infer(self, input):
        """
        Method to perform inference.

        Args:
            input (Tensor): input to the inference procedure
        """
        approx_post_mean = self.inf_mean_output(input)
        approx_post_log_var = self.inf_log_var_output(input)
        if self.inference_procedure == ['observation']:
            self.approx_post.mean = approx_post_mean
            self.approx_post.log_var = torch.clamp(approx_post_log_var, -15, 5)
        else:
            # gated highway update
            approx_post_mean_gate = self.approx_post_mean_gate(input)
            self.approx_post.mean = approx_post_mean_gate * self.approx_post.mean.detach() \
                                    + (1 - approx_post_mean_gate) * approx_post_mean
            approx_post_log_var_gate = self.approx_post_log_var_gate(input)
            self.approx_post.log_var = torch.clamp(approx_post_log_var_gate * self.approx_post.log_var.detach() \
                                       + (1 - approx_post_log_var_gate) * approx_post_log_var, -15, 5)

        # retain the gradients (for inference)
        self.approx_post.mean.retain_grad()
        self.approx_post.log_var.retain_grad()

    def generate(self, input, gen, n_samples):
        """
        Method to generate, i.e. run the model forward.

        Args:
            input (Tensor): input to the generative procedure
            gen (boolean): whether to sample from approximate poserior (False) or
                            the prior (True)
            n_samples (int): number of samples to draw
        """
        if input is not None:
            b, s, n = input.data.shape
            input = input.view(b * s, n)
            self.prior.mean = self.prior_mean(input).view(b, s, -1)
            self.prior.log_var = torch.clamp(self.prior_log_var(input).view(b, s, -1), -15, 5)
        dist = self.prior if gen else self.approx_post
        sample = dist.sample(n_samples, resample=True)
        return sample

    def re_init(self, batch_size):
        """
        Method to reinitialize the approximate posterior and prior over the variable.
        """
        # TODO: this is wrong. we shouldnt set the posterior to the prior then zero out the prior...
        self.re_init_approx_posterior(batch_size)
        self.prior.re_init(batch_size=batch_size, sample_dim=True)

    def re_init_approx_posterior(self, batch_size):
        """
        Method to reinitialize the approximate posterior.
        """
        mean = self.prior.mean.detach().mean(dim=1)
        log_var = self.prior.log_var.detach().mean(dim=1)
        self.approx_post.re_init(mean, log_var, batch_size=batch_size)
        # retain the gradients (for inference)
        # self.approx_post.mean.retain_grad()
        # self.approx_post.log_var.retain_grad()

    def error(self, averaged=True):
        """
        Calculates Gaussian error for encoding.

        Args:
            averaged (boolean): whether or not to average over samples
        """
        sample = self.approx_post.sample()
        n_samples = sample.shape[1]
        prior_mean = self.prior.mean.detach()
        if len(prior_mean.shape) == 2:
            prior_mean = prior_mean.unsqueeze(1).repeat(1, n_samples, 1)
        prior_log_var = self.prior.log_var.detach()
        if len(prior_log_var.shape) == 2:
            prior_log_var = prior_log_var.unsqueeze(1).repeat(1, n_samples, 1)
        n_error = (sample - prior_mean) / torch.exp(prior_log_var + 1e-7)
        if averaged:
            n_error = n_error.mean(dim=1)
        return n_error

    def close_gates(self):
        nn.init.constant(self.approx_post_mean_gate.linear.bias, 5.)
        nn.init.constant(self.approx_post_log_var_gate.linear.bias, 5.)

    def inference_parameters(self):
        """
        Method to obtain inference parameters.
        """
        params = nn.ParameterList()
        params.extend(list(self.inf_mean_output.parameters()))
        params.extend(list(self.inf_log_var_output.parameters()))
        if self.inference_procedure != 'direct':
            params.extend(list(self.approx_post_mean_gate.parameters()))
            params.extend(list(self.approx_post_log_var_gate.parameters()))
        return params

    def generative_parameters(self):
        """
        Method to obtain generative parameters.
        """
        params = nn.ParameterList()
        if self.prior_mean is not None:
            params.extend(list(self.prior_mean.parameters()))
            params.extend(list(self.prior_log_var.parameters()))
        return params

    def approx_posterior_parameters(self):
        return [self.approx_post.mean.detach(), self.approx_post.log_var.detach()]

    def approx_posterior_gradients(self):
        assert self.approx_post.mean.grad is not None, 'Approximate posterior gradients are None.'
        grads = [self.approx_post.mean.grad.detach()]
        grads += [self.approx_post.log_var.grad.detach()]
        # for grad in grads:
        #     grad.volatile = False
        return grads
