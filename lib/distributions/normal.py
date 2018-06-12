import math
import torch
import torch.nn as nn
from distribution import Distribution
from torch.autograd import Variable


class Normal(Distribution):
    """
    A normal (Gaussian) density.

    Args:
        mean (tensor): the mean of the density
        log_var (tensor): the log variance of the density
    """
    def __init__(self, mean=None, log_var=None, n_variables=1):
        super(Normal, self).__init__()
        self.mean_reset_value = nn.Parameter(torch.zeros(n_variables))
        self.log_var_reset_value = nn.Parameter(torch.zeros(n_variables))
        self.mean = mean
        self.log_var = log_var
        self._sample = None

    def sample(self, n_samples=1, resample=False):
        """
        Draw samples from the distribution.

        Args:
            n_samples (int): number of samples to draw
            resample (bool): whether to resample or just use current sample
        """
        if self._sample is None or resample:
            mean = self.mean
            std = self.log_var.mul(0.5).exp_()
            if len(self.mean.size()) == 2:
                mean = mean.unsqueeze(1).repeat(1, n_samples, 1)
                std = std.unsqueeze(1).repeat(1, n_samples, 1)
            rand_normal = mean.data.new(mean.size()).normal_()
            self._sample = rand_normal.mul_(std).add_(mean)
        return self._sample

    def log_prob(self, value):
        """
        Estimate the log probability (density), evaluated at the value or interval.

        Args:
            value (Variable, tensor, or tuple): the value or interval at/over
                                                which to evaluate
        """
        if type(value) == tuple:
            # evaluate the log probability mass within the interval
            return torch.log(self.cdf(value[1]) - self.cdf(value[0]) + 1e-6)
        else:
            # evaluate the log density at the value
            if value is None:
                value = self.sample()
            assert self.mean is not None and self.log_var is not None, 'Mean or log variance are None.'

            n_samples = value.data.shape[1]
            mean = self.mean
            log_var = self.log_var
            # unsqueeze the parameters along the sample dimension
            if len(mean.size()) == 2:
                mean = mean.unsqueeze(1).repeat(1, n_samples, 1)
                log_var = log_var.unsqueeze(1).repeat(1, n_samples, 1)
            elif len(mean.size()) == 4:
                mean = mean.unsqueeze(1).repeat(1, n_samples, 1, 1, 1)
                log_var = log_var.unsqueeze(1).repeat(1, n_samples, 1, 1, 1)

            return (log_var.add(math.log(2 * math.pi)).add_((value.sub(mean).pow_(2)).div_(log_var.exp().add(1e-5)))).mul_(-0.5)

    def cdf(self, value):
        """
        Evaluate the cumulative distribution function at the value.

        Args:
            value (Variable, tensor): the value at which to evaluate the cdf
        """
        n_samples = value.data.shape[1]
        mean = self.mean
        std = self.log_var.mul(0.5).exp_()
        # unsqueeze the parameters along the sample dimension
        if len(mean.size()) == 2:
            mean = mean.unsqueeze(1).repeat(1, n_samples, 1)
            std = std.unsqueeze(1).repeat(1, n_samples, 1)
        return (1 + torch.erf((value - mean) / (math.sqrt(2) * std).add(1e-5))).mul_(0.5)

    def re_init(self, mean_value=None, log_var_value=None, batch_size=1, sample_dim=False):
        """
        Re-initializes the distribution.

        Args:
            mean_value (tensor): the value to set as the mean, defaults to zero
            log_var_value (tensor): the value to set as the log variance,
                                    defaults to zero
            batch_size (int): the batch size for the values
            sample_dim (boolean): whether or not to include a sample dimension
        """
        self.re_init_mean(mean_value, batch_size, sample_dim)
        self.re_init_log_var(log_var_value, batch_size, sample_dim)

    def re_init_mean(self, value, batch_size=1, sample_dim=False):
        """
        Resets the mean to a particular value.

        Args:
            value (tensor): the value to set as the mean, defaults to zero
            batch_size (int): the batch size of the mean
            sample_dim (boolean): whether or not to include a sample dimension
        """
        mean = value if value is not None else self.mean_reset_value.unsqueeze(0)
        if mean.shape[0] != batch_size:
            mean = mean[:1].repeat(batch_size, 1)
        if sample_dim:
            mean = mean.unsqueeze(1)
        self.mean = Variable(mean, requires_grad=True)
        self._sample = None

    def re_init_log_var(self, value, batch_size=1, sample_dim=False):
        """
        Resets the log variance to a particular value.

        Args:
            value (tensor): the value to set as the log variance, defaults to zero
            batch_size (int): the batch size of the log variance
            sample_dim (boolean): whether or not to include a sample dimension
        """
        log_var = value if value is not None else self.log_var_reset_value.unsqueeze(0)
        if log_var.shape[0] != batch_size:
            log_var = log_var[:1].repeat(batch_size, 1)
        if sample_dim:
            log_var = log_var.unsqueeze(1)
        self.log_var = Variable(log_var, requires_grad=True)
        self._sample = None
