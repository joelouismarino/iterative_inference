import math
import torch
import torch.nn as nn
from distribution import Distribution

# import numpy as np
# from torch.autograd import Variable


class Normal(Distribution):
    """
    A normal (Gaussian) density.

    Args:
        mean (tensor): the mean of the density
        log_var (tensor): the log variance of the density
    """
    def __init__(self, mean=None, log_var=None):
        self.mean = mean
        self.log_var = log_var
        self._sample = None

    def sample(self, n_samples=1, resample=False):
        """
        Draws a tensor of samples.
        :param n_samples: number of samples to draw
        :param resample: whether to resample or just use current sample
        :return: a (batch_size x n_samples x n_variables) tensor of samples
        """
        if self._sample is None or resample:
            mean = self.mean
            std = self.log_var.mul(0.5).exp_()
            if len(self.mean.size()) == 2:
                mean = mean.unsqueeze(1).repeat(1, n_samples, 1)
                std = std.unsqueeze(1).repeat(1, n_samples, 1)
            rand_normal = Variable(mean.data.new(mean.size()).normal_())
            self._sample = rand_normal.mul_(std).add_(mean)
        return self._sample

    def log_prob(self, sample=None):
        """
        Estimates the log probability, evaluated at the sample.
        :param sample: the sample to evaluate log probability at
        :return: a (batch_size x n_samples x n_variables) estimate of log probabilities
        """
        if sample is None:
            sample = self.sample()
        assert self.mean is not None and self.log_var is not None, 'Mean or log variance are None.'
        n_samples = sample.size()[1]
        if len(self.mean.data.shape) == 2:
            mean = self.mean.unsqueeze(1).repeat(1, n_samples, 1)
        else:
            mean = self.mean
        if len(self.log_var.data.shape) == 2:
            log_var = self.log_var.unsqueeze(1).repeat(1, n_samples, 1)
        else:
            log_var = self.log_var
        return -0.5 * (log_var + np.log(2 * np.pi) + torch.pow(sample - mean, 2) / (torch.exp(log_var) + 1e-5))

    def reset_mean(self, value=None):
        """
        Resets the mean to a particular value.
        :param value: the value to set as the mean, defaults to zero
        :return: None
        """
        assert self.mean is not None or value is not None, 'Mean is None.'
        mean = value if value is not None else torch.zeros(self.mean.size())
        if self._cuda_device is not None:
            mean = mean.cuda(self._cuda_device)
        self.mean = Variable(mean, requires_grad=True)
        self._sample = None

    def reset_log_var(self, value=None):
        """
        Resets the log variance to a particular value.
        :param value: the value to set as the log variance, defaults to zero
        :return: None
        """
        assert self.log_var is not None or value is not None, 'Log variance is None.'
        log_var = value if value is not None else torch.zeros(self.log_var.size())
        if self._cuda_device is not None:
            log_var = log_var.cuda(self._cuda_device)
        self.log_var = Variable(log_var, requires_grad=True)
        self._sample = None

    def mean_trainable(self):
        """
        Makes the mean a trainable variable.
        :return: None
        """
        assert self.mean is not None, 'Mean is None.'
        self.mean = Variable(self.mean.data.clone(), requires_grad=True)

    def log_var_trainable(self):
        """
        Makes the log variance a trainable variable.
        :return: None
        """
        assert self.log_var is not None, 'Log variance is None.'
        self.log_var = Variable(self.log_var.data.clone(), requires_grad=True)

    def mean_not_trainable(self):
        """
        Makes the mean a non-trainable variable.
        :return: None
        """
        self.mean.requires_grad = False

    def log_var_not_trainable(self):
        """
        Makes the log variance a non-trainable variable.
        :return: None
        """
        self.log_var.requires_grad = False

    def state_parameters(self):
        """
        Gets the state parameters for this variable.
        :return: tuple of mean and log variance
        """
        return self.mean, self.log_var
