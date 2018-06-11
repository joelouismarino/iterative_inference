from distribution import Distribution
import torch
import torch.nn as nn


class Bernoulli(Distribution):
    """
    A Bernoulli distribution.

    Args:
        mean (tensor): the mean of the density
    """
    def __init__(self, mean=None):
        self.mean = mean
        self._sample = None

    def sample(self, n_samples=1, resample=False):
        """
        Draws a tensor of samples.

        Args:
            n_samples: number of samples to draw
            resample: whether to resample or just use current sample

        Return:
            a (batch_size x n_samples x n_variables) tensor of samples
        """
        if self._sample is None or resample:
            assert self.mean is not None, 'Mean is None.'
            mean = self.mean.unsqueeze(1).repeat(1, n_samples, 1)
            self._sample = torch.bernoulli(mean)
        return self._sample

    def log_prob(self, sample=None):
        """
        Estimates the log probability, evaluated at the sample.
        :param sample: the sample to evaluate log probability at
        :return: a (batch_size x n_samples x n_variables) estimate of log probabilities
        """
        if sample is None:
            sample = self.sample()
        assert self.mean is not None, 'Mean is None.'
        n_samples = sample.size()[1]
        if len(self.mean.data.shape) == 2:
            mean = self.mean.unsqueeze(1).repeat(1, n_samples, 1)
        else:
            mean = self.mean
        return sample * torch.log(mean + 1e-7) + (1 - sample) * torch.log(1 - mean + 1e-7)

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
        mean = Variable(mean, requires_grad=self.mean.requires_grad)
        self.mean = mean
        self._sample = None

    def mean_trainable(self):
        """
        Makes the mean a trainable variable.
        :return: None
        """
        assert self.mean is not None, 'Mean is None.'
        self.mean = Variable(self.mean.data, requires_grad=True)

    def mean_not_trainable(self):
        """
        Makes the mean a non-trainable variable.
        :return: None
        """
        self.mean.requires_grad = False

    def state_parameters(self):
        """
        Gets the state parameters for this variable.
        :return: tuple of mean and log variance
        """
        return self.mean
