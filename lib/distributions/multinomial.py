from distribution import Distribution
import torch
import torch.nn as nn


class Multinomial(Distribution):

    def __init__(self):
        pass

    def re_init(self):
        raise NotImplementedError

    def log_prob(self, value):
        maxval  = torch.max(self.mean, dim=2, keepdim=True)[0]
        logsoftmax  = self.mean - (maxval + torch.log(torch.sum(torch.exp(self.mean - maxval), dim=2, keepdim=True) + 1e-6))
        return logsoftmax * sample

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
