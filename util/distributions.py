import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class DiagonalGaussian(object):

    def __init__(self, mean, log_var):
        self.mean = mean
        self.log_var = log_var
        self._sample = None

    def sample(self, resample=False):
        if self._sample is None or resample:
            random_normal = Variable(torch.normal(torch.zeros(self.mean.size()), torch.ones(self.mean.size())))
            self._sample = self.mean + torch.exp(0.5 * self.log_var) * random_normal
        return self._sample

    def log_prob(self, sample=None):
        if sample is None:
            sample = self.sample()
        return -0.5 * (np.log(2 * np.pi) + log_var + (sample - mean)**2 / torch.exp(log_var))
