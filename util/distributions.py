import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class DiagonalGaussian(object):

    def __init__(self, mean, log_var):
        self.mean = mean
        self.log_var = log_var
        self._sample = None
        self._cuda_device = None

    def sample(self, resample=False):
        if self._sample is None or resample:
            random_normal = Variable(torch.randn(self.mean.size()))
            if self._cuda_device:
                random_normal.cuda(self._cuda_device)
            self._sample = self.mean + torch.exp(0.5 * self.log_var) * random_normal
        return self._sample

    def log_prob(self, sample=None):
        if sample is None:
            sample = self.sample()
        return -0.5 * (np.log(2 * np.pi) + self.log_var + (sample - self.mean)**2 / (torch.exp(self.log_var) + 1e-7))

    def reset_mean(self):
        self.mean.data.fill_(0.)
        self.sample = None

    def reset_log_var(self):
        self.log_var.data.fill_(0.)
        self.sample = None

    def mean_trainable(self, trainable=True):
        self.mean.requires_grad = trainable

    def log_var_trainable(self, trainable=True):
        self.log_var.requires_grad = trainable

    def state_parameters(self):
        return self.mean, self.log_var

    def cuda(self, device_id):
        self.mean.cuda()
        self.log_var.cuda()
        self._cuda_device = device_id


class Bernoulli(object):

    def __init__(self, mean):
        self.mean = mean
        self._sample = None
        self._cuda_device = None

    def sample(self, resample=False):
        self._sample = torch.bernoulli(self.mean)
        return self._sample

    def log_prob(self, sample=None):
        if sample is None:
            sample = self.sample()
        return sample * torch.log(self.mean + 1e-7) + (1 - sample) * torch.log(1 - self.mean + 1e-7)

    def reset_mean(self):
        self.mean.data.fill_(0.)
        self.sample = None

    def mean_trainable(self, trainable=True):
        self.mean.requires_grad = trainable

    def state_parameters(self):
        return self.mean

    def cuda(self, device_id):
        self.mean.cuda()
        self._cuda_device = device_id