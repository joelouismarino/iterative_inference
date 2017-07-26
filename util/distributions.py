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
        self.trainable_mean = None
        self.trainable_log_var = None

    def sample(self, resample=False):
        if self._sample is None or resample:
            random_normal = Variable(torch.randn(self.mean.size()))
            if self._cuda_device is not None:
                random_normal = random_normal.cuda(self._cuda_device)
            self._sample = self.mean + torch.exp(0.5 * self.log_var) * random_normal
        return self._sample

    def log_prob(self, sample=None):
        if sample is None:
            sample = self.sample()
        return -0.5 * (self.log_var + np.log(2 * np.pi) + torch.pow(sample - self.mean, 2) / (torch.exp(self.log_var) + 1e-7))

    def reset_mean(self):
        self.mean.data.fill_(0.)
        self.sample = None

    def reset_log_var(self):
        self.log_var.data.fill_(0.)
        self.sample = None

    def mean_trainable(self):
        self.trainable_mean = Variable(torch.zeros(self.mean.size()[1:]), requires_grad=True)
        if self._cuda_device is not None:
            self.trainable_mean = self.trainable_mean.cuda(self._cuda_device)
        if len(self.mean_size()) == 2:
            self.mean = self.trainable_mean.unsqueeze(0).repeat(self.mean.size()[0], 1)
        else:
            self.mean = self.trainable_mean.unsqueeze(0).repeat(self.mean.size()[0], 1, 1, 1)

    def log_var_trainable(self):
        self.trainable_log_var = Variable(torch.zeros(self.log_var.size()[1:]), requires_grad=True)
        if self._cuda_device is not None:
            self.trainable_log_var = self.trainable_log_var.cuda(self._cuda_device)
        if len(self.log_var_size()) == 2:
            self.log_var = self.trainable_log_var.unsqueeze(0).repeat(self.log_var.size()[0], 1)
        else:
            self.log_var = self.trainable_log_var.unsqueeze(0).repeat(self.log_var.size()[0], 1, 1, 1)

    def state_parameters(self):
        return self.mean, self.log_var

    def cuda(self, device_id=0):
        if self.trainable_mean is not None:
            self.trainble_mean = self.trainable_mean.cuda(device_id)
        if self.trainable_log_var is not None:
            self.trainable_log_var = self.trainable_log_var.cuda(device_id)
        self.mean = self.mean.cuda(device_id)
        self.log_var = self.log_var.cuda(device_id)
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