import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class PointEstimate(object):

    def __init__(self, mean=None):
        self.mean = mean
        self._cuda_device = None

    def sample(self):
        assert self.mean is not None, 'Point estimate is None.'
        return self.mean

    def log_prob(self, sample=None):
        #assert self.mean is not None, 'Point estimate is None.'
        #if sample is None:
        #    sample = self.mean
        pass

    def reset_mean(self, value=None):
        assert self.mean is not None or value is not None, 'Mean is None.'
        mean = value if value is not None else torch.zeros(self.mean.size())
        if self._cuda_device is not None:
            mean = mean.cuda(self._cuda_device)
        mean = Variable(mean, requires_grad=self.mean.requires_grad)
        self.mean = mean
        self._sample = None

    def mean_trainable(self):
        assert self.mean is not None, 'Mean is None.'
        self.mean = Variable(self.mean.data, requires_grad=True)

    def state_parameters(self):
        return self.mean

    def cuda(self, device_id=0):
        if self.mean is not None:
            self.mean = Variable(self.mean.data.cuda(device_id), requires_grad=self.mean.requires_grad)
        self._cuda_device = device_id

    def cpu(self):
        if self.mean is not None:
            self.mean = self.mean.cpu()
        self._cuda_device = None


class DiagonalGaussian(object):

    def __init__(self, mean=None, log_var=None):
        self.mean = mean
        self.log_var = log_var
        self._sample = None
        self._cuda_device = None

    def sample(self, resample=False):
        if self._sample is None or resample:
            random_normal = Variable(torch.randn(self.mean.size()))
            if self._cuda_device is not None:
                random_normal = random_normal.cuda(self._cuda_device)
            assert self.mean is not None and self.log_var is not None, 'Mean or log variance are None.'
            self._sample = self.mean + torch.exp(0.5 * self.log_var) * random_normal
        return self._sample

    def log_prob(self, sample=None):
        if sample is None:
            sample = self.sample()
        assert self.mean is not None and self.log_var is not None, 'Mean or log variance are None.'
        return -0.5 * (self.log_var + np.log(2 * np.pi) + torch.pow(sample - self.mean, 2) / (torch.exp(self.log_var) + 1e-7))

    def reset_mean(self, value=None):
        assert self.mean is not None or value is not None, 'Mean is None.'
        mean = value if value is not None else torch.zeros(self.mean.size())
        if self._cuda_device is not None:
            mean = mean.cuda(self._cuda_device)
        mean = Variable(mean, requires_grad=self.mean.requires_grad)
        self.mean = mean
        self._sample = None

    def reset_log_var(self, value=None):
        assert self.log_var is not None or value is not None, 'Log variance is None.'
        log_var = value if value is not None else torch.zeros(self.log_var.size())
        if self._cuda_device is not None:
            log_var = log_var.cuda(self._cuda_device)
        log_var = Variable(log_var, requires_grad=self.log_var.requires_grad)
        self.log_var = log_var
        self._sample = None

    def mean_trainable(self):
        assert self.mean is not None, 'Mean is None.'
        self.mean = Variable(self.mean.data, requires_grad=True)

    def log_var_trainable(self):
        assert self.log_var is not None, 'Log variance is None.'
        self.log_var = Variable(self.log_var.data, requires_grad=True)

    def state_parameters(self):
        return self.mean, self.log_var

    def cuda(self, device_id=0):
        if self.mean is not None:
            self.mean = Variable(self.mean.data.cuda(device_id), requires_grad=self.mean.requires_grad)
        if self.log_var is not None:
            self.log_var = Variable(self.log_var.data.cuda(device_id), requires_grad=self.log_var.requires_grad)
        if self._sample is not None:
            self._sample = Variable(self._sample.data.cuda(device_id))
        self._cuda_device = device_id

    def cpu(self):
        if self.mean is not None:
            self.mean = self.mean.cpu()
        if self.log_var is not None:
            self.log_var = self.log_var.cpu()
        if self._sample is not None:
            self._sample = self.sample.cpu()
        self._cuda_device = None


class Bernoulli(object):

    def __init__(self, mean=None):
        self.mean = mean
        self._sample = None
        self._cuda_device = None

    def sample(self, resample=False):
        if self._sample is None or resample:
            assert self.mean is not None, 'Mean is None.'
            self._sample = torch.bernoulli(self.mean)
        return self._sample

    def log_prob(self, sample=None):
        if sample is None:
            sample = self.sample()
        assert self.mean is not None, 'Mean is None.'
        return sample * torch.log(self.mean + 1e-7) + (1 - sample) * torch.log(1 - self.mean + 1e-7)

    def reset_mean(self, value=None):
        assert self.mean is not None or value is not None, 'Mean is None.'
        mean = value if value is not None else torch.zeros(self.mean.size())
        if self._cuda_device is not None:
            mean = mean.cuda(self._cuda_device)
        mean = Variable(mean, requires_grad=self.mean.requires_grad)
        self.mean = mean
        self._sample = None

    def mean_trainable(self):
        assert self.mean is not None, 'Mean is None.'
        self.mean = Variable(self.mean.data, requires_grad=True)

    def state_parameters(self):
        return self.mean

    def cuda(self, device_id):
        if self.mean is not None:
            self.mean = self.mean.cuda(device_id)
        self._cuda_device = device_id

    def cpu(self):
        if self.mean is not None:
            self.mean = self.mean.cpu()
        self._cuda_device = None
