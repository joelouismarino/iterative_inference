import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class PointEstimate(object):

    def __init__(self, mean=None):
        self.mean = mean
        self._cuda_device = None

    def sample(self, *args, **kwargs):
        assert self.mean is not None, 'Point estimate is None.'
        return self.mean

    def log_prob(self, *args, **kwargs):
        log_p = Variable(torch.zeros(self.mean.data.shape))
        if self._cuda_device is not None:
            return log_p.cuda(self._cuda_device)
        return log_p

    def reset_mean(self, value=None):
        assert self.mean is not None or value is not None, 'Mean is None.'
        mean = value if value is not None else torch.zeros(self.mean.size())
        if self._cuda_device is not None:
            mean = mean.cuda(self._cuda_device)
        self.mean = Variable(mean, requires_grad=True)
        self._sample = None

    def mean_trainable(self):
        assert self.mean is not None, 'Mean is None.'
        self.mean = Variable(self.mean.data, requires_grad=True)

    def mean_not_trainable(self):
        self.mean.requires_grad = False

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
