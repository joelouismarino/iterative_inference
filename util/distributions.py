import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class DiagonalGaussian(object):

    def __init__(self, mean, log_var):
        self.mean = mean
        self.log_var = log_var

    def sample(self):
        return self.mean + torch.exp(0.5 * self.log_var) * torch.normal(torch.zeros(self.mean.size()), torch.ones(self.mean.size()))

    def log_prob(self, sample):
        return -0.5 * (np.log(2 * np.pi) + log_var + (sample - mean)**2 / tf.exp(log_var))
