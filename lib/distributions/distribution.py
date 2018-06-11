import torch.nn as nn


class Distribution(nn.Module):
    """
    Abstract class for a probability distribution. All probability distributions
    should inherit from this class.
    """
    def __init__(self):
        super(Distribution, self).__init__()

    def sample(self, n_samples=1, resample=False):
        raise NotImplementedError

    def log_prob(self, value):
        raise NotImplementedError

    def re_init(self):
        raise NotImplementedError
