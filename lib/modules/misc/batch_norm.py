import torch.nn as nn


class BatchNorm(nn.Module):
    """
    Batch normalization layer. Normalizes the input over the first dimension.
    """
    def __init__(self):
        super(BatchNorm, self).__init__()

    def forward(self, input):
        norm_dim = 0
        mean = input.mean(dim=norm_dim, keepdim=True)
        std = input.std(dim=norm_dim, keepdim=True)
        input = (input - mean) / (std + 1e-7)
        return input
