import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Layer normalization layer. Normalizes the input over the last dimension.
    """
    def __init__(self):
        super(LayerNorm, self).__init__()

    def forward(self, input):
        norm_dim = len(input.data.shape) - 1
        mean = input.mean(dim=norm_dim, keepdim=True)
        std = input.std(dim=norm_dim, keepdim=True)
        input = (input - mean) / (std + 1e-7)
        return input
