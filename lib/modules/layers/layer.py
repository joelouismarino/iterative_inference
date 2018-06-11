import torch.nn as nn


class Layer(nn.Module):
    """
    Abstract class definition for a neural network layer.

    Args:
        layer_config (dict): dictionary containing layer configuration parameters
    """
    def __init__(self, layer_config):
        super(Layer, self).__init__()
        self.layer_config = layer_config

    def forward(self, input):
        """
        Abstract method to perform forward computation.
        """
        raise NotImplementedError

    def re_init(self):
        """
        Method to reinitialize any state variables within the layer. Overwrite
        this method if there are any such variables.
        """
        pass
