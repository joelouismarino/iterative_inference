import torch.nn as nn


class Network(nn.Module):
    """
    Abstract class definition for a neural network.

    Args:
        network_config (dict): dictionary containing layer configuration parameters
    """
    def __init__(self, network_config):
        super(Network, self).__init__()
        self.network_config = network_config

    def forward(self, input):
        """
        Abstract method to perform forward computation.
        """
        raise NotImplementedError

    def re_init(self):
        """
        Method to reinitialize any state variables within the network. Overwrite
        this method if there are any such variables.
        """
        pass
