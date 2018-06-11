import torch.nn as nn


class LatentLevel(nn.Module):
    """
    Abstract class for a latent level.
    """
    def __init__(self, level_config):
        super(LatentLevel, self).__init__()
        self.level_config = level_config
        self.latent = None

    def infer(self, input):
        """
        Abstract method to perform inference.

        Args:
            input (Tensor): input to the inference procedure
        """
        raise NotImplementedError

    def generate(self, input, gen, n_samples):
        """
        Abtract method to generate, i.e. run the model forward.

        Args:
            input (Tensor): input to the generative procedure
            gen (boolean): whether to sample from approximate poserior (False) or
                            the prior (True)
            n_samples (int): number of samples to draw
        """
        raise NotImplementedError

    def re_init(self):
        """
        Abtract method to reinitialize the latent level (latent variable and any
        state variables in the generative / inference procedures).
        """
        raise NotImplementedError

    def inference_parameters(self):
        """
        Abstract method to obtain inference parameters.
        """
        raise NotImplementedError

    def generative_parameters(self):
        """
        Abstract method to obtain generative parameters.
        """
        raise NotImplementedError
