import torch.nn as nn
from latent_level import LatentLevel
from lib.modules.networks import FullyConnectedNetwork
from lib.modules.latent_variables import FullyConnectedLatentVariable


class FullyConnectedLatentLevel(LatentLevel):
    """
    Latent level with fully connected encoding and decoding functions.

    Args:
        level_config (dict): dictionary containing level configuration parameters
    """
    def __init__(self, level_config):
        super(FullyConnectedLatentLevel, self).__init__(level_config)
        self._construct(**level_config)

    def _construct(self, latent_config, inference_procedure,
                   inference_config=None, generative_config=None):
        """
        Method to construct the latent level from the level_config dictionary
        """
        self.latent = FullyConnectedLatentVariable(latent_config)
        self.inference_procedure = inference_procedure
        if inference_config is not None:
            self.inference_model = FullyConnectedNetwork(inference_config)
        else:
            self.inference_model = lambda x:x
        if generative_config is not None:
            self.generative_model = FullyConnectedNetwork(generative_config)
        else:
            self.generative_model = lambda x:x
        
    def _get_encoding_form(self, input):
        """
        Gets the appropriate input form for the inference procedure.
        """
        if self.inference_procedure == 'direct':
            return input
        else:
            raise NotImplementedError

    def infer(self, input):
        """
        Method to perform inference.

        Args:
            input (Tensor): input to the inference procedure
        """
        input = self._get_encoding_form(input)
        input = self.inference_model(input)
        self.latent.infer(input)

    def generate(self, input, gen, n_samples):
        """
        Method to generate, i.e. run the model forward.

        Args:
            input (Tensor): input to the generative procedure
            gen (boolean): whether to sample from approximate poserior (False) or
                            the prior (True)
            n_samples (int): number of samples to draw
        """
        if input is not None:
            b, s, n = input.data.shape
            input = self.generative_model(input.view(b * s, n)).view(b, s, -1)
        return self.latent.generate(input, gen=gen, n_samples=n_samples)

    def step(self):
        """
        Method to step the latent level forward in the sequence.
        """
        self.latent.step()

    def re_init(self):
        """
        Method to reinitialize the latent level (latent variable and any state
        variables in the generative / inference procedures).
        """
        self.latent.re_init()
        if 're_init' in dir(self.inference_model):
            self.inference_model.re_init()
        if 're_init' in dir(self.generative_model):
            self.generative_model.re_init()

    def inference_parameters(self):
        """
        Method to obtain inference parameters.
        """
        params = nn.ParameterList()
        if 'parameters' in dir(self.inference_model):
            params.extend(list(self.inference_model.parameters()))
        params.extend(list(self.latent.inference_parameters()))
        return params

    def generative_parameters(self):
        """
        Method to obtain generative parameters.
        """
        params = nn.ParameterList()
        if 'parameters' in dir(self.generative_model):
            params.extend(list(self.generative_model.parameters()))
        params.extend(list(self.latent.generative_parameters()))
        return params
