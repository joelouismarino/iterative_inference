import torch
import torch.nn as nn
from latent_level import LatentLevel
from lib.modules.networks import FullyConnectedNetwork
from lib.modules.latent_variables import FullyConnectedLatentVariable
from lib.modules.misc import LayerNorm, BatchNorm


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

    def _get_encoding_form(self, input, in_out):
        """
        Gets the appropriate input form for the inference procedure.

        Args:
            input (Tensor): observation and/or lower level error if 'in',
                            observation if 'out'
            in_out (str): 'in' or 'out', specifies whether the encoding is for
                          the input of the inference procedure or the output of
                          the latent level
        """
        encoding = []
        if 'observation' in self.inference_procedure:
            encoding.append(input)
        if 'gradient' in self.inference_procedure:
            if in_out == 'in':
                grads = self.latent.approx_posterior_gradients()
                # grads = torch.cat([LayerNorm()(grad) for grad in grads], dim=1)
                grads = torch.cat([BatchNorm()(grad) for grad in grads], dim=1)
                encoding.append(grads)
                params = self.latent.approx_posterior_parameters()
                # params = torch.cat([LayerNorm()(param) for param in params], dim=1)
                params = torch.cat(params, dim=1)
                encoding.append(params)
        if 'error' in self.inference_procedure:
            error = self.latent.error()
            error = LayerNorm()(error)
            encoding.append(error)
            if in_out == 'in':
                if 'observation' not in self.inference_procedure:
                    encoding.append(input)
                params = self.latent.approx_posterior_parameters()
                params = torch.cat([LayerNorm()(param) for param in params], dim=1)
                encoding.append(params)

        if len(encoding) == 0:
            return None
        elif len(encoding) == 1:
            return encoding[0]
        else:
            return torch.cat(encoding, dim=1)

    def infer(self, input):
        """
        Method to perform inference.

        Args:
            input (Tensor): input to the inference procedure
        """
        input = self._get_encoding_form(input, 'in')
        input = self.inference_model(input)
        latent_sample = self.latent.infer(input)
        return self._get_encoding_form(latent_sample, 'out')

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

    def re_init(self, batch_size):
        """
        Method to reinitialize the latent level (latent variable and any state
        variables in the generative / inference procedures).
        """
        self.latent.re_init(batch_size)
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
