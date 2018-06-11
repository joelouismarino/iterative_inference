import torch.nn as nn
from lib.distributions import Normal


class LatentVariable(nn.Module):
    """
    Abstract class for a latent variable.
    """
    def __init__(self, latent_config):
        super(LatentVariable, self).__init__()
        self.approx_post = self.prior = None
        self.latent_config = latent_config
        self.inference_procedure = None

    def infer(self, input):
        """
        Abstract method to perform inference.

        Args:
            input (Tensor): input to the inference procedure
        """
        raise NotImplementedError

    def generate(self, input, gen, n_samples=1):
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
        Abstract method to reinitialize the approximate posterior and prior over
        the variable.
        """
        raise NotImplementedError

    def re_init_approx_posterior(self):
        """
        Abstract method to reinitialize the approximate posterior from the prior.
        """
        raise NotImplementedError

    def kl_divergence(self, analytical=False):
        """
        Method to compute KL divergence between the approximate posterior and
        prior for this variable.

        Args:
            analytical (boolean): whether to use the analytical form of the KL
                                  divergence for exact evaluation
        """
        if analytical:
            # analytical KL divergence currently only defined for Gaussians
            assert type(self.approx_post) == type(self.prior) == Normal
            post_mean = self.approx_post.mean
            post_log_var = self.approx_post.log_var
            if len(post_mean.data.shape) in [2, 4]:
                post_mean = post_mean.unsqueeze(1)
                post_log_var = post_log_var.unsqueeze(1)
            var_ratio = post_log_var.exp() / (self.prior.log_var.exp())
            t1 = (post_mean - self.prior.mean).pow(2) / (self.prior.log_var.exp())
            return 0.5 * (var_ratio + t1 - 1 - (var_ratio + 1e-7).log())
        else:
            # numerically estimated KL divergence
            z = self.approx_post.sample()
            return self.approx_post.log_prob(z) - self.prior.log_prob(z)

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

    def approx_posterior_parameters(self):
        """
        Abstract method to obtain approximate posterior parameters.
        """
        raise NotImplementedError

    def approx_posterior_gradients(self):
        """
        Abstract method to obtain approximate posterior gradients.
        """
        raise NotImplementedError
