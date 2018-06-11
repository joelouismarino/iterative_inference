import torch.nn as nn
from lib.distributions import Normal


class Model(nn.Module):
    """
    Abstract class for a latent variable model.
    """
    def __init__(self, model_config):
        super(Model, self).__init__()
        self.model_config = model_config
        self.output_interval = None

    def _construct(self, model_config):
        """
        Abstract method for constructing a latent variable model using
        the model configuration file.

        Args:
            model_config (dict): dictionary containing model configuration params
        """
        raise NotImplementedError

    def infer(self, observation):
        """
        Abstract method for perfoming inference of the approximate posterior
        over the latent variables.

        Args:
            observation (tensor): observation to infer latent variables from
        """
        raise NotImplementedError

    def generate(self, gen=False, n_samples=1):
        """
        Abstract method for generating observations, i.e. running the generative
        model forward.

        Args:
            gen (boolean): whether to sample from prior or approximate posterior
            n_samples (int): number of samples to draw and evaluate
        """
        raise NotImplementedError

    def re_init(self):
        """
        Abstract method for reinitializing the state (approximate posterior and
        priors) of the latent variable model.
        """
        raise NotImplementedError

    def kl_divergences(self, averaged=True):
        """
        Estimate the KL divergence (at each latent level).

        Args:
            averaged (boolean): whether to average over the batch dimension
        """
        kl = []
        for level_ind, latent_level in enumerate(self.latent_levels):
            analytical = (level_ind == len(self.latent_levels)-1)
            level_kl = latent_level.latent.kl_divergence(analytical)
            for dim in range(2, len(level_kl.data.shape)):
                level_kl = level_kl.sum(dim) # sum over data dimensions
            level_kl = level_kl.mean(1) # average over sample dimension
            kl.append(level_kl)
        if averaged:
            kl = [level_kl.mean(dim=0) for level_kl in kl] # average over batch dimension
        return kl

    def conditional_log_likelihoods(self, observation, averaged=True):
        """
        Estimate the conditional log-likelihood.

        Args:
            observation (tensor): observation to evaluate
            averaged (boolean): whether to average over the batch dimension
        """
        # add sample dimension
        if len(observation.data.shape) in [2, 4]:
            observation = observation.unsqueeze(1)

        if self.output_interval is not None:
            observation = (observation, observation + self.output_interval)

        log_prob = self.output_dist.log_prob(value=observation)

        while len(log_prob.data.shape) > 2:
            # sum over all of the spatial dimensions
            last_dim = len(log_prob.data.shape) - 1
            log_prob = log_prob.sum(last_dim)

        # average over sample dimension
        log_prob = log_prob.mean(1)

        if averaged:
            # average over batch dimension
            log_prob = log_prob.mean(dim=0)

        return log_prob

    def elbo(self, observation, averaged=True, anneal_weight=1.):
        """
        Estimate the evidence lower bound (ELBO).

        Args:
            observation (tensor): observation to evaluate
            averaged (boolean): whether to average over the batch dimension
        """
        cond_log_like = self.conditional_log_likelihoods(observation, averaged=False)
        kl = sum(self.kl_divergences(averaged=False))
        elbo = cond_log_like - anneal_weight * kl
        if averaged:
            return elbo.mean(dim=0)
        else:
            return elbo

    def losses(self, observation, averaged=True, anneal_weight=1.):
        """
        Estimate all losses.

        Args:
            observation (tensor): observation to evaluate
            averaged (boolean): whether to average over the batch dimension
        """
        cond_log_like = self.conditional_log_likelihoods(observation, averaged=False)
        kl = self.kl_divergences(averaged=False)
        elbo = cond_log_like - anneal_weight * sum(kl)
        if averaged:
            return elbo.mean(dim=0), cond_log_like.mean(dim=0), [level_kl.mean(dim=0) for level_kl in kl]
        else:
            return elbo, cond_log_like, kl

    def inference_parameters(self):
        """
        Abstract method for obtaining the inference parameters.
        """
        raise NotImplementedError

    def generative_parameters(self):
        """
        Abstract method for obtaining the generative parameters.
        """
        raise NotImplementedError
