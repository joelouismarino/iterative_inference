import torch
from torch.autograd import Variable
import numpy as np
from random import shuffle

from logs import log_train, log_vis
from plotting import plot_images, plot_line, plot_train, plot_model_vis


def train_on_batch(model, batch, n_iterations, optimizers):

    enc_opt, dec_opt = optimizers

    # initialize the model from the prior
    enc_opt.zero_grad()
    model.decode(generate=True)
    model.reset_state()

    # inference iterations
    for _ in range(n_iterations - 1):
        model.encode(batch)
        model.decode()
        elbo = model.elbo(batch, averaged=True)
        (-elbo).backward(retain_graph=True)

    # final iteration
    dec_opt.zero_grad()
    model.encode(batch)
    model.decode()

    elbo, cond_log_like, kl = model.losses(batch, averaged=True)
    (-elbo).backward()

    # divide encoder gradients
    for param in model.encoder_parameters():
        param.grad /= n_iterations

    # update parameters
    enc_opt.step()
    dec_opt.step()

    elbo = elbo.data.cpu().numpy()[0]
    cond_log_like = cond_log_like.data.cpu().numpy()[0]
    for level in range(len(kl)):
        kl[level] = kl[level].data.cpu().numpy()[0]

    return elbo, cond_log_like, kl


def run_on_batch(model, batch, n_iterations, vis=False):
    """Runs the model on a single batch. If visualizing, stores posteriors, priors, and output distributions."""

    batch_shape = list(batch.size())
    total_elbo = np.zeros((batch.size()[0], n_iterations+1))
    total_cond_log_like = np.zeros((batch.size()[0], n_iterations+1))
    total_kl = [np.zeros((batch.size()[0], n_iterations+1)) for _ in range(len(model.levels))]

    reconstructions = posterior = prior = None
    if vis:
        # store the reconstructions, posterior, and prior over iterations for the entire batch
        reconstructions = np.zeros([batch_shape[0], n_iterations+1] + batch_shape[1:])
        posterior = [np.zeros([batch_shape[0], n_iterations+1, 2, model.levels[level].n_latent]) for level in range(len(model.levels))]
        prior = [np.zeros([batch_shape[0], n_iterations+1, 2, model.levels[level].n_latent]) for level in range(len(model.levels))]

    # initialize the model from the prior
    model.decode(generate=True)
    model.reset_state()
    elbo, cond_log_like, kl = model.losses(batch)

    total_elbo[:, 0] = elbo.data.cpu().numpy()
    total_cond_log_like[:, 0] = cond_log_like.data.cpu().numpy()
    for level in range(len(kl)):
        total_kl[level][:, 0] = kl[level].data.cpu().numpy()

    if vis:
        reconstructions[:, 0] = model.output_dist.mean.data.cpu().numpy().reshape(batch_shape)
        for level in range(len(model.levels)):
            posterior[level][:, 0, 0, :] = model.levels[level].latent.posterior.mean.data.cpu().numpy()
            posterior[level][:, 0, 1, :] = model.levels[level].latent.posterior.log_var.data.cpu().numpy()
            prior[level][:, 0, 0, :] = model.levels[level].latent.prior.mean.data.cpu().numpy()
            prior[level][:, 0, 1, :] = model.levels[level].latent.prior.log_var.data.cpu().numpy()

    # inference iterations
    for i in range(1, n_iterations+1):
        model.encode(batch)
        model.decode()
        elbo, cond_log_like, kl = model.losses(batch)
        total_elbo[:, i] = elbo.data.cpu().numpy()
        total_cond_log_like[:, i] = cond_log_like.data.cpu().numpy()
        for level in range(len(kl)):
            total_kl[level][:, i] = kl[level].data.cpu().numpy()
        if vis:
            reconstructions[:, i] = model.output_dist.mean.data.cpu().numpy().reshape(batch_shape)
            for level in range(len(model.levels)):
                posterior[level][:, i, 0, :] = model.levels[level].latent.posterior.mean.data.cpu().numpy()
                posterior[level][:, i, 1, :] = model.levels[level].latent.posterior.log_var.data.cpu().numpy()
                prior[level][:, i, 0, :] = model.levels[level].latent.prior.mean.data.cpu().numpy()
                prior[level][:, i, 1, :] = model.levels[level].latent.prior.log_var.data.cpu().numpy()

    return total_elbo, total_cond_log_like, total_kl, reconstructions, posterior, prior


def eval_on_batch(model, batch, n_importance_samples):
    """Estimates marginal log likelihood of data using importance sampling."""
    importance_sample_estimate = np.zeros((batch.size()[0], n_importance_samples))

    for i in range(n_importance_samples):
        model.decode()
        elbo, cond_log_like, kl = model.losses(batch)
        importance_sample_estimate[:, i] = torch.exp(cond_log_like).data.cpu().numpy()
        for level in range(len(model.levels)):
            importance_weight = torch.exp(-model.levels[level].kl_divergence().sum(dim=1))
            importance_sample_estimate[:, i:i + 1] *= importance_weight.data.cpu().numpy().reshape((-1, 1))

    return np.log(np.mean(importance_sample_estimate, axis=1))


@plot_model_vis
@log_vis
def run(model, train_config, data_loader, vis=False, eval=False):
    """Runs the model on a set of data."""

    batch_size = train_config['batch_size']
    n_iterations = train_config['n_iterations']
    n_examples = batch_size * len(iter(data_loader))
    data_shape = list(next(iter(data_loader))[0].size())[1:]

    total_elbo = np.zeros((n_examples, n_iterations+1))
    total_cond_log_like = np.zeros((n_examples, n_iterations+1))
    total_kl = [np.zeros((n_examples, n_iterations+1)) for _ in range(len(model.levels))]
    total_log_like = np.zeros(n_examples) if eval else None
    total_labels = np.zeros(n_examples)
    total_recon = total_posterior = total_prior = None
    if vis:
        total_recon = np.zeros([n_examples, n_iterations + 1] + data_shape)
        total_posterior = [np.zeros([n_examples, n_iterations + 1, 2, model.levels[level].n_latent]) for level in range(len(model.levels))]
        total_prior = [np.zeros([n_examples, n_iterations + 1, 2, model.levels[level].n_latent]) for level in range(len(model.levels))]

    for batch_index, (batch, labels) in enumerate(data_loader):
        batch = Variable(batch)
        if train_config['cuda_device'] is not None:
            batch = batch.cuda(train_config['cuda_device'])

        if model.output_distribution == 'bernoulli':
            batch = torch.bernoulli(batch / 255.)

        elbo, cond_log_like, kl, recon, posterior, prior = run_on_batch(model, batch, n_iterations, vis)

        data_index = batch_index * batch_size
        total_elbo[data_index:data_index + batch_size, :] = elbo
        total_cond_log_like[data_index:data_index + batch_size, :] = cond_log_like
        for level in range(len(model.levels)):
            total_kl[level][data_index:data_index + batch_size, :] = kl[level]

        total_labels[data_index:data_index + batch_size] = labels.numpy()

        if vis:
            total_recon[data_index:data_index + batch_size] = recon
            for level in range(len(model.levels)):
                total_posterior[level][data_index:data_index + batch_size] = posterior[level]
                total_prior[level][data_index:data_index + batch_size] = prior[level]

        if eval:
            total_log_like[data_index:data_index + batch_size] = eval_on_batch(model, batch, 5000)

    samples = None
    if vis:
        # visualize samples from the model
        samples = model.decode(generate=True).mean.data.cpu().numpy().reshape([batch_size]+data_shape)

    return total_elbo, total_cond_log_like, total_kl, total_log_like, total_labels, total_recon, total_posterior, total_prior, samples


@plot_train
@log_train
def train(model, train_config, data_loader, optimizers):

    avg_elbo = []
    avg_cond_log_like = []
    avg_kl = [[] for _ in range(len(model.levels))]

    for batch, _ in data_loader:
        if train_config['cuda_device'] is not None:
            batch = Variable(batch.cuda(train_config['cuda_device']))
        else:
            batch = Variable(batch)

        if model.output_distribution == 'bernoulli':
            batch = torch.bernoulli(batch / 255.)

        elbo, cond_log_like, kl = train_on_batch(model, batch, train_config['n_iterations'], optimizers)

        avg_elbo.append(elbo)
        avg_cond_log_like.append(cond_log_like)
        for l in range(len(avg_kl)):
            avg_kl[l].append(kl[l])

    if np.isnan(np.sum(avg_elbo)):
        raise Exception('Nan encountered during training.')

    return np.mean(avg_elbo), np.mean(avg_cond_log_like), [np.mean(avg_kl[l]) for l in range(len(model.levels))]

