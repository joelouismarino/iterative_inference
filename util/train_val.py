import torch
from torch.autograd import Variable
import numpy as np
from random import shuffle
from plotting import plot_images, plot_line, plot_train, plot_model_vis

import time
import matplotlib.pyplot as plt


def train_on_batch(model, batch, n_iterations, optimizers):

    enc_opt, dec_opt = optimizers

    # initialize the model
    enc_opt.zero_grad()
    model.reset()
    model.decode()

    #ave_grad_time = 0.

    # inference iterations
    for _ in range(n_iterations - 1):
        model.encode(batch)
        model.decode()
        elbo = model.elbo(batch, averaged=True)
        (-elbo).backward(retain_variables=True)

    # final iteration
    dec_opt.zero_grad()
    model.encode(batch)
    model.decode()

    elbo, cond_log_like, kl = model.losses(batch, averaged=True)
    #start = time.time()
    (-elbo).backward()
    #ave_grad_time = time.time() - start

    enc_opt.step()
    dec_opt.step()

    elbo = elbo.data.cpu().numpy()[0]
    cond_log_like = cond_log_like.data.cpu().numpy()[0]
    for level in range(len(kl)):
        kl[level] = kl[level].data.cpu().numpy()[0]

    #print ave_grad_time

    return elbo, cond_log_like, kl


# todo: add importance sampling x 5000 samples
def run_on_batch(model, batch, n_iterations, vis=False):
    """Runs the model on a single batch."""

    total_elbo = np.zeros((batch.shape[0], n_iterations+1))
    total_cond_log_like = np.zeros((batch.shape[0], n_iterations+1))
    total_kl = [np.zeros((batch.shape[0], n_iterations+1)) for _ in range(len(model.levels))]

    reconstructions = posterior = prior = None
    if vis:
        # store the reconstructions, posterior, and prior over iterations for the entire batch
        reconstructions = np.zeros([batch.shape[0], n_iterations+1] + batch.shape[1:])
        posterior = [np.zeros([batch.shape[0], n_iterations+1, 2, model.levels[level].n_latent]) for level in range(len(model.levels))]
        prior = [np.zeros([batch.shape[0], n_iterations+1, 2, model.levels[level].n_latent]) for level in range(len(model.levels))]

    # initialize the model
    model.reset()
    model.decode()
    elbo, cond_log_like, kl = model.losses(batch)
    total_elbo[:, 0] = elbo.data.cpu().numpy()[0]
    total_cond_log_like[:, 0] = cond_log_like.data.cpu().numpy()[0]
    for level in range(len(kl)):
        total_kl[level, :, 0] = kl[level].data.cpu().numpy()[0]

    if vis:
        reconstructions[:, 0] = model.output_dist.mean.data.cpu().numpy()[0]
        for level in range(len(model.levels)):
            posterior[level, :, 0, 0, :] = model.levels[level].latent.posterior.mean.data.cpu().numpy()[0]
            posterior[level, :, 0, 1, :] = model.levels[level].latent.posterior.log_var.data.cpu().numpy()[0]
            prior[level, :, 0, 0, :] = model.levels[level].latent.prior.mean.data.cpu().numpy()[0]
            prior[level, :, 0, 1, :] = model.levels[level].latent.prior.log_var.data.cpu().numpy()[0]

    # inference iterations
    for i in range(1, n_iterations+1):
        model.encode(batch)
        model.decode()
        elbo, cond_log_like, kl = model.losses(batch)
        total_elbo[:, i] = elbo.data.cpu().numpy()[0]
        total_cond_log_like[:, i] = cond_log_like.data.cpu().numpy()[0]
        for level in range(len(kl)):
            total_kl[level, :, i] = kl[level].data.cpu().numpy()[0]

        if vis:
            reconstructions[:, i] = model.output_dist.mean.data.cpu().numpy()[0]
            for level in range(len(model.levels)):
                posterior[level, :, i, 0, :] = model.levels[level].latent.posterior.mean.data.cpu().numpy()[0]
                posterior[level, :, i, 1, :] = model.levels[level].latent.posterior.log_var.data.cpu().numpy()[0]
                prior[level, :, i, 0, :] = model.levels[level].latent.prior.mean.data.cpu().numpy()[0]
                prior[level, :, i, 1, :] = model.levels[level].latent.prior.log_var.data.cpu().numpy()[0]

    return total_elbo, total_cond_log_like, total_kl, reconstructions, posterior, prior


@plot_model_vis
def run(model, train_config, data, vis=False):
    """Runs the model on a set of data."""

    batch_size = train_config['batch_size']
    n_examples = data.shape[0]

    total_elbo = np.zeros((n_examples, n_iterations))
    total_cond_log_like = np.zeros((n_examples, n_iterations))
    total_kl = [np.zeros((n_examples, n_iterations)) for _ in range(len(model.levels))]

    n_batch = int(n_examples / batch_size)
    for batch_num in range(n_batch):
        # get data batch
        data_index = batch_num * batch_size
        batch = np.copy(data[data_index:data_index + batch_size])
        batch = Variable(torch.from_numpy(batch))

        elbo, cond_log_like, kl, _, _, _ = run_on_batch(model, batch, train_config['n_iterations'])

        total_elbo[data_index:data_index + batch_size, :] = elbo
        total_cond_log_like[data_index:data_index + batch_size, :] = cond_log_like
        for level in range(len(kl)):
            total_kl[level, data_index:data_index + batch_size, :] = kl[level]

    if vis:

        # visualize the reconstruction of the first batch
        first_batch = np.copy(data[:batch_size])
        first_batch = Variable(torch.from_numpy(first_batch))
        elbo, cond_log_like, kl, reconstructions, posterior, prior = run_on_batch(model, first_batch, train_config['n_iterations'], vis)

        # visualize samples from the model
        sample_output = model.decode(generate=True).mean.data.cpu().numpy()[0]

    return total_elbo, total_cond_log_like, total_kl

"""
@plot_train
def train(model, train_config, data, optimizers, shuffle_data=True):


    batch_size = train_config['batch_size']
    n_examples = data.shape[0]

    indices = np.arange(n_examples)
    if shuffle_data:
        shuffle(indices)

    ave_time = []

    n_batch = int(n_examples / batch_size)
    for batch_num in range(n_batch):
        # get data batch
        data_index = batch_num * batch_size
        batch = np.copy(data[indices[data_index:data_index + batch_size]])
        batch = Variable(torch.from_numpy(batch))

        if model.output_distribution == 'bernoulli':
            batch = torch.bernoulli(batch / 255.)

        tic = time.time()
        train_on_batch(model, batch, train_config['n_iterations'], optimizers)
        toc = time.time()
        ave_time.append(toc - tic)

    plt.plot(ave_time)
    plt.show()
    print ave_time

    #print avg_elbo, avg_cond_log_like, avg_kl
    #return avg_elbo, avg_cond_log_like, avg_kl

"""
@plot_train
def train(model, train_config, data, optimizers, shuffle_data=True):

    batch_size = train_config['batch_size']
    n_examples = data.shape[0]

    avg_elbo = 0.
    avg_cond_log_like = 0.
    avg_kl = [0. for _ in range(len(model.levels))]

    indices = np.arange(n_examples)
    if shuffle_data:
        shuffle(indices)

    ave_time = []

    n_batch = int(n_examples / batch_size)
    for batch_num in range(n_batch):
        # get data batch
        data_index = batch_num * batch_size
        batch = np.copy(data[indices[data_index:data_index + batch_size]])
        batch = Variable(torch.from_numpy(batch))

        if model.output_distribution == 'bernoulli':
            batch = torch.bernoulli(batch / 255.)

        tic = time.time()
        elbo, cond_log_like, kl = train_on_batch(model, batch, train_config['n_iterations'], optimizers)
        toc = time.time()
        ave_time.append(toc - tic)

        avg_elbo += elbo[0]
        avg_cond_log_like += cond_log_like[0]
        for i in range(len(avg_kl)):
            avg_kl[i] += kl[i][0]

    avg_elbo /= n_batch
    avg_cond_log_like /= n_batch
    for i in range(len(avg_kl)):
        avg_kl[i] /= n_batch

    plt.plot(ave_time)
    plt.show()
    #print ave_time

    print avg_elbo, avg_cond_log_like, avg_kl
    return avg_elbo, avg_cond_log_like, avg_kl

