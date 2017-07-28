import torch
from torch.autograd import Variable
import numpy as np
from random import shuffle
from plotting import plot_images, plot_line, plot_train, plot_model_vis


def train_on_batch(model, batch, n_iterations, optimizers):

    enc_opt, dec_opt = optimizers

    # initialize the model
    enc_opt.zero_grad()
    model.reset()
    model.decode()

    # inference iterations
    for i in range(n_iterations - 1):
        model.encode(batch)
        model.decode()
        elbo = model.elbo(batch, averaged=True)
        (-elbo).backward(retain_variables=True)

    # final iteration
    dec_opt.zero_grad()
    model.encode(batch)
    model.decode()

    elbo, cond_log_like, kl = model.losses(batch, averaged=True)
    (-elbo).backward(retain_variables=True)

    enc_opt.step()
    dec_opt.step()

    elbo = elbo.data.cpu().numpy()[0]
    cond_log_like = cond_log_like.data.cpu().numpy()[0]
    for i in range(len(kl)):
        kl[i] = kl[i].data.cpu().numpy()[0]

    return elbo, cond_log_like, kl


# todo: add importance sampling x 5000 samples
# todo: add functionality to capture latent state, etc.
def run_on_batch(model, batch, n_iterations):
    """Runs the model on a single batch."""

    total_elbo = np.zeros((n_iterations, batch.shape[0]))
    total_cond_log_like = np.zeros((n_iterations, batch.shape[0]))
    total_kl = np.zeros((n_iterations, batch.shape[0]))

    # initialize the model
    model.reset()
    model.decode()

    # inference iterations
    for i in range(n_iterations):
        model.encode(batch)
        model.decode()
        elbo, cond_log_like, kl = model.losses(batch)
        total_elbo[i] = elbo.data.cpu().numpy()[0]
        total_cond_log_like[i] = cond_log_like.data.cpu().numpy()[0]
        total_kl[i] = kl.data.cpu().numpy()[0]

    return total_elbo, total_cond_log_like, total_kl


@plot_model_vis
def run(model, train_config, data, vis=False):
    """Runs the model on a set of data."""

    batch_size = train_config['batch_size']
    n_examples = data.shape[0]

    total_elbo = np.zeros((n_iterations, n_examples))
    total_cond_log_like = np.zeros((n_iterations, n_examples))
    total_kl = np.zeros((n_iterations, n_examples))

    indices = np.arange(n_examples)
    if shuffle_data:
        shuffle(indices)

    n_batch = int(n_examples / batch_size)
    for batch_num in range(n_batch):
        # get data batch
        data_index = batch_num * batch_size
        batch = np.copy(data[indices[data_index:data_index + batch_size]])
        batch = Variable(torch.from_numpy(batch))

        elbo, cond_log_like, kl = run_on_batch(model, batch, train_config['n_iterations'])

        total_elbo[:, indices[data_index:data_index+batch_size]] = elbo
        total_cond_log_like[:, indices[data_index:data_index + batch_size]] = cond_log_like
        total_kl[:, indices[data_index:data_index + batch_size]] = kl

    """
    visualization code:
    rand_batch_num = np.random.randint(n_batch)
    rand_batch = np.copy(data[rand_batch_num:rand_batch_num+batch_size])
    plot_images(rand_batch.reshape(-1, 32, 32, 3), caption='Random Batch')
    rand_batch = Variable(torch.from_numpy(rand_batch))
    model.reset()
    output_dist = model.decode()
    plot_images(output_dist.mean.data.cpu().numpy().reshape(-1, 32, 32, 3), caption='Iteration: 0')
    for i in range(train_config['n_iterations']):
        model.encode(rand_batch)
        output_dist = model.decode()
        plot_images(output_dist.mean.data.cpu().numpy().reshape(-1, 32, 32, 3), caption='Iteration: '+str(i+1))
        elbo, cond_log_like, kl = model.losses(batch)

    output_dist = model.decode(generate=True)
    plot_images(output_dist.mean.data.cpu().numpy().reshape(-1, 32, 32, 3), caption='Generation')
    """

    return total_elbo, total_cond_log_like, total_kl


@plot_train
def train(model, train_config, data, optimizers, shuffle_data=True):
    """Trains the model on set of data using optimizers."""

    batch_size = train_config['batch_size']
    n_examples = data.shape[0]

    avg_elbo = 0.
    avg_cond_log_like = 0.
    avg_kl = [0. for _ in range(len(model.levels))]

    indices = np.arange(n_examples)
    if shuffle_data:
        shuffle(indices)

    n_batch = int(n_examples / batch_size)
    for batch_num in range(n_batch):
        # get data batch
        data_index = batch_num * batch_size
        batch = np.copy(data[indices[data_index:data_index + batch_size]])
        batch = Variable(torch.from_numpy(batch))
        elbo, cond_log_like, kl = train_on_batch(model, batch, train_config['n_iterations'], optimizers)
        avg_elbo += elbo[0]
        avg_cond_log_like += cond_log_like[0]
        for i in range(len(avg_kl)):
            avg_kl[i] += kl[i][0]

    avg_elbo /= n_batch
    avg_cond_log_like /= n_batch
    for i in range(len(avg_kl)):
        avg_kl[i] /= n_batch

    print avg_elbo, avg_cond_log_like, avg_kl
    return avg_elbo, avg_cond_log_like, avg_kl


