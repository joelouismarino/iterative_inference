import torch
from torch.autograd import Variable
import numpy as np
from random import shuffle

from visualize import plot_images, plot_line

# todo: log the losses, or should this be in main?


def train(model, train_config, data, optimizers, shuffle_data=True):

    batch_size = train_config['batch_size']
    enc_opt, dec_opt = optimizers
    indices = np.arange(data.shape[0])
    if shuffle_data:
        shuffle(indices)

    avg_elbo = avg_cond_log_like = avg_kl = 0

    n_batch = int(data.shape[0]/batch_size)
    for batch_num in range(n_batch):
        data_index = batch_num*batch_size
        batch = np.copy(data[indices[data_index:data_index+batch_size]])
        batch = Variable(torch.from_numpy(batch))

        enc_opt.zero_grad()
        model.reset()
        model.decode()
        for i in range(train_config['n_iterations']-1):
            model.encode(batch)
            model.decode()
            elbo = model.EBLO(batch)
            elbo = -elbo
            elbo.backward(retain_variables=True)
        dec_opt.zero_grad()
        model.encode(batch)
        model.decode()
        elbo, cond_log_like, kl = model.losses(batch)
        avg_elbo += elbo.data.cpu().numpy()[0]
        avg_cond_log_like += cond_log_like.data.cpu().numpy()[0]
        avg_kl += kl.data.cpu().numpy()[0]
        elbo = -elbo
        #elbo_plot = plot_line(elbo.data.cpu().numpy()[0], np.array([batch_num+1]), win=elbo_plot, xformat='log', yformat='log')
        elbo.backward(retain_variables=True)
        enc_opt.step()
        dec_opt.step()

    avg_elbo /= n_batch
    avg_cond_log_like /= n_batch
    avg_kl /= n_batch
    
    return avg_elbo, avg_cond_log_like, avg_kl


# todo: add importance sampling x 5000 examples
def validate(model, train_config, data_labels, vis=True):

    batch_size = train_config['batch_size']
    data, labels = data_labels

    avg_elbo = avg_cond_log_like = avg_kl = 0

    n_batch = int(data.shape[0] / batch_size)
    for batch_num in range(n_batch):
        data_index = batch_num*batch_size
        batch = np.copy(data[data_index:data_index+batch_size])
        batch = Variable(torch.from_numpy(batch))

        model.reset()
        model.decode()
        for i in range(train_config['n_iterations']-1):
            model.encode(batch)
            model.decode()
            elbo, cond_log_like, kl = model.losses(batch)
        model.encode(batch)
        model.decode()
        elbo, cond_log_like, kl = model.losses(batch)
        avg_elbo += elbo.data.cpu().numpy()[0]
        avg_cond_log_like += cond_log_like.data.cpu().numpy()[0]
        avg_kl += kl.data.cpu().numpy()[0]

    if vis:
        pass

    return avg_elbo, avg_cond_log_like, avg_kl
