import torch
from torch.autograd import Variable
import numpy as np
from random import shuffle
from visualize import plot_images, plot_line, plot_metrics


# more general 'run' function (under construction)
def run(model, train_config, data, mode='val', optimizers=None, shuffle_data=True):
    pass

@plot_metrics
def train(model, train_config, data, mode='train', optimizers=None, shuffle_data=True):

    batch_size = train_config['batch_size']
    enc_opt, dec_opt = optimizers
    indices = np.arange(data.shape[0])
    if shuffle_data:
        shuffle(indices)

    data = data.reshape(-1, 3072)

    avg_elbo = avg_cond_log_like = avg_kl = 0

    n_batch = int(data.shape[0]/batch_size)
    for batch_num in range(n_batch):
        #print batch_num * 1.0 / n_batch
        data_index = batch_num*batch_size
        batch = np.copy(data[indices[data_index:data_index+batch_size]])
        batch = Variable(torch.from_numpy(batch))

        enc_opt.zero_grad()
        model.reset()
        model.decode()
        for i in range(train_config['n_iterations']-1):
            model.encode(batch)
            model.decode()
            elbo = model.ELBO(batch)
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
        elbo.backward(retain_variables=True)
        enc_opt.step()
        dec_opt.step()

    avg_elbo /= n_batch
    avg_cond_log_like /= n_batch
    avg_kl /= n_batch

    return avg_elbo[0], avg_cond_log_like[0], avg_kl[0]


# todo: add importance sampling x 5000 samples
@plot_metrics
def validate(model, train_config, data_labels, vis=True):

    batch_size = train_config['batch_size']
    data, labels = data_labels
    data.reshape(-1, 3072)

    avg_elbo = avg_cond_log_like = avg_kl = 0

    n_batch = int(data.shape[0] / batch_size)
    for batch_num in range(n_batch):
        data_index = batch_num*batch_size
        batch = np.copy(data[data_index:data_index+batch_size])
        batch = Variable(torch.from_numpy(batch))

        model.reset()
        model.decode()
        elbo=cond_log_like=kl=0
        for i in range(train_config['n_iterations']):
            model.encode(batch)
            model.decode()
            elbo, cond_log_like, kl = model.losses(batch)
        avg_elbo += elbo.data.cpu().numpy()[0]
        avg_cond_log_like += cond_log_like.data.cpu().numpy()[0]
        avg_kl += kl.data.cpu().numpy()[0]

    if vis:
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

    return avg_elbo, avg_cond_log_like, avg_kl
