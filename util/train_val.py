import torch
from torch.autograd import Variable
import numpy as np
from random import shuffle

from visualize import plot_image, plot_images

# todo: average ELBO (and other losses?) to display per epoch
# todo: log the losses, or should this be in main?


def train(model, train_config, data, optimizers, shuffle_data=True):

    batch_size = train_config['batch_size']
    enc_opt, dec_opt = optimizers
    indices = np.arange(data.shape[0])
    if shuffle_data:
        shuffle(indices)

    n_batch = int(data.shape[0]/batch_size)
    for batch_num in range(n_batch):
        data_index = batch_num*batch_size
        batch = np.copy(data[indices[data_index:data_index+batch_size]])
        #plot_images(batch.reshape(-1, 32, 32, 3))
        plot_images(batch[0:1].reshape(-1, 32, 32, 3))
        batch = Variable(torch.from_numpy(batch))

        enc_opt.zero_grad()
        model.reset()
        model.decode()
        for i in range(train_config['n_iterations']-1):
            model.encode(batch)
            model.decode()
            elbo = -model.ELBO(batch)
            elbo.backward(retain_variables=True)
        dec_opt.zero_grad()
        model.encode(batch)
        model.decode()
        elbo = -model.ELBO(batch)
        elbo.backward(retain_variables=True)
        enc_opt.step()
        dec_opt.step()


def validate(model, data_labels):
    pass
