import os
import numpy as np
import cPickle as pickle
import torch
from time import strftime

global log_path


def init_log(log_root, train_config):
    # load/create log directory, format: day_month_year_hour_minutes_seconds
    global log_path

    if train_config['resume_experiment'] != '' and train_config['resume_experiment'] is not None:
        if os.path.exists(os.path.join(log_root, train_config['resume_experiment'])):
            log_path = os.path.join(log_root, train_config['resume_experiment'])
            return log_path, train_config['resume_experiment']
        else:
            raise Exception('Experiment folder ' + train_config['resume_experiment'] + ' not found.')

    log_dir = strftime("%b_%d_%Y_%H_%M_%S") + '/'
    log_path = os.path.join(log_root, log_dir)
    os.makedirs(log_path)
    os.system("rsync -au --include '*/' --include '*.py' --exclude '*' . " + log_path + "source")
    os.makedirs(os.path.join(log_path, 'metrics'))
    os.makedirs(os.path.join(log_path, 'visualizations'))
    os.makedirs(os.path.join(log_path, 'checkpoints'))
    return log_path, log_dir


def update_metric(file_name, value):
    if os.path.exists(file_name):
        metric = pickle.load(open(file_name, 'r'))
        metric.append(value)
        pickle.dump(metric, open(file_name, 'w'))
    else:
        pickle.dump([value], open(file_name, 'w'))


def log_train(func):
    """Wrapper to log train metrics."""
    global log_path

    def log_func(model, train_config, data, epoch, optimizers):
        output = func(model, train_config, data, optimizers)
        avg_elbo, avg_cond_log_like, avg_kl, avg_grad_mags = output
        update_metric(os.path.join(log_path, 'metrics', 'train_elbo.p'), (epoch, avg_elbo))
        update_metric(os.path.join(log_path, 'metrics', 'train_cond_log_like.p'), (epoch, avg_cond_log_like))
        for level in range(len(model.levels)):
            update_metric(os.path.join(log_path, 'metrics', 'train_kl_level_' + str(level) + '.p'), (epoch, avg_kl[level]))
        return output

    return log_func


def log_vis(func):
    """Wrapper to log metrics and visualizations."""
    global log_path

    def log_func(model, train_config, data_loader, epoch, vis=False, eval=False):
        output = func(model, train_config, data_loader, vis=vis, eval=eval)
        total_elbo, total_cond_log_like, total_kl, total_log_like, total_labels, total_cond_like, total_recon, total_posterior, total_prior, samples = output
        update_metric(os.path.join(log_path, 'metrics', 'val_elbo.p'), (epoch, np.mean(total_elbo[:, -1], axis=0)))
        update_metric(os.path.join(log_path, 'metrics', 'val_cond_log_like.p'), (epoch, np.mean(total_cond_log_like[:, -1], axis=0)))
        for level in range(len(model.levels)):
            update_metric(os.path.join(log_path, 'metrics', 'val_kl_level_' + str(level) + '.p'), (epoch, np.mean(total_kl[level][:, -1], axis=0)))

        if vis:
            epoch_path = os.path.join(log_path, 'visualizations', 'epoch_' + str(epoch))
            os.makedirs(epoch_path)

            batch_size = train_config['batch_size']
            n_iterations = train_config['n_iterations']
            data_shape = list(next(iter(data_loader))[0].size())[1:]

            pickle.dump(total_elbo[:batch_size], open(os.path.join(epoch_path, 'elbo.p'), 'w'))
            pickle.dump(total_cond_log_like[:batch_size], open(os.path.join(epoch_path, 'cond_log_like.p'), 'w'))
            for level in range(len(model.levels)):
                pickle.dump(total_kl[level][:batch_size], open(os.path.join(epoch_path, 'kl_level_' + str(level) + '.p'), 'w'))

            recon = total_recon[:batch_size, :].reshape([batch_size, n_iterations+1]+data_shape)
            pickle.dump(recon, open(os.path.join(epoch_path, 'reconstructions.p'), 'w'))

            samples = samples.reshape([batch_size]+data_shape)
            pickle.dump(samples, open(os.path.join(epoch_path, 'samples.p'), 'w'))

        if eval:
            pass
            # todo: save log likelihood estimate

        return output

    return log_func


def save_checkpoint(model, opt, epoch):
    global log_path
    torch.save(model, os.path.join(log_path, 'checkpoints', 'epoch_'+str(epoch)+'_model.ckpt'))
    torch.save(tuple(opt), os.path.join(log_path, 'checkpoints', 'epoch_'+str(epoch)+'_opt.ckpt'))


def get_last_epoch():
    global log_path
    last_epoch = 0
    for r, d, f in os.walk(os.path.join(log_path, 'checkpoints')):
        for ckpt_file_name in f:
            if ckpt_file_name[0] == 'e':
                epoch = int(ckpt_file_name.split('_')[1])
                if epoch > last_epoch:
                    last_epoch = epoch
    return last_epoch


def load_opt_checkpoint(epoch=-1):
    if epoch == -1:
        epoch = get_last_epoch()
    return torch.load(os.path.join(log_path, 'checkpoints', 'epoch_'+str(epoch)+'_opt.ckpt'))


def load_model_checkpoint(epoch=-1):
    if epoch == -1:
        epoch = get_last_epoch()
    return torch.load(os.path.join(log_path, 'checkpoints', 'epoch_'+str(epoch)+'_model.ckpt'))

