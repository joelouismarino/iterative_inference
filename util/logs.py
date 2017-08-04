import os
import numpy as np
import cPickle as pickle
from time import gmtime, strftime

global log_path


def init_log(log_root):
    global log_path
    # creates log directory, format: day_month_year_hour_minutes_seconds
    log_dir = strftime("%d_%b_%Y_%H_%M_%S", gmtime()) + '/'
    log_path = os.path.join(log_root + log_dir)
    os.makedirs(log_path)
    os.system("rsync -au --include '*/' --include '*.py' --exclude '*' . " + log_path + "source")
    os.makedirs(os.path.join(log_path, 'metrics'))
    os.makedirs(os.path.join(log_path, 'visualizations'))
    os.makedirs(os.path.join(log_path, 'model_checkpoints'))
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
        avg_elbo, avg_cond_log_like, avg_kl = output
        update_metric(os.path.join(log_path, 'metrics', 'train_elbo.p'), (epoch, avg_elbo))
        update_metric(os.path.join(log_path, 'metrics', 'train_cond_log_like.p'), (epoch, avg_cond_log_like))
        for level in range(len(model.levels)):
            update_metric(os.path.join(log_path, 'metrics', 'train_kl_level_' + str(level) + '.p'), (epoch, avg_kl[level]))
        return output

    return log_func


def log_vis(func):
    """Wrapper to log metrics and visualizations."""
    global log_path

    def log_func(model, train_config, data_loader, epoch, vis=True):
        output = func(model, train_config, data_loader, vis=vis)
        total_elbo, total_cond_log_like, total_kl, total_labels, total_recon, total_posterior, total_prior, samples = output
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

            recon = total_recon[:batch_size].reshape([batch_size, n_iterations+1]+data_shape)
            pickle.dump(recon, open(os.path.join(epoch_path, 'reconstructions.p'), 'w'))

            samples = samples.reshape([batch_size]+data_shape)
            pickle.dump(samples, open(os.path.join(epoch_path, 'samples.p'), 'w'))

        return output

    return log_func