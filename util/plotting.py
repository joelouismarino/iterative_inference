import visdom
import numpy as np

global vis


def initialize_env(env='main', port=8097):
    """Creates a visdom environment."""
    global vis
    vis = visdom.Visdom(port=port, env=env)
    return vis


def initialize_plots():
    handle_dict = {}

    nans = np.zeros((1, 2), dtype=float)
    nans.fill(np.nan)

    elbo_handle = plot_line(nans, np.ones((1, 2)), legend=['Train', 'Validation'], title='ELBO', xlabel='Epochs', ylabel='-ELBO (Nats)', xformat='log', yformat='log')
    handle_dict['elbo'] = elbo_handle

    cond_log_like_handle = plot_line(nans, np.ones((1, 2)), legend=['Train', 'Validation'], title='Conditional Log Likelihood', xlabel='Epochs', ylabel='-log P(x | z) (Nats)', xformat='log', yformat='log')
    handle_dict['cond_log_like'] = cond_log_like_handle

    kl_handle = plot_line(nans, np.ones((1, 2)), legend=['Train', 'Validation'], title='KL Divergence', xlabel='Epochs', ylabel='KL(q || p) (Nats)', xformat='log', yformat='log')
    handle_dict['kl'] = kl_handle

    return handle_dict


def save_env():
    """Saves the visdom environment."""
    global vis
    vis.save([vis.env])


def plot_images(imgs, caption=''):
    """Wraps visdom's image and images functions."""
    global vis
    if len(imgs.shape) == 3:
        imgs = np.expand_dims(imgs, axis=0)
    if imgs.shape[-1] == 3 or imgs.shape[-1] == 1:
        imgs = imgs.transpose((0, 3, 1, 2))
    opts = dict(caption=caption)
    win = vis.images(imgs, opts=opts)
    return win


def plot_line(Y, X=None, legend=None, win=None, title='', xlabel='', ylabel='', xformat='linear', yformat='linear'):
    """Wraps visdom's line function."""
    global vis
    opts = dict(title=title, xlabel=xlabel, ylabel=ylabel, legend=legend, xtype=xformat, ytype=yformat)
    if win is None:
        win = vis.line(Y, X, opts=opts)
    else:
        win = vis.line(Y, X, win=win, opts=opts, update='append')
    return win


def update_trace(Y, X, win, name):
    """Wraps visdom's updateTrace function."""
    global vis
    vis.updateTrace(X, Y, win=win, name=name)


def plot_metrics(func):
    """Wrapper around training/validation function to plot the outputs in corresponding visdom windows."""
    def plotting_func(model, train_config, data, epoch, handle_dict, train=False, optimizers=None, shuffle_data=True):

        output = func(model, train_config, data, optimizers=optimizers, shuffle_data=shuffle_data)

        if train:
            avg_elbo, avg_cond_log_like, avg_kl = output
            print avg_elbo, avg_cond_log_like, avg_kl
            update_trace(np.array([-avg_elbo]), np.array([epoch]).astype(int), win=handle_dict['elbo'], name='Train')
            update_trace(np.array([-avg_cond_log_like]), np.array([epoch]).astype(int), win=handle_dict['cond_log_like'], name='Train')
            update_trace(np.array([avg_kl]), np.array([epoch]).astype(int), win=handle_dict['kl'], name='Train')
        else:

            pass

        return output, handle_dict
    return plotting_func


def t_sne_plot():
    pass

