import visdom
import numpy as np

global vis


def initialize_env(env='main', port=8097):
    """Creates a visdom environment."""
    global vis
    vis = visdom.Visdom(port=port, env=env)
    return vis


def initialize_plots(train_config, arch):
    nans = np.zeros((1, 2), dtype=float)
    nans.fill(np.nan)

    # plots for average metrics on train and validation set
    kl_legend = []
    for mode in ['Train', 'Validation']:
        for level in range(len(arch['n_latent'])):
            kl_legend.append(mode + ', Level ' + str(level))
    kl_nans = np.zeros((1, 2 * len(arch['n_latent'])))
    kl_nans.fill(np.nan)
    elbo_handle = plot_line(nans, np.ones((1, 2)), legend=['Train', 'Validation'], title='ELBO', xlabel='Epochs', ylabel='-ELBO (Nats)', xformat='log', yformat='log')
    cond_log_like_handle = plot_line(nans, np.ones((1, 2)), legend=['Train', 'Validation'], title='Conditional Log Likelihood', xlabel='Epochs', ylabel='-log P(x | z) (Nats)', xformat='log', yformat='log')
    kl_handle = plot_line(kl_nans, np.ones((1, 2 * len(arch['n_latent']))), legend=kl_legend, title='KL Divergence', xlabel='Epochs', ylabel='KL(q || p) (Nats)', xformat='log', yformat='log')

    # plot of average improvement over iterations on validation set
    kl_legend = []

    for level in range(len(arch['n_latent'])):
        kl_legend.append('Level ' + str(level))

    kl_nans = None
    indices = None
    if len(arch['n_latent']) > 1:
        kl_nans = np.zeros((1, len(arch['n_latent'])))
        kl_nans.fill(np.nan)
        indices = np.ones((1, len(arch['n_latent'])))
    else:
        kl_nans = np.array(np.nan).reshape(1)
        indices = np.ones(1)

    elbo_improvement_handle = plot_line(np.array(np.nan).reshape(1), np.ones(1), legend=['ELBO'], title='Ave. Improvement in ELBO Over Inference Iterations', xlabel='Epochs', ylabel='Relative Improvement (%)', xformat='log')
    recon_improvement_handle = plot_line(np.array(np.nan).reshape(1), np.ones(1), legend=['log P(x | z)'], title='Ave. Improvement in Reconstruction Over Inference Iterations', xlabel='Epochs', ylabel='Relative Improvement (%)', xformat='log')
    kl_improvement_handle = plot_line(kl_nans, indices, legend=kl_legend, title='Ave. Improvement in KL Divergence Over Inference Iterations', xlabel='Epochs', ylabel='Relative Improvement (%)', xformat='log')

    return dict(elbo=elbo_handle, cond_log_like=cond_log_like_handle, kl=kl_handle, elbo_improvement=elbo_improvement_handle,
                recon_improvement=recon_improvement_handle, kl_improvement=kl_improvement_handle)


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


def plot_train(func):
    """Wrapper around training function to plot the outputs in corresponding visdom windows."""
    def plotting_func(model, train_config, data, epoch, handle_dict, optimizers, shuffle_data=True):
        output = func(model, train_config, data, optimizers, shuffle_data=shuffle_data)
        avg_elbo, avg_cond_log_like, avg_kl = output
        update_trace(np.array([-avg_elbo]), np.array([epoch]).astype(int), win=handle_dict['elbo'], name='Train')
        update_trace(np.array([-avg_cond_log_like]), np.array([epoch]).astype(int), win=handle_dict['cond_log_like'], name='Train')
        for level in range(len(model.levels)):
            update_trace(np.array([avg_kl[level]]), np.array([epoch]).astype(int), win=handle_dict['kl'], name='Train, Level ' + str(level))
        return output, handle_dict
    return plotting_func


def plot_model_vis(func):
    """Wrapper around run function to plot the outputs in corresponding visdom windows."""
    def plotting_func(model, train_config, data, epoch, handle_dict, vis=True):
        output = func(model, train_config, data, vis=vis)
        avg_elbo, avg_cond_log_like, avg_kl, reconstructions, samples = output
        #update_trace(np.array([-avg_elbo]), np.array([epoch]).astype(int), win=handle_dict['elbo'], name='Validation')
        #update_trace(np.array([-avg_cond_log_like]), np.array([epoch]).astype(int), win=handle_dict['cond_log_like'], name='Validation')
        #update_trace(np.array([avg_kl]), np.array([epoch]).astype(int), win=handle_dict['kl'], name='Validation')
        if vis:
            data_shape = data.shape[1:]
            print reconstructions.shape
            plot_images(reconstructions[:, 1].reshape([train_config['batch_size']]+list(data_shape)))
            plot_images(samples.reshape([train_config['batch_size']]+list(data_shape)))
        return output, handle_dict
    return plotting_func


def t_sne_plot():
    """T-SNE visualization of high-dimensional state data."""
    pass

