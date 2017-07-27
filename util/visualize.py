import visdom
import numpy as np

global vis


def initialize_environment(environment='main', port=8097):
    """Creates a visdom environment."""
    global vis
    vis = visdom.Visdom(port=port, env=environment)
    return vis


def save_environment():
    """Saves the visdom environment."""
    global vis
    vis.save([vis.env])


def plot_images(imgs):
    """Wraps visdom's image and images functions."""
    global vis
    if len(imgs.shape) == 3:
        imgs = np.expand_dims(imgs, axis=0)
    if imgs.shape[-1] == 3 or imgs.shape[-1] == 1:
        imgs = imgs.transpose((0, 3, 1, 2))
    win = vis.images(imgs)
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


def plot_metrics()


def t_sne_plot():
    pass

