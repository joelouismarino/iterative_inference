
def plot_line(vis, Y, X=None, legend=None, win=None, title='', xlabel='', ylabel='', xformat='linear', yformat='linear'):
    """Wraps visdom's line function."""
    opts = dict(title=title, xlabel=xlabel, ylabel=ylabel, legend=legend, xtype=xformat, ytype=yformat)
    if win is None:
        win = vis.line(Y, X, opts=opts)
    else:
        win = vis.line(Y, X, win=win, opts=opts, update='append')
    return win


def update_trace(vis, Y, X, win, name):
    """Wraps visdom's updateTrace function."""
    vis.updateTrace(X, Y, win=win, name=name)


def plot_images(vis, imgs, caption=''):
    """Wraps visdom's image and images functions."""
    if len(imgs.shape) == 3:
        imgs = np.expand_dims(imgs, axis=0)
    if imgs.shape[-1] == 3 or imgs.shape[-1] == 1:
        imgs = imgs.transpose((0, 3, 1, 2))
    opts = dict(caption=caption)
    win = vis.images(np.clip(imgs, 0, 255), opts=opts)
    return win

def plot_config(vis, config):
    """Wraps visdom's text box to display configuration parameters."""
    config_string = ''
    for config_item in config:
        config_string += str(config_item) + ' = ' + str(config[config_item]) + ', '
    config_win = vis.text(config_string)
    return config_win
