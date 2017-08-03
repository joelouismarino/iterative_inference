import os
from time import gmtime, strftime


def init_log(log_root):
    # creates a directory, format: day_month_year_hour_minutes_seconds
    log_dir = strftime("%d_%b_%Y_%H_%M_%S", gmtime())
    log_path = os.path.join(log_root + log_dir)
    os.makedirs(log_path)
    os.system("rsync -au --include '*/' --include '*.py' --exclude '*' . " + log_path + "source")
    return log_path, log_dir


def log_train(func):
    """Wrapper to log train metrics."""
    def log_func(model, train_config, data, epoch, optimizers):
        output = func(model, train_config, data, optimizers)
        avg_elbo, avg_cond_log_like, avg_kl = output
        # todo: log

        return output

    return log_func


def log_vis(func):

    def log_func(model, train_config, data_loader, epoch, vis=True):
        output = func(model, train_config, data_loader, vis=vis)
        total_elbo, total_cond_log_like, total_kl, total_labels, total_recon, total_posterior, total_prior, samples = output

        # todo: log
        
        return output

    return log_func