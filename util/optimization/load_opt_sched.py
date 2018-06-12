import torch.optim as opt
from torch.optim.lr_scheduler import ExponentialLR


def load_opt_sched(train_config, model, last_epoch=-1):
    inf_opt, gen_opt = load_opt(train_config, model)
    inf_sched, gen_sched = load_sched((inf_opt, gen_opt), last_epoch)
    return (inf_opt, gen_opt), (inf_sched, gen_sched)

def load_opt(train_config, model):
    inf_params = model.inference_parameters()
    inf_opt = opt.Adam(inf_params, lr=train_config['inference_learning_rate'])
    gen_params = model.generative_parameters()
    gen_opt = opt.Adam(gen_params, lr=train_config['generation_learning_rate'])
    return (inf_opt, gen_opt)

def load_sched(optimizers, last_epoch=-1):
    inf_opt, gen_opt = optimizers
    inf_sched = ExponentialLR(inf_opt, 0.999, last_epoch=last_epoch)
    gen_sched = ExponentialLR(gen_opt, 0.999, last_epoch=last_epoch)
    return (inf_sched, gen_sched)
