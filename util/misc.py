import torch.optim as opt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from logs import load_opt_checkpoint

# todo: allow for options in optimzer and scheduler


def get_optimizers(train_config, model):

    if train_config['resume_experiment'] != '' and train_config['resume_experiment'] is not None:
        enc_opt, dec_opt = load_opt_checkpoint()
    else:
        encoder_params = model.encoder_parameters()
        enc_opt = opt.Adam(encoder_params, lr=train_config['encoder_learning_rate'])

        decoder_params = model.decoder_parameters()
        dec_opt = opt.Adam(decoder_params, lr=train_config['decoder_learning_rate'])

    #enc_sched = ReduceLROnPlateau(enc_opt, mode='min', factor=0.5)
    #dec_sched = ReduceLROnPlateau(dec_opt, mode='min', factor=0.5)

    enc_sched = dec_sched = None

    return (enc_opt, enc_sched), (dec_opt, dec_sched)
