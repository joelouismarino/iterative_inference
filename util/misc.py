import torch.optim as opt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from logs import load_opt_checkpoint

# todo: allow for options in optimzer and scheduler


def get_optimizers(train_config, arch, model):

    enc_opt = dec_opt = None

    if train_config['resume_experiment'] != '' and train_config['resume_experiment'] is not None:
        enc_opt, dec_opt = load_opt_checkpoint()
    else:
        if arch['encoder_type'] == 'inference_model':
            encoder_params = model.encoder_parameters()
        else:
            encoder_params = model.state_parameters()

        if train_config['encoder_optimizer'] in ['sgd', 'SGD']:
            enc_opt = opt.SGD(encoder_params, lr=train_config['encoder_learning_rate'])
        elif train_config['encoder_optimizer'] in ['rmsprop', 'RMSprop']:
            enc_opt = opt.RMSprop(encoder_params, lr=train_config['encoder_learning_rate'])
        elif train_config['encoder_optimizer'] in ['adam', 'Adam']:
            enc_opt = opt.Adam(encoder_params, lr=train_config['encoder_learning_rate'])

        decoder_params = model.decoder_parameters()

        if train_config['decoder_optimizer'] in ['sgd', 'SGD']:
            dec_opt = opt.SGD(decoder_params, lr=train_config['decoder_learning_rate'])
        elif train_config['decoder_optimizer'] in ['rmsprop', 'RMSprop']:
            dec_opt = opt.RMSprop(decoder_params, lr=train_config['decoder_learning_rate'])
        elif train_config['decoder_optimizer'] in ['adam', 'Adam']:
            dec_opt = opt.Adam(decoder_params, lr=train_config['decoder_learning_rate'])

    enc_sched = dec_sched = None

    # enc_sched = ReduceLROnPlateau(enc_opt, mode='min', factor=0.5)
    # dec_sched = ReduceLROnPlateau(dec_opt, mode='min', factor=0.5)

    return (enc_opt, enc_sched), (dec_opt, dec_sched)
