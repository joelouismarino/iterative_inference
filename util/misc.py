import torch.optim as opt
from logs import load_opt_checkpoint


def get_optimizers(train_config, model):

    if train_config['resume_experiment'] != '' and train_config['resume_experiment'] is not None:
        enc_opt, dec_opt = load_opt_checkpoint()
    else:
        encoder_params = model.encoder_parameters()
        #enc_opt = opt.Adamax(encoder_params, lr=train_config['learning_rate'] / train_config['n_iterations'])
        enc_opt = opt.Adamax(encoder_params, lr=5*train_config['learning_rate'])

        decoder_params = model.decoder_parameters()
        dec_opt = opt.Adamax(decoder_params, lr=train_config['learning_rate'])

    #enc_sched = opt.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, mode='min', factor=0.5)
    enc_sched = None

    #dec_sched = opt.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, mode='min', factor=0.5)
    dec_sched = None

    return (enc_opt, enc_sched), (dec_opt, dec_sched)

