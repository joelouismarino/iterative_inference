import torch.optim as opt


def get_optimizers(train_config, model):
    encoder_params = model.encoder_parameters()
    enc_opt = opt.Adamax(encoder_params, lr=train_config['learning_rate'] / train_config['n_iterations'])
    #enc_sched = opt.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, mode='min', factor=0.5)
    enc_sched = None

    decoder_params = model.decoder_parameters()
    dec_opt = opt.Adamax(decoder_params, lr=train_config['learning_rate'])
    #dec_sched = opt.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, mode='min', factor=0.5)
    dec_sched = None

    return (enc_opt, enc_sched), (dec_opt, dec_sched)

