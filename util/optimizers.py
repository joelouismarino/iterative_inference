import torch.optim as opt
from torch.optim.lr_scheduler import ExponentialLR
from logs import load_opt_checkpoint


def set_gpu_recursive(var, gpu_id):
    for key in var:
        if isinstance(var[key], dict):
            var[key] = set_gpu_recursive(var[key], gpu_id)
        else:
            try:
                var[key] = var[key].cuda(gpu_id)
            except:
                pass
    return var


def get_optimizers(train_config, arch, model):

    enc_opt = dec_opt = None

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

    epoch = -1
    if train_config['resume_experiment'] != '' and train_config['resume_experiment'] is not None:
        # load the old optimizers and update the optimizer state dictionaries
        # epoch is used to set the learning rate schedulers and where to resume training
        old_enc_opt, old_dec_opt, epoch = load_opt_checkpoint()
        enc_opt.load_state_dict(old_enc_opt.state_dict())
        enc_opt.state = set_gpu_recursive(enc_opt.state, train_config['cuda_device'])
        dec_opt.load_state_dict(old_dec_opt.state_dict())
        dec_opt.state = set_gpu_recursive(dec_opt.state, train_config['cuda_device'])

    enc_sched = ExponentialLR(enc_opt, 0.999, last_epoch=epoch)
    dec_sched = ExponentialLR(dec_opt, 0.999, last_epoch=epoch)

    return (enc_opt, enc_sched), (dec_opt, dec_sched), epoch
