import torch.optim as opt
from config import train_config, arch
from models import LatentVariableModel
from util.data import load_data
from util.train_val import train, validate

data_path = '/home/joe/Datasets'

# data, labels
(train_data, train_labels), (val_data, val_labels) = load_data(train_config['dataset'], data_path)

# construct model
model = LatentVariableModel(train_config, arch)

# construct optimizers
encoder_params = model.encoder_parameters()
encoder_optimizer = opt.Adamax(encoder_params, lr=train_config['learning_rate'])
encoder_scheduler = opt.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, mode='min', factor=0.5)

decoder_params = model.decoder_parameters()
decoder_optimizer = opt.Adamax(decoder_params, lr=train_config['learning_rate'])
decoder_scheduler = opt.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, mode='min', factor=0.5)

for epoch in range(10000):

    train(model, train_data, (encoder_optimizer, decoder_optimizer))
    validate(model, (val_data, val_labels))

    encoder_scheduler.step()
    decoder_scheduler.step()

temp