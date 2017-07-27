from config import train_config, arch
from models import LatentVariableModel
from util.data import load_data
from util.misc import get_optimizers
from util.train_val import train, validate
from util.visualize import initialize_environment, save_environment

# todo: set up logging
global vis
vis = initialize_environment('test')

data_path = '/home/joe/Datasets'

# data, labels
(train_data, train_labels), (val_data, val_labels) = load_data(train_config['dataset'], data_path)

# construct model
model = LatentVariableModel(train_config, arch, train_data.shape[1:])

# get optimizers
(enc_opt, enc_sched), (dec_opt, dec_sched) = get_optimizers(train_config, model)

for epoch in range(1):
    train(model, train_config, train_data.reshape(-1, 3072), (enc_opt, dec_opt))
    validate(model, train_config, (val_data, val_labels))
    save_environment()
    #enc_sched.step(); dec_sched.step()
