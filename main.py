from lib.models import get_model
from util.data.load_data import load_data
from util.optimizers import get_optimizers
from util.train_val import train, run
from util.plotting import init_plot, save_env
from util.logs import init_log, save_checkpoint
import sys
import os
import time
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument
arg_parser.add_argument('--dataset', default='mnist', help='data set to train on, cifar10 or mnist')
arg_parser.add_argument('--model_type', default='single_level', help='model type, single_level or hierarchical')
arg_parser.add_argument('--inference_type', default='iterative', help='inference type, standard or iterative')
arg_parser.add_argument('--data_path', default='', help='path to data directory root')
arg_parser.add_argument('--log_path', default='', help='path to log directory root')
args = arg_parser.parse_args()

path_to_config = os.path.join(os.getcwd(), 'cfg', args.dataset, args.model_type, args.inference_type)
sys.path.insert(0, path_to_config)
from config import train_config, arch

train_config['data_path'] = args.data_path
train_config['log_root'] = args.log_path


log_root = train_config['log_root']
log_path, log_dir = init_log(log_root, train_config)
print 'Experiment: ' + log_dir

global vis
vis, handle_dict = init_plot(train_config, arch, env=log_dir)

# load data, labels
data_path = train_config['data_path']
train_loader, val_loader, label_names = load_data(train_config['dataset'], data_path, train_config['batch_size'],
                                                  cuda_device=train_config['cuda_device'])

# construct model
model = get_model(train_config, arch, train_loader)

# get optimizers
(enc_opt, enc_scheduler), (dec_opt, dec_scheduler), start_epoch = get_optimizers(train_config, arch, model)

for epoch in range(start_epoch+1, 2000):
    print 'Epoch: ' + str(epoch+1)
    # train
    tic = time.time()
    model.train()
    train(model, train_config, arch, train_loader, epoch+1, handle_dict, (enc_opt, dec_opt))
    toc = time.time()
    print 'Training Time: ' + str(toc - tic)
    # validation
    tic = time.time()
    visualize = False
    eval = False
    if epoch % train_config['display_iter'] == train_config['display_iter']-1:
        save_checkpoint(model, (enc_opt, dec_opt), epoch)
        visualize = True
    if epoch % train_config['eval_iter'] == train_config['eval_iter']-1:
        eval = True
    model.eval()
    _, averages, _ = run(model, train_config, arch, val_loader, epoch+1, handle_dict, vis=visualize, eval=eval, label_names=label_names)
    toc = time.time()
    print 'Validation Time: ' + str(toc - tic)
    print 'ELBO: ' + str(averages[0])
    save_env()
    enc_scheduler.step()
    dec_scheduler.step()
