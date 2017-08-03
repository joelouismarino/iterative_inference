from config import train_config, arch
from models import LatentVariableModel
from util.data.load_data import load_data
from util.misc import get_optimizers
from util.train_val import train, run
from util.plotting import init_plot, save_env
from util.logs import init_log
import time

# todo: set up logging
# todo: better visualization
# todo: better data preprocessing (normalization, etc.)
# todo: add support for online learning

log_root = '/home/joe/Research/iterative_inference_logs'
log_path, log_dir = init_log(log_root)

global vis
vis, handle_dict = init_plot(train_config, arch, env=log_dir)

# load data, labels
data_path = '/home/joe/Datasets'
train_loader, val_loader = load_data(train_config['dataset'], data_path, train_config['batch_size'], cuda_device=train_config['cuda_device'])

# construct model
model = LatentVariableModel(train_config, arch, tuple(next(iter(train_loader))[0].size()[1:]))

# get optimizers
(enc_opt, enc_sched), (dec_opt, dec_sched) = get_optimizers(train_config, model)

for epoch in range(500):
    print 'Epoch: ' + str(epoch+1)
    # train
    tic = time.time()
    train(model, train_config, train_loader, epoch+1, handle_dict, (enc_opt, dec_opt))
    toc = time.time()
    print 'Time: ' + str(toc - tic)
    # validation
    visualize = False
    if epoch % 100 == 0:
        visualize = True
    run(model, train_config, val_loader, epoch+1, handle_dict, vis=visualize)
    save_env()
    #enc_sched.step(); dec_sched.step()
