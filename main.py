from config import train_config, arch
from models import get_model
from util.data.load_data import load_data
from util.misc import get_optimizers
from util.train_val import train, run
from util.plotting import init_plot, save_env
from util.logs import init_log, save_checkpoint
import time

# todo: better visualization
#           - visualize errors at input level
#           - latent traversal of lowest variance dimensions
#           - plot number of 'dead' units or active units
# todo: better data preprocessing (normalization, etc.)
# todo: add support for online learning
# todo: implement proper evaluation


log_root = '/home/joe/Research/iterative_inference_logs/'
log_path, log_dir = init_log(log_root, train_config)
print 'Experiment: ' + log_dir

global vis
vis, handle_dict = init_plot(train_config, arch, env=log_dir)

# load data, labels
data_path = '/home/joe/Datasets'
train_loader, val_loader, label_names = load_data(train_config['dataset'], data_path, train_config['batch_size'],
                                                  cuda_device=train_config['cuda_device'])

# construct model
model = get_model(train_config, arch, tuple(next(iter(train_loader))[0].size()[1:]))

# get optimizers
(enc_opt, enc_sched), (dec_opt, dec_sched) = get_optimizers(train_config, model)

for epoch in range(1000):
    print 'Epoch: ' + str(epoch+1)
    # train
    tic = time.time()
    model.train()
    train(model, train_config, train_loader, epoch+1, handle_dict, (enc_opt, dec_opt))
    toc = time.time()
    print 'Time: ' + str(toc - tic)
    # validation
    visualize = True
    if epoch % 100 == 0:
        visualize = True
    model.eval()
    _, averages, _ = run(model, train_config, val_loader, epoch+1, handle_dict, vis=visualize, label_names=label_names)
    save_env()
    #enc_sched.step(-averages[0])
    #dec_sched.step(-averages[0])
    #if epoch % 100 == 0:
    #    save_checkpoint(model, (enc_opt, dec_opt), epoch)
