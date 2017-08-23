from config import train_config, arch
from models import get_model
from util.data.load_data import load_data
from util.misc import get_optimizers
from util.train_val import train, run
from util.plotting import init_plot, save_env
from util.logs import init_log, save_checkpoint
import time

# todo: visualization
#           - visualize errors at input level and reconstructions over inference iterations
#           - latent traversal of lowest variance dimensions
#           - plot number of 'dead' units or active units
# todo: figure out why it is using gpu 0
# todo: get convolutional model working
# todo: add IAF


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
model = get_model(train_config, arch, train_loader)

# get optimizers
(enc_opt, enc_scheduler), (dec_opt, dec_scheduler) = get_optimizers(train_config, model)

for epoch in range(1000):
    print 'Epoch: ' + str(epoch+1)
    # train
    tic = time.time()
    model.train()
    train(model, train_config, train_loader, epoch+1, handle_dict, (enc_opt, dec_opt))
    toc = time.time()
    print 'Time: ' + str(toc - tic)
    # validation
    visualize = False
    eval = False
    if epoch % train_config['display_iter'] == train_config['display_iter']-1:
        save_checkpoint(model, (enc_opt, dec_opt), epoch)
        visualize = True
    #    eval = True
    model.eval()
    _, averages, _ = run(model, train_config, val_loader, epoch+1, handle_dict, vis=visualize, eval=eval, label_names=label_names)
    save_env()
    # enc_scheduler.step(-averages[0])
    # dec_scheduler.step(-averages[0])

