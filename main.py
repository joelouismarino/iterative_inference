from cfg.config import train_config, arch
from lib.models import get_model
from util.data.load_data import load_data
from util.optimizers import get_optimizers
from util.train_val import train, run
from util.plotting import init_plot, save_env
from util.logs import init_log, save_checkpoint
import time


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

for epoch in range(start_epoch+1, 1500):
    print 'Epoch: ' + str(epoch+1)
    # train
    tic = time.time()
    model.train()
    train(model, train_config, train_loader, epoch+1, handle_dict, (enc_opt, dec_opt))
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
    _, averages, _ = run(model, train_config, val_loader, epoch+1, handle_dict, vis=visualize, eval=eval, label_names=label_names)
    toc = time.time()
    print 'Validation Time: ' + str(toc - tic)
    print 'ELBO: ' + str(averages[0])
    save_env()
    enc_scheduler.step()
    dec_scheduler.step()
