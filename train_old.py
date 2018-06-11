from cfg import run_config, train_config, data_config, model_config
from util.logging import Logger
from util.plotting import Plotter
from util.data import load_data
from lib.models import load_model
from util.optimization import load_opt_sched
from util.train_val import train, validate


# from lib.models import get_model
# from util.data.load_data import load_data
# from util.optimizers import get_optimizers
# from util.train_val import train, run
# from util.plotting import init_plot, save_env
# from util.logs import init_log, save_checkpoint
# import time

# hack to prevent the data loader from going on GPU 0
# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]=str(run_config['cuda_device'])
# torch.cuda.set_device(0)

# initialize logging and plotting
if run_config['log']:
    logger = Logger(run_config)
if run_config['plot']:
    plotter = Plotter(logger.log_dir, run_config, train_config, model_config, data_config)

if True:
    pass
    # log_root = '/home/joe/Research/iterative_inference_logs/'
    # log_path, log_dir = init_log(log_root, train_config)
    # print 'Experiment: ' + log_dir
    #
    # global vis
    # vis, handle_dict = init_plot(train_config, arch, env=log_dir)

    # load data, labels
    # data_path = '/home/joe/Datasets'
    # train_loader, val_loader, label_names = load_data(train_config['dataset'], data_path, train_config['batch_size'],
    #                                                   cuda_device=train_config['cuda_device'])

# load the data
train_loader, val_loader, label_names = load_data(data_config, train_config['batch_size'])

# load the model
model = load_model(model_config)

# load the optimizers, schedulers
optimizers, schedulers = load_opt_sched(train_config, model)

if True:
    pass
    # construct model
    # model = get_model(train_config, arch, train_loader)

    # get optimizers
    # (enc_opt, enc_scheduler), (dec_opt, dec_scheduler), start_epoch = get_optimizers(train_config, arch, model)

    # if True:
    #     run(model, train_config, val_loader, start_epoch+1, handle_dict, vis=True, eval=True, label_names=label_names)
    #     assert False

# train and validate
while True:
    pass

# for epoch in range(start_epoch+1, 150):
#     print 'Epoch: ' + str(epoch+1)
#     # train
#     tic = time.time()
#     model.train()
#     train(model, train_config, train_loader, epoch+1, handle_dict, (enc_opt, dec_opt))
#     toc = time.time()
#     print 'Training Time: ' + str(toc - tic)
#     # validation
#     tic = time.time()
#     visualize = False
#     eval = False
#     if epoch % train_config['display_iter'] == train_config['display_iter']-1:
#         save_checkpoint(model, (enc_opt, dec_opt), epoch)
#         visualize = True
#     if epoch % train_config['eval_iter'] == train_config['eval_iter']-1:
#         eval = True
#     model.eval()
#     _, averages, _ = run(model, train_config, val_loader, epoch+1, handle_dict, vis=visualize, eval=eval, label_names=label_names)
#     toc = time.time()
#     print 'Validation Time: ' + str(toc - tic)
#     print 'ELBO: ' + str(averages[0])
#     save_env()
#     enc_scheduler.step()
#     dec_scheduler.step()
