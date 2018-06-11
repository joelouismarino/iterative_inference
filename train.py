from cfg import run_config, train_config, data_config, model_config
from util.logging import Logger
from util.plotting import Plotter
from util.data import load_data
from lib.models import load_model
from util.optimization import load_opt_sched
from util.train_val import train, validate

# initialize logging and plotting
if run_config['log']:
    logger = Logger(run_config)
if run_config['plot']:
    plotter = Plotter(logger.log_dir, run_config, train_config, model_config, data_config)

# load the data
train_data, val_data, label_names = load_data(data_config, train_config['batch_size'])

# load the model
model = load_model(model_config)

# load the optimizers, schedulers
optimizers, schedulers = load_opt_sched(train_config, model)

# train, validate, step
while True:
    out = train(model, train_data, optimizers)
    if run_config['log']:
        logger.log(out, 'Train')
    if run_config['plot']:
        plotter.plot(out, 'Train')

    out = validate(model, val_data)
    if run_config['log']:
        logger.log(out, 'Val'); logger.step()
    if run_config['plot']:
        plotter.plot(out, 'Val'); plotter.step()
        plotter.save()

    schedulers[0].step(); schedulers[1].step()
