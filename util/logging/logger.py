import os
import torch
from time import strftime
from util.optimization import load_sched
from log_util import set_gpu_recursive, get_last_epoch, update_metric, best_performance


class Logger(object):

    def __init__(self, run_config):
        """
        Initialize logs, either from existing experiment or create new experiment.
        """
        print('Initializing logs...')
        log_root = run_config['log_root_path']
        self._save_iter = run_config['save_iter']
        self._best_epoch = False
        if run_config['resume_path']:
            # resume an old experiment
            self.log_dir = run_config['resume_path']
            if os.path.exists(os.path.join(log_root, self.log_dir)):
                self.log_path = os.path.join(log_root, self.log_dir)
                print(' Resuming experiment ' + self.log_dir)
            else:
                raise Exception('Experiment folder ' + self.log_dir + ' not found.')
        else:
            # start a new experiment
            if 'log_dir' not in run_config:
                self.log_dir = ''
            else:
                self.log_dir = run_config['log_dir']
            self.log_dir += strftime("%b_%d_%Y_%H_%M_%S") + '/'
            self.log_path = os.path.join(log_root, self.log_dir)
            os.makedirs(self.log_path)
            os.system("rsync -au --include '*/' --include '*.py' --exclude '*' . " + self.log_path + "source")
            os.makedirs(os.path.join(self.log_path, 'metrics'))
            os.makedirs(os.path.join(self.log_path, 'checkpoints'))
            self.epoch = 1
            print(' Starting experiment ' + self.log_dir)

    def save_epoch(self):
        """
        Specifies whether or not to save the model at the current epoch.
        """
        if self._best_epoch:
            # save if we have the best performance
            return True
        # otherwise save only every save iter
        return (self.epoch % self._save_iter) == 0

    def save_checkpoint(self, model, optimizers):
        """
        Save the model, optimizers.

        Args:
            model (LatentVariableModel): model to save
            optimizers (tuple): inference and generative optimizers
        """

        def _save(path, model, optimizers):
            if not os.path.exists(path):
                os.makedirs(path)
            # TODO: put everything on CPU first
            torch.save(model.state_dict(), os.path.join(path, 'model.ckpt'))
            torch.save(tuple([optimizer.opt.state_dict() for optimizer in optimizers]),
                        os.path.join(path, 'opt.ckpt'))

        if (self.epoch % self._save_iter) == 0:
            # we're at a save iteration
            ckpt_path = os.path.join(self.log_path, 'checkpoints', str(self.epoch))
            _save(ckpt_path, model, optimizers)

        if self._best_epoch:
            # overwrite the best model
            ckpt_path = os.path.join(self.log_path, 'checkpoints', 'best')
            _save(ckpt_path, model, optimizers)
            self._best_epoch = False

    def load_checkpoint(self, model, optimizers):
        """
        Load the model and optimizers from the most recent epoch.

        Args:
            model (LatentVariableModel): model to load
            optimizers (tuple): inference and generative optimizers
        """
        self.epoch = get_last_epoch(self.log_path)

        model_state_dict = torch.load(os.path.join(self.log_path, 'checkpoints', str(self.epoch), 'model.ckpt'))
        model.load_state_dict(model_state_dict)

        optimizer_state_dict = torch.load(os.path.join(self.log_path, 'checkpoints', str(self.epoch), 'opt.ckpt'))
        for opt_ind in range(len(optimizers)):
            optimizers[opt_ind].opt.load_state_dict(optimizer_state_dict[opt_ind])
            optimizers[opt_ind].opt.state = set_gpu_recursive(optimizers[opt_ind].opt.state, torch.cuda.current_device())

        schedulers = load_sched(optimizers, self.epoch)

        return model, optimizers, schedulers

    def load_epoch(self, model, epoch):
        self.epoch = epoch
        model_state_dict = torch.load(os.path.join(self.log_path, 'checkpoints', str(epoch), 'model.ckpt'))
        model.load_state_dict(model_state_dict)
        return model

    def load_best(self, model):
        model_state_dict = torch.load(os.path.join(self.log_path, 'checkpoints', 'best', 'model.ckpt'))
        model.load_state_dict(model_state_dict)
        return model

    def _set_best_epoch(self, free_energy):
        """
        Sets the self._best_epoch flag by comparing current (val) free energy
        with logged values. If the current epoch is the best performance, the
        flag is set to True, which prompts the current model to be logged.

        Args:
            free energy (ndarray): numpy array containing (val) free energy
        """
        path = os.path.join(self.log_path, 'metrics', 'val' + '_free_energy.p')
        self._best_epoch = best_performance(free_energy, path)

    def log(self, out_dict, train_val):
        """
        Function to log results.

        Args:
            out_dict (dict): dictionary of metrics from current epoch
            train_val (str): determines whether results are from training or validation
        """
        train_val = train_val.lower()
        update_metric(os.path.join(self.log_path, 'metrics', train_val + '_free_energy.p'),
                      (self.epoch, out_dict['free_energy']))
        update_metric(os.path.join(self.log_path, 'metrics', train_val + '_cond_log_like.p'),
                      (self.epoch, out_dict['cond_log_like']))
        update_metric(os.path.join(self.log_path, 'metrics', train_val + '_kl_div.p'),
                      (self.epoch, out_dict['kl_div']))
        if train_val == 'val':
            self._set_best_epoch(out_dict['free_energy'])

    def step(self):
        self.epoch += 1
