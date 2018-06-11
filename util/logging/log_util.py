import os
import cPickle as pickle


def set_gpu_recursive(var, gpu_id):
    for key in var:
        if isinstance(var[key], dict):
            var[key] = set_gpu_recursive(var[key], gpu_id)
        else:
            try:
                var[key] = var[key].cuda(gpu_id)
            except:
                pass
    return var


def get_last_epoch(path):
    last_epoch = 0
    for r, d, f in os.walk(os.path.join(path, 'checkpoints')):
        for ckpt_dir in d:
            if ckpt_dir != 'best':
                epoch = int(ckpt_dir)
                if epoch > last_epoch:
                    last_epoch = epoch
    return last_epoch


def update_metric(file_name, value):
    if os.path.exists(file_name):
        metric = pickle.load(open(file_name, 'r'))
        metric.append(value)
        pickle.dump(metric, open(file_name, 'w'))
    else:
        pickle.dump([value], open(file_name, 'w'))


def best_performance(free_energy, path):
    # current performance
    fe = free_energy[-1].sum()

    # logged performance
    metric = pickle.load(open(path, 'r'))
    metric = [m[1][-1].sum() for m in metric]

    # compare
    if fe <= min(metric):
        return True
    return False
