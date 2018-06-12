import numpy as np
from cfg import run_config, train_config


def train(model, data, optimizers):
    return run(model, data, optimizers)

def validate(model, data):
    return run(model, data)

def run(model, data, optimizers=None):
    """
    Runs the model on the data, optionally updates parameters with optimizers.

    Args:
        model (FullyConnectedModel): the model to be run
        data (DataLoader): the data set to run on
        optimizers (tuple, optional): the inference and generative optimizers
    """
    output = {}

    batch_size = train_config['batch_size']
    n_examples = len(data) * batch_size
    n_inf_iter = train_config['n_inf_iterations']

    if optimizers:
        inf_opt, gen_opt = optimizers

    total_elbo = np.zeros((n_examples, n_inf_iter+1))
    total_cll  = np.zeros((n_examples, n_inf_iter+1))
    total_kl   = np.zeros((n_examples, n_inf_iter+1))

    for batch_ind, batch in enumerate(data):
        print('Batch Index: ' + str(batch_ind) + ' / ' + str(len(data)))
        batch_data, _ = batch
        batch_data = batch_data.cuda(run_config['cuda_device']).view(batch_size, -1)

        model.re_init(batch_size=batch_data.shape[0])
        model.generate(gen=False, n_samples=train_config['n_samples'])

        elbo, cll, kl = model.losses(batch_data, averaged=False)
        total_elbo[batch_ind*batch_size:(batch_ind+1)*batch_size, 0] = elbo.detach().cpu().numpy()
        total_cll[batch_ind*batch_size:(batch_ind+1)*batch_size, 0]  = cll.detach().cpu().numpy()
        total_kl[batch_ind*batch_size:(batch_ind+1)*batch_size, 0]   = sum(kl).detach().cpu().numpy()
        (-elbo.mean()).backward(retain_graph=True)
        print(str(float(elbo.mean().detach().cpu().numpy())))

        for inf_it in range(n_inf_iter):

            if optimizers:
                if inf_it == n_inf_iter - 1:
                    gen_opt.zero_grad()

            model.infer(batch_data)
            model.generate(gen=False, n_samples=train_config['n_samples'])

            elbo, cll, kl = model.losses(batch_data, averaged=False)
            total_elbo[batch_ind*batch_size:(batch_ind+1)*batch_size, inf_it+1] = elbo.detach().cpu().numpy()
            total_cll[batch_ind*batch_size:(batch_ind+1)*batch_size, inf_it+1]  = cll.detach().cpu().numpy()
            total_kl[batch_ind*batch_size:(batch_ind+1)*batch_size, inf_it+1]   = sum(kl).detach().cpu().numpy()
            (-elbo.mean()).backward(retain_graph=True)
            print(str(float(elbo.mean().detach().cpu().numpy())))

        if optimizers:
            for param in model.inference_parameters():
                param.grad /= n_inf_iter
            inf_opt.step(); gen_opt.step()

    output['elbo'] = total_elbo
    output['cll']  = total_cll
    output['kl']   = total_kl

    return output
