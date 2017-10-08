import torch
from torch.autograd import Variable
import numpy as np
from random import shuffle
from cfg.config import train_config, arch

from logs import log_train, log_vis
from plotting import plot_images, plot_line, plot_train, plot_model_vis


def train_on_batch(model, batch, n_iterations, optimizers, train_enc=True, train_dec=True):

    output_dict = dict()

    enc_opt, dec_opt = optimizers
    # initialize the posterior estimate from the prior
    enc_opt.zero_grad()
    model.decode(generate=True)
    model.reset_state()

    if 'gradient' in arch['encoding_form']\
            or 'log_gradient' in arch['encoding_form']\
            or 'scaled_log_gradient' in arch['encoding_form']\
            or 'sign_gradient' in arch['encoding_form']:
        # initialize state gradients
        model.decode()
        elbo = model.elbo(batch, averaged=True)
        (-elbo).backward(retain_graph=True)

        # keep track of state gradient magnitudes
        approx_post_grads = np.zeros((n_iterations + 1, len(model.levels), 2))
        for level_num, level in enumerate(model.levels):
            grads = level.state_gradients()
            approx_post_grads[0, level_num, 0] = grads[0].abs().mean().data.cpu().numpy()[0]
            if len(grads) > 1:
                approx_post_grads[0, level_num, 1] = grads[1].abs().mean().data.cpu().numpy()[0]

    model.not_trainable_state()
    # inference iterations

    for it in range(n_iterations - 1):
        model.encode(batch)
        model.decode()
        elbo = model.elbo(batch, averaged=True)
        (-elbo).backward(retain_graph=True)

        for level_num, level in enumerate(model.levels):
            grads = level.state_gradients()
            approx_post_grads[it+1, level_num, 0] = grads[0].abs().mean().data.cpu().numpy()[0]
            if len(grads) > 1:
                approx_post_grads[it+1, level_num, 1] = grads[1].abs().mean().data.cpu().numpy()[0]

        if not train_config['average_gradient'] or arch['encoder_type'] in ['em', 'EM']:
            if train_enc:
                enc_opt.step()
            enc_opt.zero_grad()

    # final iteration
    dec_opt.zero_grad()
    model.encode(batch)
    model.decode()

    elbo, cond_log_like, kl = model.losses(batch, averaged=True)
    (-elbo).backward()

    for level_num, level in enumerate(model.levels):
        grads = level.state_gradients()
        approx_post_grads[-1, level_num, 0] = grads[0].abs().mean().data.cpu().numpy()[0]
        if len(grads) > 1:
            approx_post_grads[-1, level_num, 1] = grads[1].abs().mean().data.cpu().numpy()[0]

    output_dict['state_grad_mags'] = approx_post_grads

    # divide encoder gradients
    if train_config['average_gradient']:
        for param in model.encoder_parameters():
            param.grad /= n_iterations

    # calculate average gradient magnitudes
    def ave_grad_mag(params):
        grad_mag = 0.
        num_params = 1
        for param in params:
            if param.grad is not None:
                grad_mag += param.grad.abs().sum().data.cpu().numpy()[0]
                num_params += param.grad.view(-1).size()[0]
        return grad_mag / num_params

    grad_mags = np.zeros((len(model.levels)+1, 2))
    for level_num, level in enumerate(model.levels):
        encoder_grad_mag = ave_grad_mag(level.encoder_parameters())
        decoder_grad_mag = ave_grad_mag(level.decoder_parameters())
        grad_mags[level_num, :] = np.array([encoder_grad_mag, decoder_grad_mag])
    output_decoder_grad_mag = ave_grad_mag(model.output_decoder.parameters())
    grad_mags[-1, :] = np.array([0., output_decoder_grad_mag])
    output_dict['param_grad_mags'] = grad_mags

    # update parameters
    if train_enc:
        enc_opt.step()
    if train_dec:
        dec_opt.step()

    output_dict['elbo'] = elbo.data.cpu().numpy()[0]
    output_dict['cond_log_like'] = cond_log_like.data.cpu().numpy()[0]
    for level in range(len(kl)):
        kl[level] = kl[level].data.cpu().numpy()[0]
    output_dict['kl'] = kl

    return output_dict


def run_on_batch(model, batch, n_iterations, vis=False):
    """Runs the model on a single batch. If visualizing, stores posteriors, priors, and output distributions."""

    output_dict = dict()

    batch_shape = list(batch.size())
    total_elbo = np.zeros((batch.size()[0], n_iterations+1))
    total_cond_log_like = np.zeros((batch.size()[0], n_iterations+1))
    total_kl = [np.zeros((batch.size()[0], n_iterations+1)) for _ in range(len(model.levels))]

    cond_like = reconstructions = posterior = prior = None
    if vis:
        # store the cond_like, reconstructions, posterior, and prior over iterations for the entire batch
        # note: cond_like is the network output, reconstructions are the images
        cond_like = np.zeros([batch_shape[0], n_iterations+1, 2] + batch_shape[1:])
        reconstructions = np.zeros([batch_shape[0], n_iterations+1] + batch_shape[1:])
        posterior = [np.zeros([batch_shape[0], n_iterations+1, 2, model.levels[level].n_latent]) for level in range(len(model.levels))]
        prior = [np.zeros([batch_shape[0], n_iterations+1, 2, model.levels[level].n_latent]) for level in range(len(model.levels))]

    # initialize the model from the prior
    model.decode(generate=True)
    model.reset_state()
    elbo, cond_log_like, kl = model.losses(batch)

    total_elbo[:, 0] = elbo.data.cpu().numpy()
    total_cond_log_like[:, 0] = cond_log_like.data.cpu().numpy()
    for level in range(len(kl)):
        total_kl[level][:, 0] = kl[level].data.cpu().numpy()

    if vis:
        cond_like[:, 0, 0] = model.output_dist.mean.data.cpu().numpy().reshape(batch_shape)
        reconstructions[:, 0] = model.reconstruction.data.cpu().numpy().reshape(batch_shape)
        if model.output_distribution == 'gaussian':
            cond_like[:, 0, 1] = model.output_dist.log_var.data.cpu().numpy().reshape(batch_shape)
        for level in range(len(model.levels)):
            posterior[level][:, 0, 0, :] = model.levels[level].latent.posterior.mean.data.cpu().numpy()
            if arch['posterior_form'] == 'gaussian':
                posterior[level][:, 0, 1, :] = model.levels[level].latent.posterior.log_var.data.cpu().numpy()
            prior[level][:, 0, 0, :] = model.levels[level].latent.prior.mean.data.cpu().numpy()
            prior[level][:, 0, 1, :] = model.levels[level].latent.prior.log_var.data.cpu().numpy()

    if 'gradient' in arch['encoding_form'] \
            or 'log_gradient' in arch['encoding_form'] \
            or 'scaled_log_gradient' in arch['encoding_form'] \
            or 'sign_gradient' in arch['encoding_form']:
        # initialize state gradients
        model.decode()
        elbo = model.elbo(batch, averaged=True)
        (-elbo).backward(retain_graph=True)

    model.not_trainable_state()

    # inference iterations
    for i in range(1, n_iterations+1):
        model.encode(batch)
        model.decode()
        elbo, cond_log_like, kl = model.losses(batch)
        if 'gradient' in arch['encoding_form'] \
                or 'log_gradient' in arch['encoding_form'] \
                or 'scaled_log_gradient' in arch['encoding_form'] \
                or 'sign_gradient' in arch['encoding_form']:
            (-elbo.mean(0)).backward(retain_graph=True)
        total_elbo[:, i] = elbo.data.cpu().numpy()
        total_cond_log_like[:, i] = cond_log_like.data.cpu().numpy()
        for level in range(len(kl)):
            total_kl[level][:, i] = kl[level].data.cpu().numpy()
        if vis:
            cond_like[:, 0, 0] = model.output_dist.mean.data.cpu().numpy().reshape(batch_shape)
            reconstructions[:, i] = model.reconstruction.data.cpu().numpy().reshape(batch_shape)
            if model.output_distribution == 'gaussian':
                cond_like[:, i, 1] = model.output_dist.log_var.data.cpu().numpy().reshape(batch_shape)
            for level in range(len(model.levels)):
                posterior[level][:, i, 0, :] = model.levels[level].latent.posterior.mean.data.cpu().numpy()
                if arch['posterior_form'] == 'gaussian':
                    posterior[level][:, i, 1, :] = model.levels[level].latent.posterior.log_var.data.cpu().numpy()
                prior[level][:, i, 0, :] = model.levels[level].latent.prior.mean.data.cpu().numpy()
                prior[level][:, i, 1, :] = model.levels[level].latent.prior.log_var.data.cpu().numpy()

    output_dict['total_elbo'] = total_elbo
    output_dict['total_cond_log_like'] = total_cond_log_like
    output_dict['total_kl'] = total_kl
    output_dict['cond_like'] = cond_like
    output_dict['reconstructions'] = reconstructions
    output_dict['posterior'] = posterior
    output_dict['prior'] = prior

    return output_dict


def eval_on_batch(model, batch, n_importance_samples):
    """Estimates marginal log likelihood of data using importance sampling."""
    importance_sample_estimate = np.zeros((batch.size()[0], n_importance_samples))

    for i in range(n_importance_samples):
        model.decode()
        elbo, cond_log_like, kl = model.losses(batch)
        importance_sample_estimate[:, i] = torch.exp(cond_log_like).data.cpu().numpy()
        for level in range(len(model.levels)):
            importance_weight = torch.exp(-model.levels[level].kl_divergence().sum(dim=1))
            importance_sample_estimate[:, i:i + 1] *= importance_weight.data.cpu().numpy().reshape((-1, 1))

    return np.log(np.mean(importance_sample_estimate, axis=1))


@plot_model_vis
@log_vis
def run(model, train_config, data_loader, vis=False, eval=False):
    """Runs the model on a set of data."""

    output_dict = dict()

    batch_size = train_config['batch_size']
    n_iterations = train_config['n_iterations']
    n_examples = batch_size * len(iter(data_loader))
    data_shape = list(next(iter(data_loader))[0].size())[1:]

    total_elbo = np.zeros((n_examples, n_iterations+1))
    total_cond_log_like = np.zeros((n_examples, n_iterations+1))
    total_kl = [np.zeros((n_examples, n_iterations+1)) for _ in range(len(model.levels))]
    total_log_like = np.zeros(n_examples) if eval else None
    total_labels = np.zeros(n_examples)
    total_cond_like = total_recon = total_posterior = total_prior = None
    if vis:
        # to capture all of the val set: replace batch_size with n_examples
        total_cond_like = np.zeros([batch_size, n_iterations + 1, 2] + data_shape)
        total_recon = np.zeros([batch_size, n_iterations + 1] + data_shape)
        total_posterior = [np.zeros([batch_size, n_iterations + 1, 2, model.levels[level].n_latent]) for level in range(len(model.levels))]
        total_prior = [np.zeros([batch_size, n_iterations + 1, 2, model.levels[level].n_latent]) for level in range(len(model.levels))]

    for batch_index, (batch, labels) in enumerate(data_loader):
        batch = Variable(batch)
        if train_config['cuda_device'] is not None:
            batch = batch.cuda(train_config['cuda_device'])

        if model.output_distribution == 'bernoulli':
            batch = 255. * torch.bernoulli(batch / 255.)
        elif model.output_distribution == 'gaussian':
            rand_values = torch.rand(tuple(batch.data.shape)) - 0.5
            if train_config['cuda_device'] is not None:
                rand_values = Variable(rand_values.cuda(train_config['cuda_device']))
            else:
                rand_values = Variable(rand_values)
            batch = torch.clamp(batch + rand_values, 0., 255.)

        batch_output = run_on_batch(model, batch, n_iterations, vis)

        data_index = batch_index * batch_size
        total_elbo[data_index:data_index + batch_size, :] = batch_output['total_elbo']
        total_cond_log_like[data_index:data_index + batch_size, :] = batch_output['total_cond_log_like']
        for level in range(len(model.levels)):
            total_kl[level][data_index:data_index + batch_size, :] = batch_output['total_kl'][level]

        total_labels[data_index:data_index + batch_size] = labels.numpy()

        if vis and batch_index == 0:
            total_cond_like[data_index:data_index + batch_size] = batch_output['cond_like']
            total_recon[data_index:data_index + batch_size] = batch_output['reconstructions']
            for level in range(len(model.levels)):
                total_posterior[level][data_index:data_index + batch_size] = batch_output['posterior'][level]
                total_prior[level][data_index:data_index + batch_size] = batch_output['prior'][level]

        if eval:
            total_log_like[data_index:data_index + batch_size] = eval_on_batch(model, batch, 5000)

    samples = None
    optimization_surface = None
    if vis:
        # visualize samples from the model
        model.decode(generate=True)
        samples = model.reconstruction.data.cpu().numpy().reshape([batch_size]+data_shape)

        # visualize the latent optimization surface
        if arch['n_latent'][0] == 2 and len(arch['n_latent']) == 1:
            print 'Visualizing latent space...'
            optimization_surface = dict()
            optimization_surface['elbo'] = np.zeros((batch_size, 200, 200))
            optimization_surface['kl'] = np.zeros((batch_size, 200, 200))
            optimization_surface['cond_log_like'] = np.zeros((batch_size, 200, 200))
            optimization_surface['gradients'] = np.zeros((batch_size, 2, 200, 200))

            batch = next(iter(data_loader))[0]
            batch = Variable(batch)
            if train_config['cuda_device'] is not None:
                batch = batch.cuda(train_config['cuda_device'])

            if model.output_distribution == 'bernoulli':
                batch = 255. * torch.bernoulli(batch / 255.)
            elif model.output_distribution == 'gaussian':
                rand_values = torch.rand(tuple(batch.data.shape)) - 0.5
                if train_config['cuda_device'] is not None:
                    rand_values = Variable(rand_values.cuda(train_config['cuda_device']))
                else:
                    rand_values = Variable(rand_values)
                batch = torch.clamp(batch + rand_values, 0., 255.)
            model.trainable_state()
            for i_iter, i in enumerate(np.arange(-5, 5, 0.05)):
                for j_iter, j in enumerate(np.arange(-5, 5, 0.05)):
                    # set the approximate posterior and evaluate the loss
                    mean = torch.cat((i * torch.ones(batch_size, 1), j * torch.ones(batch_size, 1)), dim=1)
                    model.levels[0].latent.reset_mean(value=mean)
                    model.decode()
                    elbo, cond_log_like, kl = model.losses(batch)
                    elbo.mean(0).backward()
                    optimization_surface['elbo'][:, i_iter, j_iter] = elbo.data.cpu().numpy()
                    optimization_surface['kl'][:, i_iter, j_iter] = kl[0].data.cpu().numpy()
                    optimization_surface['cond_log_like'][:, i_iter, j_iter] = cond_log_like.data.cpu().numpy()
                    optimization_surface['gradients'][:, :, i_iter, j_iter] = model.levels[0].latent.posterior.mean.grad.data.cpu().numpy()
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad.data.zero_()

    output_dict['total_elbo'] = total_elbo
    output_dict['total_cond_log_like'] = total_cond_log_like
    output_dict['total_kl'] = total_kl
    output_dict['total_log_like'] = total_log_like
    output_dict['total_labels'] = total_labels
    output_dict['total_cond_like'] = total_cond_like
    output_dict['total_recon'] = total_recon
    output_dict['total_posterior'] = total_posterior
    output_dict['total_prior'] = total_prior
    output_dict['samples'] = samples
    output_dict['optimization_surface'] = optimization_surface

    return output_dict


@plot_train
@log_train
def train(model, train_config, data_loader, epoch, optimizers):

    output_dict = dict()

    avg_elbo = []
    avg_cond_log_like = []
    avg_kl = [[] for _ in range(len(model.levels))]
    avg_param_grad_mags = np.zeros((len(model.levels) + 1, 2))
    avg_state_grad_mags = np.zeros((train_config['n_iterations']+1, len(model.levels), 2))

    if train_config['kl_warm_up']:
        # if epoch < 50:
        #     model.kl_weight = 0.
        # elif epoch < 200:
        #     model.kl_weight = (epoch-50) * 1. / 150
        if epoch < 200:
            model.kl_weight = epoch * 1. / 200
        else:
            model.kl_weight = 1.

    for batch, _ in data_loader:
        if train_config['cuda_device'] is not None:
            batch = Variable(batch.cuda(train_config['cuda_device']))
        else:
            batch = Variable(batch)

        if model.output_distribution == 'bernoulli':
            batch = 255. * torch.bernoulli(batch / 255.)
        elif model.output_distribution == 'gaussian':
            rand_values = torch.rand(tuple(batch.data.shape)) - 0.5
            if train_config['cuda_device'] is not None:
                rand_values = Variable(rand_values.cuda(train_config['cuda_device']))
            else:
                rand_values = Variable(rand_values)
            batch = torch.clamp(batch + rand_values, 0., 255.)

        for _ in range(train_config['encoder_decoder_train_multiple']-1):
            train_on_batch(model, batch, train_config['n_iterations'], optimizers, train_enc=True, train_dec=False)

        batch_output = train_on_batch(model, batch, train_config['n_iterations'], optimizers)

        avg_elbo.append(batch_output['elbo'])
        avg_cond_log_like.append(batch_output['cond_log_like'])
        for l in range(len(avg_kl)):
            avg_kl[l].append(batch_output['kl'][l])
        avg_param_grad_mags += batch_output['param_grad_mags']
        avg_state_grad_mags += batch_output['state_grad_mags']

    if np.isnan(np.sum(avg_elbo)):
        raise Exception('Nan encountered during training.')

    model.kl_weight = 1.

    output_dict['avg_elbo'] = np.mean(avg_elbo)
    output_dict['avg_cond_log_like'] = np.mean(avg_cond_log_like)
    output_dict['avg_kl'] = [np.mean(avg_kl[l]) for l in range(len(model.levels))]
    output_dict['avg_param_grad_mags'] = avg_param_grad_mags/len(iter(data_loader))
    output_dict['avg_state_grad_mags'] = avg_state_grad_mags/len(iter(data_loader))

    return output_dict
