"""
Importance sampling for VAE.
"""

import torch
import math
import wandb

import hmc
import svgd
import utils


def importance_sampling(data, model, batch_size, meta_optimizer, device, nr_samples, test_mode,
                        verbose=False, mode="vae", adapt=False, epsilon=None, log_step=0):
    """
    Computes importance sampling estimate of data probability.

    :param data: datapoints
    :param model: VAE model
    :param batch_size: batch size
    :param meta_optimizer: savae meta optimizer
    :param device: torch device
    :param nr_samples: number of samples for importance sampling
    :param test_mode: true if run on test data
    :param verbose: true if current estimate should be outputted
    :param mode: mode of evaluation - way of generating sampling distribution
    :param adapt: adapt parameter for hmc
    :param epsilon: epsilon parameter for hmc
    :param log_step: step for metrics logging
    :return: importance sampling estimate of data probability
    """

    model.eval()
    results = torch.zeros(batch_size * len(data))
    disp_const = 2*math.log(1.2)
    t = 0
    S = nr_samples

    for datum in data:
        img_batch, _ = datum
        img_batch = img_batch.to(device)
        for img_single in img_batch:

            img_single = torch.where(img_single < 0.5, torch.zeros_like(img_single), torch.ones_like(img_single))
            img = img_single.repeat(S, 1, 1)

            if mode == 'svi':
                mean_svi = 0.1 * torch.zeros(batch_size, model.latent_dim, device=device, requires_grad=True)
                logvar_svi = 0.1 * torch.zeros(batch_size, model.latent_dim, device=device, requires_grad=True)
                var_params_svi = meta_optimizer.forward([mean_svi, logvar_svi], img)
                mean_svi_final, logvar_svi_final = var_params_svi
                z_samples = model.reparameterize(mean_svi_final.detach(), logvar_svi_final.detach())
                preds = model.dec_forward(z_samples)
                mean, logvar = mean_svi_final, logvar_svi_final

            elif mode == 'cdiv' or mode == "cdiv_svgd":
                mean, logvar = model.enc_forward(img_single.unsqueeze(0))
                z_samples = model.reparameterize(mean, disp_const + logvar)
                if mode == "cdiv":
                    z, samples, acc_rate, _ = hmc.hmc_vae(z_samples,
                                                          model,
                                                          img_single.unsqueeze(0), epsilon=epsilon,
                                                          Burn=0, T=300, adapt=adapt, L=5)
                    mean = torch.mean(samples, 2)
                    var = torch.var(samples, 2)
                else:
                    ind = 0
                    z_samples = mean[ind].unsqueeze(0) + \
                                torch.randn(300, mean.size(1), device=device) * \
                                torch.exp(0.5 * logvar[ind]).unsqueeze(0)
                    samples = svgd.svgd(z_samples, model,
                                        img_single.view(-1, 784), iter=20)
                    mean = torch.mean(samples, 0)
                    var = torch.var(samples, 0)

                eps = 0.00000001
                var = torch.where(var < eps, torch.ones_like(var) * eps, var)
                logvar = torch.log(var)
                z_samples = model.reparameterize(mean.repeat(S, 1), disp_const + logvar.repeat(S, 1))
                preds = model.dec_forward(z_samples)
            else:
                mean, logvar = model.enc_forward(img)
                z_samples = model.reparameterize(mean, disp_const + logvar)
                preds = model.dec_forward(z_samples)
                if mode == 'savae':
                    mean_svi = mean.data.clone().detach().requires_grad_(True)
                    logvar_svi = logvar.clone().detach().requires_grad_(True)
                    var_params_svi = meta_optimizer.forward([mean_svi, logvar_svi], img)
                    mean_svi_final, logvar_svi_final = var_params_svi
                    z_samples = model.reparameterize(mean_svi_final, disp_const + logvar_svi_final)
                    preds = model.dec_forward(z_samples.detach())
                    mean, logvar = mean_svi_final, logvar_svi_final

            log_pxz = utils.log_pxz(preds, img, z_samples)
            logqz = utils.log_normal_pdf(z_samples, mean, disp_const + logvar)
            results[t] = (torch.logsumexp(log_pxz - logqz, 0) - math.log(S)).detach()
            current_mean = torch.sum(results) / (t+1)
            if verbose and t % 100 == 0 and t:
                print("---> IS estimate after {} examples".format(t))
                print(current_mean.item())

            if test_mode and t > 100:
                wandb.log({
                    "IS_mean": current_mean
                })
            t += 1

    if test_mode:
        wandb.log({
            "test_importance_sampling_nll": current_mean
        })
        print('--------------------------------')
        print("IS {} samples per datapoint".format(S))
        print("test IS estimate: {}".format(current_mean.item()))
    else:
        wandb.log({
            "val_importance_sampling_nll": current_mean
        }, step=log_step)
        print('--------------------------------')
        print("IS {} samples per datapoint".format(S))
        print("val IS estimate: {}".format(current_mean.item()))



    model.train()
    return current_mean.item()
