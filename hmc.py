"""
Hamilitonian Monte Carlo implementation.
Adapted from https://github.com/franrruiz/vcd_divergence/blob/master/mcmc/hmc_vae.m
"""

import torch

import utils


def hmc_vae(current_q, model, img, epsilon=None, Burn=3, T=10, adapt=0, L=5):
    """
    Hamilitonian Monte Carlo sampler.

    :param current_q: initial samples
    :param model: vae model for probability computations
    :param img: datapoints
    :param epsilon: initial step size
    :param Burn: number of burn in iterations
    :param T: number of MC iterations
    :param adapt: 1 if step size should be adapted during burn in
    :param L: number of leapfrog steps
    :return: final samples, all produced samples, average acceptance rate, adapted step size
    """

    if epsilon is None:
        epsilon = 0.5 / current_q.size(1)

    N = current_q.size(0)
    n = current_q.size(1)

    acceptHist = torch.zeros((N, Burn + T), device=current_q.device)
    logpxzHist = torch.zeros((N, Burn + T), device=current_q.device)
    samples = torch.zeros((N, n, T), device=current_q.device)

    if (Burn + T) == 0:
        z = current_q
        delta = -1
        accRate = 0
        return z, samples, accRate, delta

    eta = 0.01
    opt = 0.9
    cnt = 0
    for i in range(0, Burn + T - 1):
        q = current_q
        p = torch.normal(mean=0., std=1., size=(N, n), device=current_q.device)

        current_p = p
        pred = model.dec_forward(q)
        log_pxz = utils.log_pxz(pred, img, q)
        gradz = torch.autograd.grad(torch.sum(log_pxz), q)[0]

        current_U = - log_pxz
        grad_U = - gradz
        p = p - epsilon * grad_U / 2

        for j in range(0, L - 1):
            q = q + epsilon * p
            if j != L:
                pred = model.dec_forward(q)
                log_pxz = utils.log_pxz(pred, img, q)
                gradz = torch.autograd.grad(torch.sum(log_pxz), q)[0]
                proposed_U = - log_pxz
                grad_U = - gradz
                p = p - epsilon * grad_U

        pred = model.dec_forward(q)
        log_pxz = utils.log_pxz(pred, img, q)
        gradz = torch.autograd.grad(torch.sum(log_pxz), q)[0]
        proposed_U = - log_pxz
        grad_U = - gradz
        p = p - epsilon * grad_U / 2
        p = -p

        current_K = torch.sum(current_p ** 2, 1) / 2
        proposed_K = torch.sum(p ** 2, 1) / 2
        accept = torch.normal(mean=torch.zeros(N), std=torch.ones(N)).to(current_q.device) < torch.exp(
            current_U - proposed_U + current_K - proposed_K)

        acceptHist[:, i] = accept
        current_q = torch.where(accept.unsqueeze(1), q, current_q)
        current_U = torch.where(accept, proposed_U, current_U)

        if (i < Burn) and (adapt == 1):

            change = eta * ((torch.mean(accept.to(torch.float32)) - opt) / opt)
            epsilon = epsilon + change * epsilon

        elif i >= Burn:
            cnt = cnt + 1
            samples[:, :, cnt] = current_q
        logpxzHist[:, i] = - current_U

    z = current_q
    return z, samples, torch.mean(acceptHist.to(torch.float32), 1), epsilon
