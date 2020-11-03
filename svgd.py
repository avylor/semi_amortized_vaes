"""
Stein Variational Gradient Descent implementation.
"""

import torch
import utils
import math

import numpy as np


def get_diff_sq(a, b):
    """
    Computes squared pairwise differences between a and b.

    :param a: tensor a
    :param b: tensor b
    :return: squared pairwise differences between a and b
    """
    aa = a.matmul(a.t())
    bb = b.matmul(b.t())
    ab = a.matmul(b.t())
    diff_sq = -2 * ab + aa.diag().unsqueeze(1) + bb.diag().unsqueeze(0)
    return diff_sq


def K_RBF(x, y):
    """
    Computes RBF kernel of x and y with bandwidth sqrt(0.5 * median(diff_squared) / (nr_datapoints + 1)).

    :param x: tensor x
    :param y: tensor y
    :return: rbf kernel (x,y)
    """
    diff_sq = get_diff_sq(x, y)
    h = torch.median(diff_sq)
    h = torch.sqrt(0.5 * h / math.log(x.size(0) + 1.))
    Kxy = torch.exp(-diff_sq / h ** 2 / 2)
    return Kxy


def svgd(current_particles, model, img, iter=20, lr=0.01):
    """
    Stein Variational Gradient Descent procedure.

    :param current_particles: initial particles
    :param model: vae model for probability computations
    :param img: datapoints
    :param iter: number of iterations
    :param lr: learning rate
    :return: particles after SVGD optimisation
    """
    final_particles = current_particles.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([final_particles], lr=lr)
    for i in range(0, iter):
        particles = final_particles.detach().requires_grad_(True)

        pred = model.dec_forward(particles)
        log_pxz = utils.log_pxz(pred, img, particles)
        grad_z = torch.autograd.grad(torch.sum(log_pxz), particles)[0]

        K_zz = K_RBF(particles, particles.detach())
        grad_K = -torch.autograd.grad(torch.sum(K_zz), particles)[0]

        phi = (K_zz.detach().matmul(grad_z) + grad_K) / particles.size(0)
        optimizer.zero_grad()
        final_particles.grad = -phi

        optimizer.step()

    return final_particles


def svgd_batched(nr_particles, batch_size, current_particles, model, img, iter=20, lr=0.01):
    """
    Stein Variational Gradient Descent procedure for batches.
    Please note that RBF kernel bandwidth is computed on per-batch level.

    :param nr_particles: number of particles per datapoint
    :param batch_size: number of datapoints
    :param current_particles: initial particles -> particles per datapoint 1, particles per datapoint 2, ...
    :param model: vae model for probability computations
    :param img: datapoints -> datapoint 1 x nr_particles, datapoint 2 x nr_particles, ....
    :param iter: number of iterations
    :param lr: learning rate
    :return: particles after SVGD optimisation
    """
    mask = torch.from_numpy(np.kron(np.eye(batch_size), np.ones((nr_particles, nr_particles)))).float().to(img.device)

    final_particles = current_particles.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([final_particles], lr=lr)
    for i in range(0, iter):
        particles = final_particles.detach().requires_grad_(True)

        pred = model.dec_forward(particles)
        log_pxz = utils.log_pxz(pred, img, particles)
        grad_z = torch.autograd.grad(torch.sum(log_pxz), particles)[0]

        K_zz = K_RBF(particles, particles.detach()) * mask
        grad_K = -torch.autograd.grad(torch.sum(K_zz), particles)[0]

        phi = (K_zz.detach().matmul(grad_z) + grad_K) / nr_particles
        optimizer.zero_grad()
        final_particles.grad = -phi

        optimizer.step()

    return final_particles
