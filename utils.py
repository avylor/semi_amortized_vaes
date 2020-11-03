import torch
import torch.distributions
import numpy as np


def log_normal_pdf(sample, mean, logvar):
    """
    Returns log pdf of normal distribution with diagonal covariance.

    :param sample: datapoints
    :param mean: distribution mean
    :param logvar: logarithm of the covariance diagonal
    :return: log normal pdf of sample
    """

    log2pi = torch.log(2 * torch.tensor(np.pi, device=mean.device))
    return torch.sum(-.5 * ((sample - mean) ** 2. * torch.exp(-logvar) + logvar + log2pi), axis=1)


def log_px_z(pred_logits, outcome):
    """
    Returns Bernoulli log probability.

    :param pred_logits: logits for outcome 1
    :param outcome: datapoint
    :return: log Bernoulli probability of outcome given logits in pred_logits
    """

    pred = pred_logits.view(pred_logits.size(0), -1)
    y = outcome.view(outcome.size(0), -1)
    return -torch.sum(torch.max(pred, torch.tensor(0., device=pred.device)) - pred * y +
                      torch.log(1 + torch.exp(-torch.abs(pred))), 1)


def log_pxz(pred_logits, outcome, z):
    """
    Returns log probability of outcome and z assuming standard normal prior for z and Bernoulli for p(outcome | z).

    :param pred_logits: logits for outcome 1
    :param outcome: datapoint
    :param z: latent variable
    :return: log probability p(outcome, z) assuming standard normal prior for z and Bernoulli for p(outcome | z)
    """
    return log_px_z(pred_logits, outcome) + log_normal_pdf(z, torch.zeros_like(z), torch.zeros_like(z))


def log_bernoulli_loss(pred_logits, outcome):
    """
    Computes Bernoulli loss - sigmoid cross-entropy with logits.

    :param pred_logits: logits for outcome 1
    :param outcome: datapoint
    :return: sigmoid cross-entropy with logits
    """
    return -torch.mean(log_px_z(pred_logits, outcome))


def kl_loss(mean1, logvar1, mean2, logvar2):
    """
    KL divergence of two multivariate normal distributions with diagonal covariance.
    :param mean1: mean of distribution 1
    :param logvar1: logarithm of the covariance diagonal of distribution 1
    :param mean2: mean of distribution 2
    :param logvar2: logarithm of the covariance diagonal of distribution 2
    :return: KL divergence of distribution 1 and 2
    """
    result = -0.5 * torch.sum(logvar1 - logvar2 - torch.pow(mean1 - mean2, 2) / logvar2.exp() -
                              torch.exp(logvar1 - logvar2) + 1, 1)
    return result.mean()


def kl_loss_diag(mean, logvar):
    """
    KL divergence of normal distributions with diagonal covariance and standard multivariate normal.

    :param mean1: mean of distribution 1
    :param logvar1: logarithm of the covariance diagonal of distribution 1
    :return: KL divergence of the given distribution and standard multivariate normal
    """
    result = -0.5 * torch.sum(logvar - torch.pow(mean, 2) - torch.exp(logvar) + 1, 1)
    return result.mean()


def kl_loss_full(mean, var, mean_prior, var_prior):
    """
    KL divergence of two multivariate normal distributions.

    :param mean: mean of distribution 1
    :param var: covariance of distribution 1
    :param mean_prior: mean of distribution 2
    :param var_prior: covariance of distribution 2
    :return: KL divergence of distribution 1 and 2
    """
    mvn = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=var)
    prior = torch.distributions.MultivariateNormal(loc=mean_prior, covariance_matrix=var_prior)
    return torch.distributions.kl_divergence(mvn, prior).mean()


def variational_loss(input, img, model, beta_ten, z=None):
    """
    Computes negative ELBO loss with scaling beta for KL part.

    :param input: [mean, logarithm of the covariance diagonal]
    :param img: datapoints
    :param model: vae model for probability computations
    :param beta_ten: tensor with scaling parameter beta
    :param z: latent variables
    :return: negative log likelihood + beta * KL
    """
    mean, logvar = input
    z_samples = model.reparameterize(mean, logvar, z)
    preds = model.dec_forward(z_samples)
    nll = log_bernoulli_loss(preds, img)
    kl = kl_loss_diag(mean, logvar)
    return nll + beta_ten.item() * kl
