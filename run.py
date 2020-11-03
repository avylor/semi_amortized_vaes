"""
Training and evaluation for VAE types comparison.

SVI and SA-VAE implementation adapted from
Adapted from: https://github.com/harvardnlp/sa-vae
Contrastive Divergence VAE implementation adapted from
https://github.com/franrruiz/vcd_divergence/
"""

import torch
import torch.utils.data
import numpy as np
import time
import wandb
import argparse

import hmc
import svgd
import utils
from models import VAE
from optim_n2n import OptimN2N
from data_utils import load_data
from importance_sampling import importance_sampling

parser = argparse.ArgumentParser()

# input data
parser.add_argument('--dataset', default='fashion_mnist')
parser.add_argument('--checkpoint_path', default='checkpoints/baseline')
parser.add_argument('--data_file', default='data/omniglot/omniglot.pt')
parser.add_argument('--train_from', default='')
parser.add_argument('--batch_size', default=100, type=int)

# model options
parser.add_argument('--img_size', default=[1, 28, 28])
parser.add_argument('--latent_dim', default=10, type=int)
parser.add_argument('--model', default='vae', type=str, choices=['vae', 'svi', 'savae', 'cdiv', 'cdiv_svgd'])
parser.add_argument('--test', type=int, default=0)

# optimization options
parser.add_argument('--num_epochs', default=400, type=int)
parser.add_argument('--svi_steps', default=20, type=int)
parser.add_argument('--svi_lr1', default=1, type=float)
parser.add_argument('--svi_lr2', default=1, type=float)
parser.add_argument('--eps', default=1e-5, type=float)
parser.add_argument('--momentum', default=0.5, type=float)
parser.add_argument('--warmup', default=10., type=int)
parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--max_grad_norm', default=5, type=float)
parser.add_argument('--svi_max_grad_norm', default=5, type=float)
parser.add_argument('--num_svgd_particles', default=20, type=int)
parser.add_argument('--num_svgd_iters', default=20, type=int)
parser.add_argument('--num_hmc_iters', default=8, type=int)
parser.add_argument('--no_validation', type=int, default=0)
parser.add_argument('--shuffle', type=int, default=0)
parser.add_argument('--test_type', type=str, default="vae")

# device and printing settings
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--print_every', type=int, default=100)

args = parser.parse_args()


def main():
    wandb.init(project="vae-comparison")
    wandb.config.update(args)
    log_step = 0

    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # set device
    use_gpu = args.use_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    print("training on {} device".format("cuda" if use_gpu else "cpu"))

    # load dataset
    train_loader, val_loader, test_loader = load_data(dataset=args.dataset,
                                                      batch_size=args.batch_size,
                                                      no_validation=args.no_validation,
                                                      shuffle=args.shuffle,
                                                      data_file=args.data_file)

    # define model or load checkpoint
    if args.train_from == '':
        print('--------------------------------')
        print("initializing new model")
        model = VAE(latent_dim=args.latent_dim)

    else:
        print('--------------------------------')
        print('loading model from ' + args.train_from)
        checkpoint = torch.load(args.train_from)
        model = checkpoint['model']

    print('--------------------------------')
    print("model architecture")
    print(model)

    # set model for training
    model.to(device)
    model.train()

    # define optimizers and their schedulers
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer_enc = torch.optim.Adam(model.enc.parameters(), lr=args.lr)
    optimizer_dec = torch.optim.Adam(model.dec.parameters(), lr=args.lr)
    lr_lambda = lambda count: 0.9
    lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_lambda)
    lr_scheduler_enc = torch.optim.lr_scheduler.MultiplicativeLR(optimizer_enc, lr_lambda=lr_lambda)
    lr_scheduler_dec = torch.optim.lr_scheduler.MultiplicativeLR(optimizer_dec, lr_lambda=lr_lambda)

    # set beta KL scaling parameter
    if args.warmup == 0:
        beta_ten = torch.tensor(1.)
    else:
        beta_ten = torch.tensor(0.1)

    # set savae meta optimizer
    update_params = list(model.dec.parameters())
    meta_optimizer = OptimN2N(utils.variational_loss, model, update_params,
                              beta=beta_ten, eps=args.eps,
                              lr=[args.svi_lr1, args.svi_lr2],
                              iters=args.svi_steps, momentum=args.momentum,
                              acc_param_grads=1,
                              max_grad_norm=args.svi_max_grad_norm)

    # if test flag set, evaluate and exit
    if args.test == 1:
        beta_ten.data.fill_(1.)
        eval(test_loader, model, meta_optimizer, device)
        importance_sampling(data=test_loader, model=model, batch_size=args.batch_size, meta_optimizer=meta_optimizer,
                            device=device, nr_samples=20000, test_mode=True, verbose=True, mode=args.test_type)
        exit()

    # initialize counters and stats
    epoch = 0
    t = 0
    best_val_metric = 100000000
    best_epoch = 0
    loss_stats = []
    # training loop
    C = torch.tensor(0., device=device)
    C_local = torch.zeros(args.batch_size * len(train_loader), device=device)
    epsilon = None
    step = 0
    while epoch < args.num_epochs:

        start_time = time.time()
        epoch += 1

        print('--------------------------------')
        print('starting epoch %d' % epoch)
        train_nll_vae = 0.
        train_kl_vae = 0.
        train_nll_svi = 0.
        train_kl_svi = 0.
        train_cdiv = 0.
        train_nll = 0.
        train_acc_rate = 0.
        num_examples = 0
        count_one_pixels = 0

        for b, datum in enumerate(train_loader):
            t += 1

            if args.warmup > 0:
                beta_ten.data.fill_(torch.min(torch.tensor(1.), beta_ten + 1 / (args.warmup * len(train_loader))).data)

            img, _ = datum
            img = torch.where(img < 0.5, torch.zeros_like(img), torch.ones_like(img))
            if epoch == 1:
                count_one_pixels += torch.sum(img).item()
            img = img.to(device)

            optimizer.zero_grad()
            optimizer_enc.zero_grad()
            optimizer_dec.zero_grad()

            if args.model == 'svi':
                mean_svi = torch.zeros(args.batch_size, args.latent_dim, requires_grad=True, device=device)
                logvar_svi = torch.zeros(args.batch_size, args.latent_dim, requires_grad=True, device=device)
                var_params_svi = meta_optimizer.forward([mean_svi, logvar_svi], img)
                mean_svi_final, logvar_svi_final = var_params_svi
                z_samples = model.reparameterize(mean_svi_final.detach(), logvar_svi_final.detach())
                preds = model.dec_forward(z_samples)
                nll_svi = utils.log_bernoulli_loss(preds, img)
                train_nll_svi += nll_svi.item() * args.batch_size
                kl_svi = utils.kl_loss_diag(mean_svi_final, logvar_svi_final)
                train_kl_svi += kl_svi.item() * args.batch_size
                var_loss = nll_svi + beta_ten.item() * kl_svi
                var_loss.backward()

                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            else:
                mean, logvar = model.enc_forward(img)
                z_samples = model.reparameterize(mean, logvar)
                preds = model.dec_forward(z_samples)
                nll_vae = utils.log_bernoulli_loss(preds, img)
                train_nll_vae += nll_vae.item() * args.batch_size
                kl_vae = utils.kl_loss_diag(mean, logvar)
                train_kl_vae += kl_vae.item() * args.batch_size

                if args.model == 'vae':
                    vae_loss = nll_vae + beta_ten.item() * kl_vae
                    vae_loss.backward()

                    optimizer.step()

                if args.model == 'savae':
                    var_params = torch.cat([mean, logvar], 1)
                    mean_svi = mean.clone().detach().requires_grad_(True)
                    logvar_svi = logvar.clone().detach().requires_grad_(True)

                    var_params_svi = meta_optimizer.forward([mean_svi, logvar_svi], img)
                    mean_svi_final, logvar_svi_final = var_params_svi

                    z_samples = model.reparameterize(mean_svi_final, logvar_svi_final)
                    preds = model.dec_forward(z_samples)
                    nll_svi = utils.log_bernoulli_loss(preds, img)
                    train_nll_svi += nll_svi.item() * args.batch_size
                    kl_svi = utils.kl_loss_diag(mean_svi_final, logvar_svi_final)
                    train_kl_svi += kl_svi.item() * args.batch_size
                    var_loss = nll_svi + beta_ten.item() * kl_svi
                    var_loss.backward(retain_graph=True)
                    var_param_grads = meta_optimizer.backward([mean_svi_final.grad, logvar_svi_final.grad])
                    var_param_grads = torch.cat(var_param_grads, 1)
                    var_params.backward(var_param_grads)

                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.model == "cdiv" or args.model == "cdiv_svgd":

                    pxz = utils.log_pxz(preds, img, z_samples)
                    first_term = torch.mean(pxz) + 0.5 * args.latent_dim
                    logqz = utils.log_normal_pdf(z_samples, mean, logvar)

                    if epoch == 7 and b == 0:  # switch to local variate control
                        C_local = torch.ones(args.batch_size * len(train_loader), device=device) * C

                    if args.model == "cdiv":
                        zt, samples, acc_rate, epsilon = hmc.hmc_vae(z_samples.clone().detach().requires_grad_(),
                                                                     model, img, epsilon=epsilon,
                                                                     Burn=0, T=args.num_hmc_iters, adapt=0, L=5)
                        train_acc_rate += torch.mean(acc_rate) * args.batch_size
                    else:
                        mean_all = torch.repeat_interleave(mean, args.num_svgd_particles, 0)
                        logvar_all = torch.repeat_interleave(logvar, args.num_svgd_particles, 0)
                        img_all = torch.repeat_interleave(img, args.num_svgd_particles, 0)
                        z_samples = mean_all + torch.randn(args.num_svgd_particles * args.batch_size,
                                                           args.latent_dim, device=device) * torch.exp(0.5 * logvar_all)
                        samples = svgd.svgd_batched(args.num_svgd_particles, args.batch_size,
                                                    z_samples, model, img_all.view(-1, 784), iter=args.num_svgd_iters)
                        z_ind = torch.randint(low=0, high=args.num_svgd_particles, size=(args.batch_size,),
                                              device=device) + \
                                torch.tensor(args.num_svgd_particles, device=device) * \
                                torch.arange(0, args.batch_size, device=device)
                        zt = samples[z_ind]

                    preds_zt = model.dec_forward(zt)

                    pxzt = utils.log_pxz(preds_zt, img, zt)
                    g_zt = pxzt + torch.sum(0.5 * ((zt - mean) ** 2) * torch.exp(-logvar), 1)

                    second_term = torch.mean(g_zt)
                    cdiv = -first_term + second_term
                    train_cdiv += cdiv.item() * args.batch_size
                    train_nll += -torch.mean(pxzt).item() * args.batch_size

                    if epoch <= 6:
                        loss = -first_term + torch.mean(torch.sum(0.5 * ((zt - mean) ** 2) * torch.exp(-logvar), 1) +
                                                        (g_zt.detach() - C) * logqz)
                        if b == 0:
                            C = torch.mean(g_zt.detach())
                        else:
                            C = 0.9 * C + 0.1 * torch.mean(g_zt.detach())
                    else:
                        control = C_local[b * args.batch_size:(b + 1) * args.batch_size]
                        loss = -first_term + torch.mean(torch.sum(0.5 * ((zt - mean) ** 2) * torch.exp(-logvar), 1) +
                                                        (g_zt.detach() - control) * logqz)
                        C_local[b * args.batch_size:(b + 1) * args.batch_size] = \
                            0.9 * C_local[b * args.batch_size:(b + 1) * args.batch_size] + 0.1 * g_zt.detach()

                    loss.backward(retain_graph=True)
                    optimizer_enc.step()

                    optimizer_dec.zero_grad()
                    torch.mean(-utils.log_pxz(preds_zt, img, zt)).backward()
                    optimizer_dec.step()

            if t % 15000 == 0:
                if args.model == "cdiv" or args.model == "cdiv_svgd":
                    lr_scheduler_enc.step()
                    lr_scheduler_dec.step()
                else:
                    lr_scheduler.step()

            num_examples += args.batch_size
            if b and (b + 1) % args.print_every == 0:
                step += 1

                print('--------------------------------')
                print('iteration: %d, epoch: %d, batch: %d/%d' %
                      (t, epoch, b + 1, len(train_loader)))
                if epoch > 1:
                    print('best epoch: %d: %.2f' % (best_epoch, best_val_metric))
                print('throughput: %.2f examples/sec' %
                      (num_examples / (time.time() - start_time)))

                if args.model != 'svi':
                    print('train_VAE_NLL: %.2f, train_VAE_KL: %.4f, train_VAE_NLLBnd: %.2f' %
                          (train_nll_vae / num_examples, train_kl_vae / num_examples,
                           (train_nll_vae + train_kl_vae) / num_examples))
                    wandb.log({"train_vae_nll": train_nll_vae / num_examples,
                               "train_vae_kl": train_kl_vae / num_examples,
                               "train_vae_nll_bound": (train_nll_vae + train_kl_vae) / num_examples,
                               }, step=log_step)

                if args.model == 'svi' or args.model == 'savae':
                    print('train_SVI_NLL: %.2f, train_SVI_KL: %.4f, train_SVI_NLLBnd: %.2f' %
                          (train_nll_svi / num_examples, train_kl_svi / num_examples,
                           (train_nll_svi + train_kl_svi) / num_examples))
                    wandb.log({"train_svi_nll": train_nll_svi / num_examples,
                               "train_svi_kl": train_kl_svi / num_examples,
                               "train_svi_nll_bound": (train_nll_svi + train_kl_svi) / num_examples,
                               }, step=log_step)

                if args.model == "cdiv" or args.model == "cdiv_svgd":
                    print('train_NLL: %.2f, train_CDIV: %.4f' %
                          (train_nll / num_examples, train_cdiv / num_examples))
                    wandb.log({"train_nll": train_nll / num_examples,
                               "train_cdiv": train_cdiv / num_examples,
                               }, step=log_step)

                    if args.model == "cdiv":
                        print('train_average_acc_rate: %.3f' %
                              (train_acc_rate / num_examples))
                        wandb.log({"train_average_acc_rate": train_acc_rate / num_examples,
                                   }, step=log_step)
                log_step += 1

        if epoch == 1:
            print('--------------------------------')
            print("count of pixels 1 in training data: {}".format(count_one_pixels))
            wandb.log({"dataset_pixel_check": count_one_pixels}, step=log_step)
        if args.no_validation:
            print('--------------------------------')
            print("[validation disabled!]")
        else:
            val_metric = eval(val_loader, model, meta_optimizer, device, epoch, epsilon, log_step)

        checkpoint = {
            'args': args.__dict__,
            'model': model,
            'loss_stats': loss_stats
        }
        torch.save(checkpoint, args.checkpoint_path + "_last.pt")
        if not args.no_validation:
            loss_stats.append(val_metric)
            if val_metric < best_val_metric:
                best_val_metric = val_metric
                best_epoch = epoch
                print('saving checkpoint to %s' % (args.checkpoint_path + "_best.pt"))
                torch.save(checkpoint, args.checkpoint_path + "_best.pt")


def eval(data, model, meta_optimizer, device, epoch=0, epsilon=None, log_step=0):
    print("********************************")
    print("validation epoch {}".format(epoch))

    num_examples = 0
    total_nll_vae = 0.
    total_kl_vae = 0.
    total_nll_svi = 0.
    total_kl_svi = 0.
    total_cdiv = 0.
    total_nll = 0.

    mean_llh = importance_sampling(data=data, model=model, batch_size=args.batch_size, meta_optimizer=meta_optimizer,
                                   device=device, nr_samples=10, test_mode=False,
                                   verbose=False, mode="vae", log_step=log_step)
    model.eval()

    for datum in data:
        img_pre, _ = datum
        batch_size = args.batch_size
        img = img_pre.to(device)
        img = torch.where(img < 0.5, torch.zeros_like(img), torch.ones_like(img))

        if args.model == 'svi':
            mean_svi = 0.1 * torch.zeros(batch_size, model.latent_dim, device=device, requires_grad=True)
            logvar_svi = 0.1 * torch.zeros(batch_size, model.latent_dim, device=device, requires_grad=True)
            var_params_svi = meta_optimizer.forward([mean_svi, logvar_svi], img)
            mean_svi_final, logvar_svi_final = var_params_svi
            z_samples = model.reparameterize(mean_svi_final.detach(), logvar_svi_final.detach())
            preds = model.dec_forward(z_samples)
            nll_svi = utils.log_bernoulli_loss(preds, img)
            total_nll_svi += nll_svi.item() * batch_size
            kl_svi = utils.kl_loss_diag(mean_svi_final, logvar_svi_final)
            total_kl_svi += kl_svi.item() * batch_size
        else:
            mean, logvar = model.enc_forward(img)
            z_samples = model.reparameterize(mean, logvar)
            preds = model.dec_forward(z_samples)
            nll_vae = utils.log_bernoulli_loss(preds, img)
            total_nll_vae += nll_vae.item() * batch_size
            kl_vae = utils.kl_loss_diag(mean, logvar)
            total_kl_vae += kl_vae.item() * batch_size
            if args.model == 'cdiv' or args.model == "cdiv_svgd":

                pxz = utils.log_pxz(preds, img, z_samples)
                first_term = torch.mean(pxz) + 0.5 * model.latent_dim

                if args.model == "cdiv":
                    zt, samples, acc_rate, epsilon = hmc.hmc_vae(z_samples, model, img, epsilon=epsilon,
                                                                 Burn=0, T=args.num_hmc_iters, adapt=0, L=5)
                else:
                    mean_all = torch.repeat_interleave(mean, args.num_svgd_particles, 0)
                    logvar_all = torch.repeat_interleave(logvar, args.num_svgd_particles, 0)
                    img_all = torch.repeat_interleave(img, args.num_svgd_particles, 0)
                    z_samples = mean_all + torch.randn(args.num_svgd_particles * args.batch_size,
                                                       args.latent_dim, device=device) * torch.exp(0.5 * logvar_all)
                    samples = svgd.svgd_batched(args.num_svgd_particles, args.batch_size,
                                                z_samples, model, img_all.view(-1, 784), iter=args.num_svgd_iters)
                    z_ind = torch.randint(low=0, high=args.num_svgd_particles, size=(args.batch_size,),
                                          device=device) + \
                            torch.tensor(args.num_svgd_particles, device=device) * \
                            torch.arange(0, args.batch_size, device=device)
                    zt = samples[z_ind]

                preds_zt = model.dec_forward(zt)
                preds = preds_zt

                pxzt = utils.log_pxz(preds_zt, img, zt)
                g_zt = pxzt + torch.sum(0.5 * ((zt - mean) ** 2) * torch.exp(-logvar), 1)

                second_term = torch.mean(g_zt)
                cdiv = -first_term + second_term
                total_cdiv += cdiv.item() * batch_size
                total_nll += utils.log_bernoulli_loss(preds_zt, img).item() * batch_size

            if args.model == 'savae':
                mean_svi = mean.data.clone().detach().requires_grad_(True)
                logvar_svi = logvar.data.clone().detach().requires_grad_(True)
                var_params_svi = meta_optimizer.forward([mean_svi, logvar_svi], img)
                mean_svi_final, logvar_svi_final = var_params_svi
                z_samples = model.reparameterize(mean_svi_final, logvar_svi_final)
                preds = model.dec_forward(z_samples.detach())
                nll_svi = utils.log_bernoulli_loss(preds, img)
                total_nll_svi += nll_svi.item() * batch_size
                kl_svi = utils.kl_loss_diag(mean_svi_final, logvar_svi_final)
                total_kl_svi += kl_svi.item() * batch_size

        num_examples += batch_size

    n = min(img.size(0), 8)
    comparison = torch.cat(
        [img_pre[:n], torch.sigmoid(preds.view(-1, 1, args.img_size[1], args.img_size[2])).cpu()[:n]])

    example_images = wandb.Image(comparison, caption="images epoch {}".format(epoch))
    wandb.log({"validation images": example_images}, step=log_step)

    nll_vae = total_nll_vae / num_examples
    kl_vae = total_kl_vae / num_examples
    nll_bound_vae = (total_nll_vae + total_kl_vae) / num_examples

    nll_svi = total_nll_svi / num_examples
    kl_svi = total_kl_svi / num_examples
    nll_bound_svi = (total_nll_svi + total_kl_svi) / num_examples

    total_cdiv = total_cdiv / num_examples
    total_nll = total_nll / num_examples

    val_metric = -1

    if args.model != 'svi':
        print('val_VAE_NLL: %.2f, val_VAE_KL: %.4f, val_VAE_NLLBnd: %.2f' %
              (nll_vae, kl_vae, nll_bound_vae))
        wandb.log({"val_vae_nll": nll_vae,
                   "val_vae_kl": kl_vae,
                   "val_vae_nll_bound": nll_bound_vae,
                   }, step=log_step)
        val_metric = nll_bound_vae
        if args.model == "vae":
            wandb.log({"bernoulli_loss": nll_vae}, step=log_step)

    if args.model == 'svi' or args.model == 'savae':
        print('val_SVI_NLL: %.2f, val_SVI_KL: %.4f, val_SVI_NLLBnd: %.2f' %
              (nll_svi, kl_svi, nll_bound_svi))
        wandb.log({"val_svi_nll": nll_svi,
                   "val_svi_kl": kl_svi,
                   "val_svi_nll_bound": nll_bound_svi,
                   }, step=log_step)
        val_metric = nll_bound_svi
        wandb.log({"bernoulli_loss": nll_svi}, step=log_step)

    if args.model == "cdiv" or args.model == "cdiv_svgd":
        print('val_NLL: %.2f, val_CDIV: %.4f' %
              (total_nll, total_cdiv))
        wandb.log({"val_nll": total_nll,
                   "val_cdiv": total_cdiv,
                   }, step=log_step)
        val_metric = total_cdiv
        wandb.log({"bernoulli_loss": total_nll}, step=log_step)

    wandb.log({"val_metric": val_metric}, step=log_step)

    model.train()
    return val_metric


if __name__ == '__main__':
    main()
