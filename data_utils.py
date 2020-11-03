"""
Data loading and plotting utility functions.
"""

import torch
from torchvision import datasets, transforms
import glob
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np


def load_data(dataset, batch_size, no_validation=False, shuffle=False, data_file=None):
    """
    Loads training, validation and test data from the required dataset.

    :param dataset: name of the dataset
    :param batch_size: batch size for loaded dataset
    :param shuffle: true if training and validation datasets should be shuffled each iteration
    :param no_validation:  true if empty validation set should be returned
    :param data_file: directory of data files [needed for omniglot only]
    :return: (train data loader, validation data loader, test data loader)
    """
    if dataset == "omniglot":
        all_data = torch.load(data_file)
        x_train, x_val, x_test = all_data

        # dummy y values since no classes for omniglot
        y_train = torch.zeros(x_train.size(0), 1)
        y_val = torch.zeros(x_val.size(0), 1)
        y_test = torch.zeros(x_test.size(0), 1)

        train = torch.utils.data.TensorDataset(x_train, y_train)
        val = torch.utils.data.TensorDataset(x_val, y_val)
        test = torch.utils.data.TensorDataset(x_test, y_test)

        if no_validation:
            train = torch.utils.data.ConcatDataset([train, val])
            val = torch.utils.data.TensorDataset(torch.tensor([]), torch.tensor([]))

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    elif dataset == "mnist":
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
        train_examples, val_examples = (50000, 10000)
        if no_validation:
            train_examples, val_examples = (60000, 0)
        train_set, val_set = torch.utils.data.random_split(train_dataset, [train_examples, val_examples])

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=False)

    elif dataset == "fashion_mnist":
        train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transforms.ToTensor())
        train_examples, val_examples = (50000, 10000)
        if no_validation:
            train_examples, val_examples = (60000, 0)
        train_set, val_set = torch.utils.data.random_split(train_dataset, [train_examples, val_examples])

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('./data', train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=False)

    else:
        print("Error: {} dataset not supported!")
        quit()

    print('--------------------------------')
    print("{} dataset loaded".format(dataset))
    print("batch size: {}".format(batch_size))
    print("train batches: {}".format(len(train_loader)))
    print("validation batches: {}".format(len(val_loader)))
    print("test batches: {}".format(len(test_loader)))
    if shuffle:
        print("train and validation data will be shuffled")

    return train_loader, val_loader, test_loader


def plot_gaussian_optim(mean, var, good_mean, good_var, mode, step):
    """
    Plots the goal gaussian distribution (blue) and its approximation in the given optimisation step (red).

    :param mean: approximated mean
    :param var: approximated variance
    :param good_mean: goal mean
    :param good_var: goal variance
    :param mode: optimisation procedure name
    :param step: optimisation step number
    """

    mean_plot = mean[0].detach().cpu().numpy()
    var_plot = var[0].detach().cpu().numpy()
    rv = multivariate_normal(mean_plot, var_plot)

    mean_prior_plot = (good_mean * torch.ones_like(mean[0])).cpu().numpy()
    var_prior_plot = (good_var * torch.ones_like(var[0])).cpu().numpy()
    rv2 = multivariate_normal(mean_prior_plot, var_prior_plot)

    lim_min = np.min(mean_prior_plot)
    lim_max = np.max(mean_prior_plot)
    x, y = np.mgrid[lim_min - 5:lim_max + 5:.01, lim_min - 5:lim_max + 5:.01]
    pos = np.dstack((x, y))

    fig = plt.figure()
    ax = plt.subplot(111)
    a = ax.contourf(x, y, rv.pdf(pos), alpha=.5, cmap='Reds', extend='max')
    b = ax.contourf(x, y, rv2.pdf(pos), alpha=.5, cmap='Blues', extend='max')

    plt.savefig("results/{}/{}_optimisations_step_{}.png".format(mode, mode, step))
    plt.close()


def generate_gif(anim_save_path, time_per_image=30):
    """
    Generates a gif and saves it in anim_save_path directory as animation.gif.

    :param anim_save_path: path where images are saved
    :param time_per_image: duration of one frame
    """
    image_files = sorted(glob.glob(anim_save_path + "*.png"))
    images = [Image.open(img) for img in image_files]
    images[0].save(anim_save_path + "/animation.gif", save_all=True, append_images=images[1:],
                   optimize=False, duration=time_per_image * len(image_files), loop=0)
