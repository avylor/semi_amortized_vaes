"""
VAE model nn.Module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, latent_dim, img_size=28 * 28):
        super(VAE, self).__init__()

        self.img_size = img_size
        self.latent_dim = latent_dim

        self.fc11 = nn.Linear(img_size, 200)
        self.fc12 = nn.Linear(img_size, 200)

        self.fc21 = nn.Linear(200, 200)
        self.fc22 = nn.Linear(200, 200)

        self.fc31 = nn.Linear(200, latent_dim)
        self.fc32 = nn.Linear(200, latent_dim)

        self.fc4 = nn.Linear(latent_dim, 200)
        self.fc5 = nn.Linear(200, 200)
        self.fc6 = nn.Linear(200, img_size)

        self.enc = nn.ModuleList([self.fc11, self.fc12, self.fc21, self.fc22, self.fc31, self.fc32])
        self.dec = nn.ModuleList([self.fc4, self.fc5, self.fc6])

    def enc_forward(self, x):
        h11 = F.relu(self.fc11(x.view(-1, self.img_size)))
        h12 = F.relu(self.fc12(x.view(-1, self.img_size)))
        h21 = F.relu(self.fc21(h11))
        h22 = F.relu(self.fc22(h12))
        return self.fc31(h21), self.fc32(h22)

    def reparameterize(self, mu, logvar, z=None):
        std = torch.exp(0.5 * logvar)
        if z is None:
            z = torch.randn_like(std)
        return mu + z * std

    def dec_forward(self, z):
        h4 = F.relu(self.fc4(z))
        h5 = F.relu(self.fc5(h4))
        return self.fc6(h5)

    def forward(self, x):
        mu, logvar = self.enc_forward(x.view(-1, self.img_size))
        z = self.reparameterize(mu, logvar)
        return self.dec_forward(z), mu, logvar
