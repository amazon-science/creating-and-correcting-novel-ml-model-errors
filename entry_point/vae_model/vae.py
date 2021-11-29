# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, z_dimension=10):
        super(Encoder, self).__init__()
        self.z_dimension = z_dimension

        # convolutional layers
        self.conv1 = nn.Conv2d(3, 64, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 4, stride=2, padding=1)

        # mean, variance for latent space
        self.encoder_mean = nn.Linear(1024, z_dimension)
        self.encoder_logvar = nn.Linear(1024, z_dimension)

    def forward(self, x):

        batch_size = x.size(0)

        # convolution pass
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))

        # linear pass
        output = torch.flatten(x, start_dim=1)

        # get mean and variance
        logvar = self.encoder_logvar(output)
        mu = self.encoder_mean(output)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, z_dimension=10):
        super(Decoder, self).__init__()

        # linear layers
        self.linear1 = nn.Linear(z_dimension, 1024)

        # transpose convolutional layers
        self.deconv1 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1)

    def forward(self, x):
        batch_size = x.size(0)

        # linear layer pass
        x = torch.relu(self.linear1(x))
        x = x.view(batch_size, 64, 4, 4)

        # deconvoluational layer pass
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = torch.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))

        return x


class VAE(nn.Module):
    def __init__(self, z_dimension, device):
        super(VAE, self).__init__()

        # VAE encoder and decoder
        self.encoder = Encoder(z_dimension=z_dimension).to(device)
        self.decoder = Decoder(z_dimension=z_dimension).to(device)

        # weight intialization
        self.encoder.apply(self.weights_init)
        self.decoder.apply(self.weights_init)

    def weights_init(self, tensor):
        # initialize weights with Xavier intiializer
        if (
            isinstance(tensor, nn.Conv2d)
            or isinstance(tensor, nn.Linear)
            or isinstance(tensor, nn.ConvTranspose2d)
        ):
            torch.nn.init.xavier_uniform_(tensor.weight)

    def forward(self, x):

        # encode inputs
        mu, logvar = self.encoder(x)

        # reparameterize
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            sample = mu + std * eps
        else:
            sample = mu

        # decode outputs
        reconstruct = self.decoder(sample)

        return reconstruct, mu, logvar, sample
