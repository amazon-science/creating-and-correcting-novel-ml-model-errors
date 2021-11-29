# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import os

import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from pytorch_msssim import SSIM
from vae_model import vae


class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return 1 - super(SSIM_Loss, self).forward(img1, img2)


def get_dataloader(batch_size):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(os.environ["SM_CHANNEL_TRAIN"], train_transform)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader


def train(epochs, batch_size, learning_rate, z_dimension, beta, annealing_steps):

    # check if GPU is available and set context
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get the dataloaders for train and test data
    train_loader = get_dataloader(batch_size)

    # create and setup VAE
    model = vae.VAE(z_dimension=z_dimension, device=device)

    # assign model on GPU
    model.to(device)
    print(model)

    # specify loss function
    ssim_loss = SSIM_Loss(data_range=1.0, size_average=True, channel=3, nonnegative_ssim=True)
    kld = torch.nn.KLDivLoss(reduction="batchmean")

    # setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    annealing_factor = 1
    total_steps = 0

    # training
    for epoch in range(epochs):

        # training loss for each epoch
        total_loss = 0
        total_reconstruction_loss = 0
        total_kld_loss = 0

        # training mode
        model.train()

        # training phase
        for train_step, (batch, label) in enumerate(train_loader):

            # reset optimizer
            optimizer.zero_grad()

            batch = batch.to(device)

            # Pass batch through model. Model returns reconstructed batch and latent distribution
            recon_batch, mu, logvar, latent_sample = model(batch)

            # reconstruction loss
            reconstruction_loss = ssim_loss(batch, recon_batch)

            # KL loss
            var = logvar.exp()
            dist = torch.distributions.Normal(mu, var).sample()
            unit = torch.distributions.Normal(mu * 0.0, var.pow(0)).sample()
            kld_loss = kld(dist, unit) + kld(unit, dist) / 2.0

            # compute annealing factor
            total_steps += 1
            annealing_factor = min(total_steps / annealing_steps, 1)

            # VAE loss: reconsstruction loss between input and outputs + KL loss
            loss = reconstruction_loss + (beta * annealing_factor * kld_loss)

            # record total loss
            total_loss += loss.item()
            total_reconstruction_loss += reconstruction_loss.item()
            total_kld_loss += kld_loss.item()

            # backward pass
            loss.backward()

            # update parameters
            optimizer.step()

        print(
            "Epoch {} training loss {} reconstruction loss {} KL loss {}".format(
                epoch,
                total_loss / (train_step + 1),
                total_reconstruction_loss / (train_step + 1),
                total_kld_loss / (train_step + 1),
            )
        )

    # save model
    with open(os.path.join(os.environ["SM_MODEL_DIR"], "vae.pt"), "wb") as f:
        torch.save(model.state_dict(), f)

    with open(os.path.join(os.environ["SM_MODEL_DIR"], "encoder.pt"), "wb") as f:
        torch.save(model.encoder.state_dict(), f)

    with open(os.path.join(os.environ["SM_MODEL_DIR"], "decoder.pt"), "wb") as f:
        torch.save(model.decoder.state_dict(), f)


if __name__ == "__main__":

    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--z_dimension", type=int, default=10)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--annealing_steps", type=int, default=50000)

    args, _ = parser.parse_known_args()

    model = train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        z_dimension=args.z_dimension,
        beta=args.beta,
        annealing_steps=args.annealing_steps,
    )
