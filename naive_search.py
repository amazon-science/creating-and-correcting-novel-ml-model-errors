# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import numpy as np
import torch

"""
This file contains the code for the latent search.
"""

# Distribution of noise applied to codes
# We keep this set at a = b = 50 for mnist and german traffic signs
# for svhn we use 75
m = torch.distributions.beta.Beta(torch.tensor(50.0), torch.tensor(50.0))


def naive_latent_search(
    prediction_model, vae, data, original_class, n_samples=100, device="cuda:0", sparse=False
):
    """
    The search alogorithm for finding natural adversarial examples.

    : param prediction_model : cnn
        The CNN used to classify the images.  This is the model we're finding natural adversarial examples for.

    : param vae : vae
        The VAE used to encode the data.

    : param original_class : int
        The original class of the data.

    : param n_samples : int
        The number of perturbations to do in the latent space.

    : param device : str
        The device for the model

    : param sparse : bool
        sparse: whether to generate sparse perturbations (i.e. vary only one latent factor.
    """

    # Convert to original class to numpy array
    if torch.is_tensor(original_class):
        original_class = original_class.detach().cpu().numpy()

    model = prediction_model.to(device)
    data = data.to(device)

    if len(data.shape) == 3:
        data = data.unsqueeze(0)

    # Encode image
    encoded_mu, encoded_logvar = vae.encoder(data)
    encoded_std = torch.exp(0.5 * encoded_logvar)

    # Dictionary to store results
    results = {}

    # Set the original label
    results["original_class"] = original_class

    cur_encoded_mu = encoded_mu.repeat(n_samples, 1)

    if sparse:
        mult_arr = torch.zeros_like(cur_encoded_mu)
        choices = np.random.choice(mult_arr.shape[1], size=mult_arr.shape[0])
        for q in range(mult_arr.shape[0]):
            mult_arr[q, choices[q]] = 1

    # Sample noise
    s = m.sample((n_samples, encoded_mu.shape[1])).to(device)

    # Add to mu
    perturbed_encoded_mu = cur_encoded_mu + s

    if sparse:
        perturbed_encoded_mu = cur_encoded_mu + (
            mult_arr * torch.randn(n_samples, encoded_mu.shape[1], device=device) * std
        )

    # Decode
    decoded = vae.decoder(perturbed_encoded_mu)

    # Get classifier predictions
    predictions = np.argmax(prediction_model(decoded).detach().cpu().numpy(), axis=1)

    results["switched_predictions"] = predictions[predictions != original_class]
    results["switched_codes"] = (
        perturbed_encoded_mu[predictions != original_class].cpu().detach().numpy()
    )
    results["switched_original_classes"] = np.repeat(
        original_class, np.sum([predictions != original_class])
    )
    results["epsilon"] = s[predictions != original_class]
    results["original_mu"] = encoded_mu

    return results


def full_search(
    prediction_model,
    vae,
    data,
    n_samples,
    number_adv_examples_per_image=5,
    device="cuda",
    iters=None,
):
    """
    The full naive search of natural adversarial examples.

    : param prediction_model : cnn
        The CNN model. Should accept array of images and output class probabilities

    : param vae : vae
        The VAE model that models the legitimate set of images.  This model should have *encoder* and *decoder*
        methods which can be called to produce either mu and log variance when encoding and the decoeded instances
        from a latent vector z.

    : param data : dataloader
        A pytorch data loader with the training data set.

    : n_samples : int
        The number of perturbations to make in the latent space in the identification step (Q in the paper).

    : number_adv_examples_per_image : int
        The maximum number of adversarial examples to save for a single image.

    : iters : int
        The number of batches of "data" to run defuse for. For large datasets, you may consider not using the entire dataset.
    """

    q = 0

    # Set iters to all data if iters is None
    if iters is None:
        iters = float("+inf")

    # Store results
    mistaken_codes = []

    number_iters = 0

    for x, y in data:
        for j in range(x.shape[0]):

            # Find natural adversarial examples for an instance
            results = naive_latent_search(
                prediction_model, vae, x[j], y[j], n_samples=n_samples, device=device
            )

            switched_codes = results["switched_codes"]

            # In case we didn't find any examples
            if switched_codes.shape[0] == 0:
                continue

            switched_labels = results["switched_predictions"]
            switched_actual_labels = results["switched_original_classes"]
            epsilon = results["epsilon"]

            # If there are more found than the max allowable size, then choose closest zs
            if epsilon.shape[0] > number_adv_examples_per_image:

                # Using l2 norm
                distances = torch.norm(epsilon, 2, dim=1)

                # Sort the distances and take number_adv_examples_per_image number of codes
                indices = (
                    torch.argsort(distances, dim=0)[:number_adv_examples_per_image]
                    .cpu()
                    .detach()
                    .numpy()
                )
                switched_codes = switched_codes[indices]

            mistaken_codes.extend(switched_codes)

        q += 1

        # Print progress
        if q % 5 == 0:
            print("{}:{}".format(q, number_iters))

        # If we complete iters break
        if q > iters:
            break

    results = {"mistaken_codes": np.array(mistaken_codes)}

    return results
