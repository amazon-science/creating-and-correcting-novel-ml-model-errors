import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms


def create_grid_with_labels(tensor, labels):

    xmax = min(8, tensor.size(0))
    ymax = int(math.ceil(float(tensor.size(0)) / xmax))

    # create empty tensor
    grid = tensor.new_full((tensor.size(1), tensor.size(2) * ymax, tensor.size(3) * xmax), 0)

    counter = 0
    for y in range(ymax):
        for x in range(xmax):
            label = labels[counter]
            original_image = tensor[counter]
            transposed_image = np.asarray(np.transpose(original_image, (1, 2, 0)) * 255).astype(
                "uint8"
            )

            # create opencv 2D matrix from numpy array
            cv2_image = cv2.UMat(transposed_image)

            # add labels to image
            image_with_label = cv2.putText(
                cv2_image,
                f"{str(label)}",
                (0, int(original_image.shape[1] * 0.3)),
                1,  # font
                1,  # scale of font
                (255, 0, 0),  # color
                1,  # thickness
                cv2.LINE_AA,  # line color
            )
            # convert to torch tensor
            image = transforms.ToTensor()(image_with_label.get())

            # add tensor to grid tensor
            grid.narrow(1, y * tensor.size(2), tensor.size(3)).narrow(
                2, x * tensor.size(3), tensor.size(3)
            ).copy_(image)
            counter += 1

    return grid
