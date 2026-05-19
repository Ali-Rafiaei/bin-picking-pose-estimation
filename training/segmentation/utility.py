import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks_path", type=str, default=None)
    parser.add_argument("--image_path", type=str, default=None)
    args = parser.parse_args()

    masks = np.load(args.masks_path)
    image = np.array(Image.open(args.image_path)).transpose(2, 0, 1)

    tr = v2.Compose([v2.RandomGrayscale(1.0)])
    transformed_image = tr(torch.from_numpy(image))

    plt.imshow(transformed_image.numpy().transpose(1, 2, 0))
    print(transformed_image.numpy().transpose(1, 2, 0)[:10, :10, 0])
    plt.show()
