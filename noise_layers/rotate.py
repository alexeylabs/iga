import torch.nn as nn
import torchvision.transforms.functional as F
import numpy as np
from noise_layers.crop import random_float
import random


class Rotate(nn.Module):
    """
    Rotate the image random angle from range
    """

    def __init__(self, rotation_range):
        super(Rotate, self).__init__()
        self.angle_min = rotation_range[0]
        self.angle_max = rotation_range[1]

    def forward(self, noised_and_cover):
        angle = random.randrange(self.angle_min, self.angle_max)
        noised_image = noised_and_cover[0]
        noised_and_cover[0] = F.rotate(
            noised_image,
            angle)

        return noised_and_cover



