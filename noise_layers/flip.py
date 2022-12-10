import torch.nn as nn
import torchvision.transforms.functional as F
import random


class Flip(nn.Module):
    """
    Flip the image
    """

    def __init__(self):
        super(Flip, self).__init__()

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        noised_and_cover[0] = F.hflip(noised_image)

        return noised_and_cover
