"""Custom transforms for the datasets."""

from PIL import Image


# torch
from torchvision.transforms import PILToTensor, Lambda, ToPILImage
import torch
import torchxrayvision as xrv


def __convert_three_channel_to_one(x):
    return x.mean(2)[None, ...]


def RGBToGrayscale():
    return Lambda(__convert_three_channel_to_one)


def __convert_one_channel_to_three(x: torch.Tensor, mode="BGR"):
    # x = ToPILImage()(x)
    # x = x.convert(mode)
    # return PILToTensor()(x)
    x = x.repeat(3, 1, 1)
    return x


def GrayscaleToRGB():
    return Lambda(__convert_one_channel_to_three)


def __apply_xrv_normalize(x, maxval=255.0):
    return xrv.datasets.normalize(x, maxval)


def XRVNormalize():
    return Lambda(__apply_xrv_normalize)
