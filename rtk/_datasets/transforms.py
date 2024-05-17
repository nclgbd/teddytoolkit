"""Custom transforms for formatted datasets."""

# torch
from torchvision.transforms import Lambda
import torch
import torchxrayvision as xrv


def __convert_three_channel_to_one(x):
    return x.mean(2)[None, ...]


def RGBToGrayscale():
    return Lambda(__convert_three_channel_to_one)


def __convert_one_channel_to_three(x: torch.Tensor, mode="BGR"):
    x = x.repeat(3, 1, 1)
    return x


def GrayscaleToRGB():
    return Lambda(__convert_one_channel_to_three)


def __apply_xrv_normalize(x, maxval=255.0):
    return xrv.datasets.normalize(x, maxval)


def XRVNormalize():
    return Lambda(__apply_xrv_normalize)
