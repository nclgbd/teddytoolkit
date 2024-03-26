"""Custom transforms for the datasets."""

# torch
from torchvision.transforms import PILToTensor, Lambda, ToPILImage
import torch
import torchxrayvision as xrv


def __convert_rgb_to_grayscale(x):
    return x.mean(2)[None, ...]


def RGBToGrayscale():
    return Lambda(__convert_rgb_to_grayscale)


def __apply_xrv_normalize(x, maxval=255.0):
    return xrv.datasets.normalize(x, maxval)


def XRVNormalize():
    return Lambda(__apply_xrv_normalize)


def __convert_grayscale_to_rgb(x: torch.Tensor):
    x = ToPILImage()(x)
    x = x.convert("RGB")
    return PILToTensor()(x)


def GrayscaleToRGB():
    return Lambda(__convert_grayscale_to_rgb)
