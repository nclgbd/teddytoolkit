"""Custom transforms for the datasets."""

# torch
from torchvision.transforms import PILToTensor, Lambda
import torch
import torchxrayvision as xrv


def __convert_rgb_to_single_channel(x):
    # return Image.fromarray(np.asarray(x)[:, :, 0], mode="L")
    return x.mean(2)[None, ...]


def RGBToSingleChannel():
    return Lambda(__convert_rgb_to_single_channel)


# class RGBToSingleChannel(Lambda):
#     def __init__(self):
#         super().__init__(lambd=__convert_rgb_to_single_channel)


def __apply_xrv_normalize(x, maxval=255.0):
    return xrv.datasets.normalize(x, maxval)


def XRVNormalize():
    return Lambda(__apply_xrv_normalize)


# class XRVNormalize(Lambda):
#     def __init__(self):
#         super().__init__(lambd=__apply_xrv_normalize)
