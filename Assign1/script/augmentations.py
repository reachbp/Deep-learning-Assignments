from __future__ import print_function
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import scipy
from scipy import ndimage
import torch


# Transforms a tensor into a numpy array
# ONLY FOR SINGLE CHANNEL
def to_npimg(tensor):
    t = (tensor[0]).numpy()
    return t


def random_translate(npimg, min_translate=-0.2, max_translate=0.2):
    
    h, w = npimg.shape
    min_pixels = ((h + w) / 2) * min_translate
    max_pixels = ((h + w) / 2) * max_translate
    shift = int(random.uniform(min_pixels, max_pixels))
    
    return ndimage.interpolation.shift(npimg, shift, mode='nearest')


def random_rotate(npimg, min_rotate=-30, max_rotate=30):
    
    theta = random.randint(int(min_rotate), int(max_rotate + 1))
    return ndimage.interpolation.rotate(npimg, theta, reshape=False, mode='nearest')


def random_scale(npimg, min_scale=0.7, max_scale=1.3):
    
    scale_factor = random.uniform(min_scale, max_scale)
    sc_img = clipped_zoom(npimg, scale_factor, mode='nearest')
    return sc_img
    

# Converts a HxW ndarray into a 1xHxW tensor
def to_tensor(npimg):
    h, w = npimg.shape
    npimg = npimg.reshape(1, h, w)
    return torch.from_numpy(npimg)


def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # width and height of the zoomed image
    zh = int(np.round(zoom_factor * h))
    zw = int(np.round(zoom_factor * w))

    # for multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # zooming out
    if zoom_factor < 1:
        # bounding box of the clip region within the output array
        top = (h - zh) // 2
        left = (w - zw) // 2
        # zero-padding
        out = np.zeros_like(img)
        out += np.amin(img)
        out[top:top+zh, left:left+zw] = ndimage.zoom(img, zoom_tuple, **kwargs)

    # zooming in
    elif zoom_factor > 1:
        # bounding box of the clip region within the input array
        top = (zh - h) // 2
        left = (zw - w) // 2
        out = ndimage.zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)
        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # if zoom_factor == 1, just return the input array
    else:
        out = img
    return out


# Single channel only
def tensor_imshow(tensor):    
    npimg = tensor.numpy()[0]
    np_imshow(npimg)


def np_imshow(npimg):
    plt.figure(1)
    plt.imshow(npimg, cmap='gray')
    plt.show()
