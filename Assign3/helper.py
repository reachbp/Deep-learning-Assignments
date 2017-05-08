import numpy as np
import torch
import matplotlib.pyplot as plt

# Single channel only
def tensor_imshow(tensor):    
    npimg = tensor.numpy()[0]
    np_imshow(npimg)


def np_imshow(npimg):
    shape_len = len(npimg.shape)
    if shape_len in [2, 3]:
        plt.figure(1)
        if len(npimg.shape) == 3:
            plt.imshow(npimg[0], cmap='gray')
        else:
            plt.imshow(npimg, cmap='gray')
        plt.show()
    else:
        print('Invalid dimensions, can only accept a CxHxW or HxW.')
