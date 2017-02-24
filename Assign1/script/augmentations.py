from __future__ import print_function
import pickle 
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import PIL
from PIL import Image
import matplotlib.pyplot as plt
import random
import math

def to_2D(image):
    
    c, h, w = image.size()

    new_image = torch.FloatTensor(2, h, w).fill_(image.max())
    new_image[0] = image

    return new_image

def to_1D(image):
    return image[0]

def imshow_tensor(tensor, display=True):
    loader = transforms.ToPILImage()
    image = tensor.clone().cpu()
    image.resize_(3, 28, 28)
    
    image = loader(image)
    
    if display:
        plt.figure(1)
        plt.imshow(image)
        plt.show()
    else:
        return image
    
def imshow_image(image):
    plt.figure(1)
    plt.imshow(image, cmap='gray')
    plt.show()
    
def translate(image, min_translate=0.0, max_translate=0.2):
    
    width, height = image.size
    
    # Random horizontal shift
    # TODO: Modify to shift in any direction
    shift = int(random.uniform(min_translate, max_translate) * 
                ((width + height) / 2))
    
    shifted_image = Image.new('L', (width + shift, height))
    shifted_image.paste(image, (shift, 0))
    shifted_image = shifted_image.crop((0, 0, width, height))
    
    # Compute the average intensity of the edge 
    # and apply it to the shifted dark region
    idata = list(shifted_image.getdata())
    
    intensity = 0.0
    for x in range(height):
        intensity += shifted_image.getpixel((shift, x))
    intensity = int(intensity / height)
    
    # Set all pixels in the shifted region
    pix = shifted_image.load()
    for y in range(height):
        for x in range(shift):
            pix[x, y] = intensity
    
    return shifted_image

def scale(image, min_scale=0.8, max_scale=1.2):
    
    width, height = image.size

    # Generate a random scale factor
    scale_factor = random.uniform(min_scale, max_scale)
    new_size = (int(width * scale_factor), int(height * scale_factor))

    image_clone = image.copy()

    # Resize the image to the scale factor
    resized_image = image_clone.resize(new_size)

    # Identify box co-ordinates
    top_left_x = int(math.fabs((width - new_size[0]) / 2))
    top_left_y = int(math.fabs((height - new_size[1]) / 2))
    bottom_left_x = top_left_x + width
    bottom_right_y = top_left_y + height

    # Fit resized image to original size
    if scale_factor < 1:
        image_clone.paste(resized_image, (top_left_x, top_left_y))

    elif scale_factor > 1:
        image_clone = resized_image.crop((
            top_left_x, top_left_y, bottom_left_x, bottom_right_y))
    
    return image_clone

def rotate(image, min_rotate=-20, max_rotate=20):
    
    theta = random.randint(int(min_rotate), int(max_rotate + 1))
    
#     image_clone = image.copy()
#     image_clone.paste(image.rotate(theta), (0, 0))
#     return image_clone

    return image.rotate(theta)
