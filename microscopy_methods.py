import os
import cv2
import numpy as np
from ncempy.io import dm
import matplotlib.pyplot as plt

# change matplotlib settings to make plots look nicer
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['xtick.minor.width'] = 3
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.minor.width'] = 3
plt.rcParams['ytick.major.width'] = 3

def plot_setup(labels=['X', 'Y'], fsize=18, title=''):
    """Creates a custom plot configuration to make graphs look nice.
    This should be called between plt.plot() and plt.show() commands."""
    plt.xlabel(str(labels[0]), fontsize=fsize)
    plt.ylabel(str(labels[1]), fontsize=fsize)
    plt.title(title, fontsize=fsize)
    fig = plt.gcf()
    fig.set_size_inches(6, 6)

def read_stack(filepath):
    """Read dm3 stack file and get image information."""
    data = dm.dmReader(filepath)
    x_span = data['pixelSize'][1]*len(data['data'][1])
    y_span = data['pixelSize'][2]*len(data['data'][2])
    data['label'] = filepath.split('/')[-1].split('.dm3')[0]
    data['span'] = (0, x_span, 0, y_span)
    return data

def norm_image(img):
    """Normalize an image so its min and max stretch from 0 to 1."""
    img = img - img.min()
    img = img / img.max()
    return img
