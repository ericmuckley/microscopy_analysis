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
    fig.set_size_inches(8, 8)

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

def get_window_slices(img, samples, oversamples):
    """Get index slices which describe bounds of a sliding
    window that rasters accross a 2D image.
    Two sliding windows are created:
    1. a sampling window, with 'samples' number of pixels
    2. an oversmaplign window, with 'oversamples' number of pixels
    Returns a dictionary of the index slices for both sliding windows."""
    # get list of coordinates at which to anchor each sampling window
    window_anchors_x = np.arange(0, len(img[0]), samples)
    window_anchors_y = np.arange(0, len(img), samples)
    window_coords = np.array(np.meshgrid(
        window_anchors_x, window_anchors_y)).T.reshape(-1,2)
    # create lists for sample and oversample window index slices
    s_slices, os_slices = [], []
    # loop over each set of coordinates for the window anchor
    for x0, y0 in window_coords:
        # find indices of oversampled window
        over_x0 = np.clip(x0-oversamples, 0, len(img[0]))
        over_x1 = np.clip(x0+oversamples, 0, len(img[0]))
        over_y0 = np.clip(y0-oversamples, 0, len(img))
        over_y1 = np.clip(y0+oversamples, 0, len(img))
        os_slice = np.s_[over_y0:over_y1, over_x0:over_x1]
        s_slice = np.s_[y0:y0+samples, x0:x0+samples]
        s_slices.append(s_slice)
        os_slices.append(os_slice)
    return {'s': s_slices, 'os': os_slices}
