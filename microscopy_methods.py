"""
This module contains functions for analysis of 
a stack of ".dm3" format microscopy images.
To perform the analysis, run the Jupyter Noteboook
called "read_stack.ipynb", which will call this module.
The full repository is located at:
https://github.com/ericmuckley/microscopy_analysis


Created April 8, 2020
author: ericmuckley@gmail.com
"""

import os
import cv2
import sklearn
import skimage
import numpy as np
from ncempy.io import dm
import scipy.fftpack as fp
from sklearn import cluster
import matplotlib.pyplot as plt

# change matplotlib settings to make plots look nicer
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['xtick.minor.width'] = 3
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.minor.width'] = 3
plt.rcParams['ytick.major.width'] = 3


def plot_domains(domains, plot_contours=True, plot_areas=True, plot_domain_markers=True):
    """Plot domain boundaries and areas on an image.
    Uses 'domains' dictionary which holds information
    about domain contours and areas."""
    # loop over each domain
    for i in range(len(domains['areas'])):
        if plot_contours:
            # plot contour line
            plt.plot(domains['contours'][i][:,0],
                     domains['contours'][i][:,1],
                     linewidth=1.5,
                     c='r')
        if plot_areas:
            # plot areas on domain map
            plt.text(domains['mean_x'][i],
                     domains['mean_y'][i],
                     str(round(domains['areas'][i])), fontsize=14)
        if plot_domain_markers:
            # plot marker in the center of each domain
            plt.scatter(domains['mean_x'][i],
                        domains['mean_y'][i],
                        marker='x',
                        c='r',
                        s=50)

def print_domain_info(domains):
    """Print statistics related to calculated domains."""
    print('Detected {} domains with mean area of {} pixels'.format(
        len(domains['areas']), round(np.mean(domains['areas']),1)))
    print('Total image area (pixels): {}'.format(domains['tot_area']))
    print('Total domain area (pixels): {}'.format(int(domains['tot_domain_area'])))
    print('Total domain area (%): {}'.format(round(domains['domain_percent'], 1)))

def get_kmeans_map(img, n_clusters=2):
    """Perform K-means clustering algorithm to find similarly-colored
    regions in an image. Number fo regions is equal to 'n_clusters'."""
    # perform k-means clustering and detect contours of kmeans map
    kmeans = cluster.KMeans(n_clusters=n_clusters)
    kmeans.fit(img.reshape((-1, 1)))
    # create 2D map of k-means clustering results
    kmeans_map = norm_image(kmeans.labels_.reshape(img.shape))
    return kmeans_map

def get_domains(img, contour_level=0.5, min_area=1, max_area=100000):
    """Get crystallite domain information extracted from an image.
    Use a highly contrasted image, such as one output from the
    'get_kmeans_map' function. areas are calculated. Returns a dictionary of countour lines which
    define each cluster domain and area of each domain."""
    # extract contour line ordered pairs from the map
    contours = skimage.measure.find_contours(img, contour_level)
    # calculate area of inside each contour line
    areas = [cv2.contourArea(cv2.UMat(
        np.expand_dims(c.astype(np.float32), 1))) for c in contours]
    # create zipped list of areas and contour lines below area threshold
    zipped = [i for i in zip(areas, contours) if min_area <= i[0] <= max_area]
    # sort from large area to small area domains
    zipped.sort(key=lambda x: x[0], reverse=True)
    # get domain statistics
    tot_area = img.shape[0]*img.shape[1]
    tot_domain_area = np.sum([z[0] for z in zipped])
    domain_percent = 100 * tot_domain_area / tot_area
    # create dictionary to hold results
    domains = {
        'tot_area': int(tot_area),
        'tot_domain_area': int(tot_domain_area),
        'domain_percent': domain_percent,
        'areas': [z[0] for z in zipped],
        'contours': [np.flip(z[1]) for z in zipped],
        'mean_x': [np.mean(z[1][:, 1]) for z in zipped],
        'mean_y': [np.mean(z[1][:, 0]) for z in zipped]}
    return domains

def show_image_stack(data):
    """Shows each image in an image stack. The input
    is a dictionary which contains an image stack in
    the key 'data'."""
    # set number of columns and rows for subplots array
    columns = 4
    rows = int(np.ceil(data['data'].shape[0]/columns))
    # loop over each image in stak and plot it
    for i in range(data['data'].shape[0]):
        ax = plt.subplot(rows, columns, i+1)
        ax.axis(False)
        ax.imshow(
            norm_image(data['data'][i,:,:]),
            origin='lower',
            cmap='gray')
        ax.text(100, 100, str(i), color='k', fontsize=24)
    fig = plt.gcf()
    fig.set_size_inches((9, 20))
    plt.tight_layout()
    plt.show()

def plot_setup(labels=['X', 'Y'], fsize=18, title='',
               axes_on=True):
    """Creates a custom plot configuration to make graphs look nice.
    This should be called between plt.plot() and plt.show() commands."""
    plt.xlabel(str(labels[0]), fontsize=fsize)
    plt.ylabel(str(labels[1]), fontsize=fsize)
    plt.axis(axes_on)
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
    median, std = np.median(img), np.std(img)
    low_clip, high_clip = median-5*std, median+5*std
    img = np.clip(img, low_clip, high_clip)
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


def get_gradient_info(img):
    """Get magnitude and phase of gradient of 2D image."""
    grad_x, grad_y = np.abs(np.gradient(img))
    grad_mag = np.mean(np.hypot(grad_x, grad_y))
    #grad_phase = np.mean(np.arctan(grad_y/grad_x))
    return grad_mag

def map_local_fft(img, slices):
    """Create FFT maps of an image using a sliding window."""
    # create dictionary to hold each layer map
    layers = {'intensity': np.zeros_like(img),
              'angle': np.zeros_like(img)}
    # loop over each sliding window
    for w in range(len(slices['s'])):
        # get oversampled window
        osw = img[slices['os'][w]]
        # get slice of sampled window
        sw = slices['s'][w]
        # acquire local FFT
        fft_full = fp.fftshift(fp.fft2((osw).astype(float)))
        fft_re = norm_image(np.abs(np.real(fft_full)))
        # filter out highest center peak
        fft_re = np.where(fft_re == 1, 0, fft_re)
        # save result
        layers['intensity'][sw] = fft_re.max()
    return layers

def map_image(img, slices):
    """Create gradient maps of an image using a sliding window."""
    # create dictionary to hold each layer map
    keys = [
        'intensity',
        'gradient_mag',
        #'gradient_phase'
        #'variance',
        ]
    layers = {k: np.zeros_like(img).astype(float) for k in keys}
    # loop over each sliding window
    for w in range(len(slices['s'])):
        # get oversampled window
        osw = img[slices['os'][w]]
        # get statistics of oversampled window
        gradient_mag = get_gradient_info(osw)
        # get slice of sampled window
        sw = slices['s'][w]
        # save statistics of sampled image window
        layers['intensity'][sw] = np.mean(osw)
        layers['gradient_mag'][sw] = gradient_mag
        #layers['gradient_phase'][sw] = gradient_phase
        #layers['variance'][sw] = np.var(osw)

    layers['intensity_mask'] = np.where(
        norm_image(layers['intensity']) < 0.6, 1, 0)
    
    return layers

def print_info_message():
    """Print a message if this module is run by itself."""
    print('\n\n=======================================================')
    print('"microscopy_methods.py":')
    print('-------------------------------------------------------')
    print('This module contains functions for analysis of ')
    print('a stack of ".dm3" format microscopy images.')
    print('To perform the analysis, run the Jupyter Noteboook')
    print('called "read_stack.ipynb", which will call this module.')
    print('The full repository is located at:')
    print('https://github.com/ericmuckley/microscopy_analysis')
    print('=======================================================')      

if __name__ == '__main__':
    print_info_message()
