# Microscopy analysis

This notebook and associated methods perform analysis of microscopy images in **.dm3** format. The notebook can be viewed at https://nbviewer.jupyter.org/github/ericmuckley/microscopy_analysis/blob/master/read_stack.ipynb


### Methods

The procedure used for identification of crystallite domain boundaries:
1. high frequency noise in the images is reduced by applying a Gaussian filter
2. the filtered images are rescaled to lower resolution to enable faster processing
3. a K-means clustering algorithm is applied to the filtered, rescaled image in order to distinguish pixels which belong to the substrate those which which belong to crystallite domains
4. the boundary of each cluster is extracted
5. the area inside each cluster boundary is calculated

The procedure used for applying statistics over the domains:
1. a sliding window rasters over each image and extracts satstistical information about crystallite phases and orientation
2. in progress...


### Description of files

* **read_stack.ipynb**: Jupyter Notebook which performs analysis of microscopy images
* **microscopy_methods.py**: python module which contains functions for opening image files and implementing sliding analysis window

