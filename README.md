# Microscopy analysis

This notebook and associated methods perform analysis of microscopy images in **.dm3** format. The notebook can be viewed at https://nbviewer.jupyter.org/github/ericmuckley/microscopy_analysis/blob/master/read_stack.ipynb


### Methods

There is a significant difference in images which contain assemblies of de-wet particles and images which show film-like samples with mostly amorphous material. Detection of domains is performed differently depending on the type of image:

**For images with de-wet particle assemblies:**
1. high frequency noise in the images is reduced by applying a Gaussian filter
2. the filtered images are rescaled to lower resolution to enable faster processing
3. a K-means clustering algorithm is applied to the filtered, rescaled image in order to distinguish pixels which belong to the substrate those which which belong to large domains
4. the boundary of each cluster is extracted
5. the area inside each cluster boundary is calculated


**For images with amorphous film-like material:**
1. a sliding window is used to raster over each image and extract local FFT information
2. the magnitude of the highest FFT peak is used as an estimate for crystallinity of the area inside the sampling window
3. regions with FFT strength which is above a specific threshold are regarded as 'crystalline'
4. the boundaries and areas of crystalline areas are calculated


Final results are shown in the last plots in the **read_stack.ipynb** notebook.

### Description of files

* **read_stack.ipynb**: Jupyter Notebook which performs analysis of microscopy images
* **microscopy_methods.py**: python module which contains functions for opening image files and implementing sliding analysis window

