import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
import cv2
import math

from astropy.io import fits
from alive_progress import alive_bar
from time import sleep
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from tqdm import tqdm
from scipy import ndimage
from skimage.measure import regionprops

import matplotlib.cm as cm
import cmasher as cmr

#np.set_printoptions(threshold=np.inf)

###################################################################################################


class EXTRACT:

    def __init__(self, fits_file=None, distance=None, sigma=5, y_c=None, x_c=None, **kwargs):        
                                
        self._fits_info(fits_file)
        self._compute_geometric_parameters()
        #self._trace_surface()
        #self._extract_surface_info()

        return
    

    def _fits_info(self, fits_file):

        self.filename = os.path.normpath(os.path.expanduser(fits_file))
        hdu = fits.open(self.filename)

        # header
        self.source = hdu[0].header['OBJECT']
        
        self.nx = hdu[0].header['NAXIS1']
        self.ny = hdu[0].header['NAXIS2']
        self.nv = hdu[0].header['NAXIS3']
        self.dv = hdu[0].header['CDELT3']

        self.pixelscale = hdu[0].header['CDELT2'] * 3600
        self.bmaj = hdu[0].header['BMAJ'] * 3600 
        self.bmin = hdu[0].header['BMIN'] * 3600
        self.bpa = hdu[0].header['BPA']

        self.unit = hdu[0].header['BUNIT']
        self.restfreq = hdu[0].header['RESTFRQ']

        if hdu[0].header['CTYPE3'] == 'VRAD':
            self.velocity = hdu[0].header['CRVAL3'] + (np.arange(hdu[0].header['NAXIS3']) * hdu[0].header['CDELT3'])
            if hdu[0].header['CUNIT3'] == 'km/s':
                # convert to m/s
                self.velocity *= 1000
                self.dv *= 1000
        else:
            raise ValueError("Velocity type is unknown:", hdu[0].header['CTYPE3'])

        # data
        self.image = np.ma.masked_array(hdu[0].data)
        if self.image.ndim == 4:
            self.image = self.image[0,:,:,:]
        elif self.image.ndim != 3:
            raise ValueError("Unknown number of dimensions:", self.image.ndim)

        hdu.close()
        
        
    def _compute_geometric_parameters(self):

        print('GETTING GEOMETRIC PROPERTIES')
        
        cube = np.nan_to_num(self.image)
        rms = np.nanstd(cube)

        # systemic velocity
        print('computing systemic velocity')

        line_profile = np.nansum(cube, axis=(1,2))
        peaks, properties = _peak_finder(-line_profile)
        vsyst_idx = peaks[np.argmin(-line_profile[peaks])] 
        vsyst = self.velocity[0] + (self.dv * vsyst_idx)
        
        print('systemic velocity (m/s) =', vsyst)
        
        self.vsyst = vsyst
        
        # velocity map
        print('computing velocity map')
        '''
        ## peak velocity
        peak_array = np.zeros([self.nx,self.ny], dtype='int')
        for i, k in tqdm(itertools.product(range(self.nx), range(self.ny)), total=self.nx*self.ny):
            peaks, properties = _peak_finder(cube[:,i,k], width=5, height=5*rms, prominence=5*rms)
            if len(peaks) > 0:
                max_peak = peaks[np.argmax(properties["peak_heights"])]
                peak_array[i,k] = max_peak
                
        vpeak = self.velocity[peak_array]
        vpeak[vmap == self.velocity[0]] = None
        vpeak -= vsyst

        self.vpeak = vpeak
        '''
        ## moment 1
        threshold_cube = cube.copy()
        threshold_cube[cube < 3*rms] = None

        M0 = np.trapz(threshold_cube, dx=self.dv, axis=0)
        
        int_component = self.velocity[:,None,None] * threshold_cube[:,:,:]
        M1 = np.array(np.trapz(int_component, dx=self.dv, axis=0) / M0)
        M1[M1 == 0] = None

        #box_blur_kernel = np.array([[1/9, 1/9, 1/9],
    #                        [1/9, 1/9, 1/9],
    #                        [1/9, 1/9, 1/9]])
    box_blur_kernel = np.array(np.ones([int(beam),int(beam)])/np.square(int(beam)))

    bgrad_img = ndimage.convolve(grad_img, box_blur_kernel)
        
        M1 -= vsyst

        self.M1 = M1
        '''
        ## moment 2
        M2 = int_component = threshold_cube[:,:,:] * (self.velocity[:,None,None] - M1)**2
        M2 = np.array(np.trapz(int_component, dx=self.dv, axis=0) / M0)
        M2[M2 == 0] = None
        
        self.M2 = M2
        '''
        # center of mass
        print('computing center of mass')
        
        vmap = self.M1.copy()
        #vmap[abs(self.M1) < np.nanpercentile(abs(self.M1),[2])] = None
        #vmap[abs(self.M1) > np.nanpercentile(abs(self.M1),[98])] = None
        print('image size =', vmap.shape)
        
        cm_coords = _center_of_mass(vmap, beam=self.bmaj/self.pixelscale)
        print('center coordinates (pixels) =', cm_coords)

        self.cm_coords = cm_coords

        # plotting
        
        plt.imshow(self.M1, origin='lower', cmap=cm.RdBu_r, vmin=np.nanpercentile(M1,[2]), vmax=np.nanpercentile(M1,[98]))
        plt.xlim(cm_coords[1]-50, cm_coords[1]+50)
        plt.ylim(cm_coords[0]-50, cm_coords[1]+50)
        plt.colorbar(extend='both')
        plt.plot(self.cm_coords[1], self.cm_coords[0], marker='.', color='black', markersize=10)
        
        #plt.plot(line_profile, color='blue')
        #plt.plot(vsyst_idx, line_profile[vsyst_idx], '.', markersize=5, color='black')
        
        plt.show()


def _peak_finder(profile, height=None, threshold=None, distance=None, prominence=None, width=None):

    peaks, properties = find_peaks(profile, height=height, threshold=threshold, distance=distance, prominence=prominence, width=width)
    return peaks, properties


def _center_of_mass(img, beam=None):
    
    '''
    img = np.nan_to_num(img)
    #img[img != 0] = 1

    M = cv2.moments(img)

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    x_coord = cx
    y_coord = cy
    '''
    '''
    img = img / np.nansum(np.nansum(img))

    dx = np.nansum(img, 1)
    dy = np.nansum(img, 0)

    (X, Y) = img.shape
    
    x_coord = np.nansum(dx * np.arange(X))
    y_coord = np.nansum(dy * np.arange(Y))
    '''
    '''
    #img = np.nan_to_num(img)
    #img[img != 0] = 1
    
    #img -= np.mean(img)
    #img = abs(img)
    #img = (img - np.nanmin(img))/(np.nanmax(img) - np.nanmin(img))
    #img = img * (10 - 0.1) + 0.1
    #img = np.exp(img)
    #threshold = np.nanpercentile(img, [90])
    #img[img < threshold] = None
    
    normalizer = np.nansum(img)
    grids = np.ogrid[[slice(0, i) for i in img.shape]]

    results = [np.nansum(img * grids[dir].astype(float)) / normalizer
               for dir in range(img.ndim)]

    if np.isscalar(results[0]):
        centre_of_mass = tuple(results)
    else:
        centre_of_mass = [tuple(v) for v in np.array(results).T]

    x_coord = centre_of_mass[1]
    y_coord = centre_of_mass[0]
    '''
    blue = np.unravel_index(np.nanargmin(img), img.shape)
    red = np.unravel_index(np.nanargmax(img), img.shape)
    print(blue, red)
    
    x_coord = np.mean([blue[1], red[1]])
    y_coord = np.mean([blue[0], red[0]])
    print(x_coord, y_coord)
    
    y,x = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
    radius = np.hypot(y-y_coord,x-x_coord)
    mask = radius <= (5*beam)

    masked_img = img.copy()
    masked_img[~mask] = None
    masked_img = abs(masked_img)

    dx, dy = np.gradient(masked_img, edge_order=2)
    grad_img = sobel=np.hypot(dy,dx)
    skip = (slice(None, None, 1), slice(None, None, 1))

    #box_blur_kernel = np.array([[1/9, 1/9, 1/9],
    #                        [1/9, 1/9, 1/9],
    #                        [1/9, 1/9, 1/9]])
    box_blur_kernel = np.array(np.ones([int(beam),int(beam)])/np.square(int(beam)))

    bgrad_img = ndimage.convolve(grad_img, box_blur_kernel)
    y_coord, x_coord = np.unravel_index(np.nanargmax(bgrad_img), grad_img.shape)
    print(y_coord, x_coord)
    
    #plt.imshow(img, origin='lower', cmap=cm.RdBu_r, vmin=np.nanmin(img), vmax=np.nanmax(img))
    #plt.colorbar()#extend='both')
    #plt.quiver(x[skip], y[skip], dx[skip].T, dy[skip].T)
    #plt.plot(x_coord, y_coord, marker='.', color='magenta', markersize=10)
    #plt.xlim(x_coord-50, x_coord+50)
    #plt.ylim(y_coord-50, y_coord+50)
    #plt.show()
    #sys.exit()
    
    '''
    #img = np.nan_to_num(img)
    #img[img != 0] = 1
    
    total = np.nansum(img)
    x_coord = np.matmul(np.nansum(img, axis=0), np.arange(0,img.shape[1])) / total
    y_coord = np.matmul(np.nansum(img, axis=1), np.arange(0,img.shape[0])) / total

    #plt.imshow(img, origin='lower', cmap=cm.RdBu_r)
    #plt.colorbar(extend='both')
    #plt.show()
    #sys.exit()
    '''
    
    '''
    # refining the centre point. inprogress (may not be necessary)
    y,x = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
    radius = np.hypot(y-y_coord,x-x_coord)
    mask = radius <= (2*beam)

    masked_img = img.copy()
    masked_img[~mask] = None

    refined_coords = np.unravel_index(np.nanargmin(masked_img, axis=None), masked_img.shape) #np.nanargmin(masked_img, axis=1)
    x_coord_refined = refined_coords[1]
    y_coord_refined = refined_coords[0]
    '''
    
    return [y_coord,x_coord]


#vmap = self.velocity[np.argmax(self.image, axis=0)]




