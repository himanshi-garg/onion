import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.backends.backend_pdf
import itertools

from astropy.io import fits
from scipy.optimize import curve_fit
from scipy import stats
from scipy import signal
from scipy import ndimage
from tqdm import tqdm
from skimage import color, morphology
from skimage.transform import warp_polar
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
from matplotlib.patches import Ellipse
from scipy import interpolate

import matplotlib.cm as cm
import cmasher as cmr

np.set_printoptions(threshold=np.inf)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

###################################################################################################

class EXTRACT:

    def __init__(self, fits_file, dist=None, cx=None, cy=None, com_unit=None, inc=None, PA=None, vsyst=None):        

        if inc == None:
            raise ValueError("Need to specify source inclination [degrees]")
        else:
            self.inc = inc

        if cx is not None and cy is not None:
            if com_unit is None:
                raise ValueError("Need to specify units for centre of mass coordinates, either 'arcseconds' or 'pixels'")
            
        self.com_unit = com_unit
        self.dist = dist
        
        self._fits_info(fits_file)
        self._compute_geometric_parameters(cx=cx, cy=cy, com_unit=com_unit, PA=PA, vsyst=vsyst)
        self._trace_surface()
        self._extract_surface()
        self._plots()

        return
    

    def _fits_info(self, fits_file):

        self.filename = os.path.normpath(os.path.expanduser(fits_file))        
        hdu = fits.open(self.filename)

        # header
        self.source = hdu[0].header['OBJECT']
        print('OBJECT =', self.source)
        
        self.nx = hdu[0].header['NAXIS1']
        self.ny = hdu[0].header['NAXIS2']
        self.nv = hdu[0].header['NAXIS3']
        self.dv = hdu[0].header['CDELT3']

        self.pixelscale = hdu[0].header['CDELT2'] * 3600
        self.bmaj = hdu[0].header['BMAJ'] * 3600 
        self.bmin = hdu[0].header['BMIN'] * 3600
        self.bpa = hdu[0].header['BPA']
        self.aunit = hdu[0].header['CUNIT1']
        self.iunit = hdu[0].header['BUNIT'] 
        self.restfreq = hdu[0].header['RESTFRQ']
        if self.aunit == 'rad':
            self.bpa = np.rad2deg(self.bpa)
        elif self.aunit != 'deg':
            raise ValueError("unknown angle units:", self.aunit)
        self.imgcx = hdu[0].header['CRPIX1']
        self.imgcy = hdu[0].header['CRPIX2']

        if hdu[0].header['CTYPE3'] == 'VRAD' or hdu[0].header['CTYPE3'] == 'VELO-LSR':
            self.velocity = hdu[0].header['CRVAL3'] + (np.arange(hdu[0].header['NAXIS3']) * hdu[0].header['CDELT3'])
            if hdu[0].header['CUNIT3'] == 'km/s':
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
        
        
    def _compute_geometric_parameters(self, cx=None, cy=None, com_unit=None, PA=None, vsyst=None):
        
        print('EXTRACTING GEOMETRIC PROPERTIES')
        
        cube = np.nan_to_num(self.image)
        rms = np.nanstd(cube)
        print(f'rms [{self.iunit}] =', rms)
        self.cube = cube
        self.rms = rms

        # systemic velocity
        line_profile = np.nansum(cube, axis=(1,2))
        if vsyst is not None:
            vsyst, vsyst_idx = vsyst, (vsyst - self.velocity[0])/self.dv
            self.gauss_fit = None
        else:
            vsyst, vsyst_idx, gauss_fit = _systemic_velocity(line_profile, nchans=self.nv, v0=self.velocity[0], dv=self.dv)
            self.gauss_fit = gauss_fit
        
        print('systemic velocity (m/s) =', vsyst)
        
        self.vsyst = vsyst
        self.vsyst_idx = vsyst_idx
        self.line_profile = line_profile
        
        ## moment 1
        threshold_cube = cube.copy()
        threshold_cube[cube < 2*rms] = None

        M0 = np.trapz(threshold_cube, dx=self.dv, axis=0)
        
        int_component = self.velocity[:,None,None] * threshold_cube[:,:,:]
        M1 = np.array(np.trapz(int_component, dx=self.dv, axis=0) / M0)
        M1[M1 == 0] = None
        M1 -= vsyst

        self.M1 = M1
        
        # center of mass        
        vmap = self.M1.copy()
        beam_pix = self.bmaj/self.pixelscale
        com, M1p = _center_of_mass(vmap, beam=beam_pix)
        if cx is not None and cy is not None and com_unit is not None:
            if com_unit == 'pixels':
                com_img = np.zeros([int(2*cy),int(2*cx)])
                com_img[int(cy),int(cx)] = 1
                cy, cx = np.unravel_index(np.nanargmax(com_img), com_img.shape)
                com = [cy, cx]
            elif com_unit == 'arcseconds':
                cy = int(float(cy)/self.pixelscale + self.imgcy)
                cx = int(float(cx)/self.pixelscale + self.imgcx)
                com_img = np.zeros([2*cy,2*cx])
                com_img[cy,cx] = 1
                cy, cx = np.unravel_index(np.nanargmax(com_img), com_img.shape)
                com = [cy, cx]
            else:
                raise ValueError("centre of mass units need to be either 'arcseconds' or 'pixels'")

        print('center coordinates (pixels) =', com)

        self.com = com
        self.M1p = M1p

        # position angle
        if PA is not None:
            PA = float(PA)
            _, Rout = _position_angle(self.M1p, cx=self.com[1], cy=self.com[0], beam=beam_pix)
        else:
            PA, Rout = _position_angle(self.M1p, cx=self.com[1], cy=self.com[0], beam=beam_pix)
            
        print('position angle (degrees) =', PA)

        self.PA = PA
        self.Rout = Rout
        
        
    def _trace_surface(self):

        print('TRACING LAYERS')

        trms = 0.1*np.nanmax(self.line_profile)
        channels = np.arange(0,self.nv,1)
        tchans = channels[np.where(self.line_profile > trms)]
        beam = self.bmaj/self.pixelscale

        surfaces = np.full([self.nv,(self.Rout+1).astype(np.int),4,4], None)
        rsurfaces = np.full([self.nv,(self.Rout+1).astype(np.int),4,4], None)

        phi = np.deg2rad(np.arange(0,360,1))
        phi[phi > np.pi] -= 2*np.pi
        rad = np.arange(1, self.Rout.astype(np.int), 1)
        
        for i in tqdm(tchans):

            chan = self.cube[i,:,:]
            chan_rot = _rotate_disc(chan, angle=self.PA+90, cx=self.com[1], cy=self.com[0])
            polar = warp_polar(chan_rot, center=(self.com[0],self.com[1]), radius=self.Rout)
                    
            for k in rad:
                
                annuli = polar[:,k]
                annuli_wrapped = np.concatenate((annuli, annuli[:90]))

                peaks, properties = _peak_finder(annuli_wrapped, height=5*self.rms, distance=beam, prominence=3*self.rms, width=0.5*beam)
                peaks[peaks >= 360] -= 360
                peaks = np.unique(peaks)
                sorted_peaks = peaks[np.argsort(annuli[peaks])][::-1]

                if len(sorted_peaks) >= 2:
                    if np.all(phi[sorted_peaks[:2]] > np.pi/2) or np.all(phi[sorted_peaks[:2]] < -np.pi/2):
                        phi0 = np.min(phi[sorted_peaks[:2]])
                        phi1 = np.max(phi[sorted_peaks[:2]])
                    else:
                        phi0 = np.max(phi[sorted_peaks[:2]])
                        phi1 = np.min(phi[sorted_peaks[:2]])
                    up1 = _pol2cart(k, phi0, cx=self.com[1], cy=self.com[0])
                    up2 = _pol2cart(k, phi1, cx=self.com[1], cy=self.com[0])
                    
                    surfaces[i,k,0,:] = np.concatenate((up1,[phi0],[k.astype(float)]))
                    surfaces[i,k,1,:] = np.concatenate((up2,[phi1],[k.astype(float)]))
                    rsurfaces[i,k,0,:] = np.concatenate((up1,[phi0],[k.astype(float)]))
                    rsurfaces[i,k,1,:] = np.concatenate((up2,[phi1],[k.astype(float)]))
                    
                if len(sorted_peaks) >= 4:
                    if np.all(phi[sorted_peaks[2:4]] > np.pi/2) or np.all(phi[sorted_peaks[2:4]] < -np.pi/2):
                        phi2 = np.min(phi[sorted_peaks[2:4]])
                        phi3 = np.max(phi[sorted_peaks[2:4]])
                    else:
                        phi2 = np.max(phi[sorted_peaks[2:4]])
                        phi3 = np.min(phi[sorted_peaks[2:4]])
                    lo1 = _pol2cart(k, phi2, cx=self.com[1], cy=self.com[0])
                    lo2 = _pol2cart(k, phi3, cx=self.com[1], cy=self.com[0])
                    
                    surfaces[i,k,2,:] = np.concatenate((lo1,[phi2],[k.astype(float)]))
                    surfaces[i,k,3,:] = np.concatenate((lo2,[phi3],[k.astype(float)]))
                    rsurfaces[i,k,2,:] = np.concatenate((lo1,[phi2],[k.astype(float)]))
                    rsurfaces[i,k,3,:] = np.concatenate((lo2,[phi3],[k.astype(float)]))

            grad0 = np.full([4], None)
            cav = np.full([4,len(surfaces[i,:,0,3])], False)
            dr = np.full([4,len(surfaces[i,:,0,3])], None)
            
            for vx in range(4):
                if surfaces[i,:,vx,:].any():

                    layerx = np.hstack((self.com[1],surfaces[i,:,vx,1][np.where(surfaces[i,:,vx,1] != None)]))
                    layery = np.hstack((self.com[0],surfaces[i,:,vx,0][np.where(surfaces[i,:,vx,0] != None)]))
                    layerr = np.hstack((0.,surfaces[i,:,vx,3][np.where(surfaces[i,:,vx,3] != None)]))
                    if len(layerr) <= 3:
                        continue

                    ggrad = np.full([len(layery)], 1.)
                    ggrad[0] = abs(-self.com[1] / self.com[0])
                    gvol_factor = np.full([len(layery)], 0.)
                    
                    for lx in range(1, len(layery)):
                        rs = abs(layerr[lx] - layerr[lx-1]) 
                        ggrad[lx] = abs(-layerx[lx] / layery[lx])
                        gvol_factor[lx] = abs(np.exp(abs(np.log(ggrad[lx] / ggrad[lx-1]))) - 1)
                        
                        gap = True if lx == 1 else False
                        if rs > 1 and lx > 1:
                            Int_gap = polar[:,[int(layerr[lx-1]+1), int(layerr[lx])]]
                            Int_gapf = Int_gap[np.where(Int_gap > 3*self.rms)]
                            if len(Int_gapf) == 0:
                                gap = True
                            else:
                                avg_Int = np.mean(Int_gapf) / self.rms
                                dx = layerx[lx] - layerx[lx-1]
                                dy = layery[lx] - layery[lx-1]
                                sep = np.hypot(dx,dy)
                                fac = abs(sep - rs)
                                gap = True if (avg_Int < 5 and fac < 5) else False

                        gvol_factor[lx] /= rs if gap == True else 1
                        cav[vx,int(layerr[lx])] = gap
                        dr[vx,int(layerr[lx])] = rs
                        
                    gvol_median = np.median(gvol_factor[2:])
                    Q1 = np.nanpercentile(gvol_factor[2:], [16])
                    Q3 = np.nanpercentile(gvol_factor[2:], [84])
                    iqr = Q3[0]-Q1[0]
                    std = np.std(gvol_factor[2:])
                    threshold = 3 * np.sqrt(std/iqr) * iqr
                    
                    for k in rad:
                        if surfaces[i,k,vx,:].all():

                            grad = abs(-surfaces[i,k,vx,1] / surfaces[i,k,vx,0])

                            if grad0[vx] is None:
                                grad0[vx] = grad
                            else:
                                vol_factor = abs(np.exp(abs(np.log((grad / grad0[vx])))) - 1)
                                vol_factor /= dr[vx,k] if cav[vx,k] == True else 1
                                vol = (vol_factor - gvol_median) / threshold
                                
                                if vol > 1 and vol_factor > 0.01:
                                    surfaces[i,k,vx,:] = None
                                else:
                                    grad0[vx] = grad
                                             
            for k in rad:
                if np.any(surfaces[i,k,:2,:] == None):
                    surfaces[i,k,:2,:] = None
                if np.any(surfaces[i,k,2:4,:] == None):
                    surfaces[i,k,2:4,:] = None
        
        self.surfaces = surfaces
        self.rsurfaces = rsurfaces
        self.tchans = tchans


    def _extract_surface(self):

        print('EXTRACTING SURFACES')
        
        mid = np.full([self.nv,2*(self.Rout+1).astype(np.int),2,2], None)
        top = np.full([self.nv,2*(self.Rout+1).astype(np.int),2,2], None)
        bot = np.full([self.nv,2*(self.Rout+1).astype(np.int),2,2], None)
        
        vchans = []
        hslope = []

        for i in tqdm(self.tchans):
            for vx in range(2):
                if self.surfaces[i,:,(vx+1)*vx:(vx*2)+2,:].any():
                    phi0 = self.surfaces[i,:,(vx+1)*vx:(vx*2)+2,2][np.where(self.surfaces[i,:,(vx+1)*vx:(vx*2)+2,2] != None)]
                    if len(phi0[np.where(abs(abs(phi0)-np.pi/2) < np.deg2rad(15))]) > 0.5*len(phi0):
                        continue
                    else:
                        x0 = self.surfaces[i,:,(vx+1)*vx,1][np.where(self.surfaces[i,:,(vx+1)*vx,1] != None)]
                        x1 = self.surfaces[i,:,(vx*2)+1,1][np.where(self.surfaces[i,:,(vx*2)+1,1] != None)]
                        y0 = self.surfaces[i,:,(vx+1)*vx,0][np.where(self.surfaces[i,:,(vx+1)*vx,0] != None)]
                        y1 = self.surfaces[i,:,(vx*2)+1,0][np.where(self.surfaces[i,:,(vx*2)+1,0] != None)]
                        b0 = np.max([np.min(x0),np.min(x1)])
                        b1 = np.min([np.max(x0),np.max(x1)])
                        if b0 > b1:
                            continue
                        x = np.hstack((x0,x1))
                        x = x[(x >= b0) & (x <= b1)]
                        x = np.sort(x)[::-1]
                        if len(x) == 0:
                            continue
                        else:
                            vchans.append(i)
                            idx1 = np.argsort(x0)
                            idx2 = np.argsort(x1)
                            yf0 = np.interp(x.astype(float), x0[idx1].astype(float), y0[idx1].astype(float))
                            yf1 = np.interp(x.astype(float), x1[idx2].astype(float), y1[idx2].astype(float))
                            yfm = np.mean([yf0,yf1], axis=0)
                            hslope.append(np.nanmean(yfm))

                            mid[i,:len(x),vx,0], mid[i,:len(x),vx,1] = yfm, x
                            top[i,:len(x),vx,0], top[i,:len(x),vx,1] = yf0, x
                            bot[i,:len(x),vx,0], bot[i,:len(x),vx,1] = yf1, x
                else:
                    continue

        if np.all(self.surfaces == None):
            self.nearside = None
        else:
            if np.nanmean(hslope - self.com[0]) < 0:
                self.nearside = self.PA + 90
                hdir = -1
            else:
                self.nearside = self.PA - 90
                hdir = 1

            self.nearside += 360 if self.nearside < -90 else -360 if self.nearside > 270 else 0

        sR = np.full([self.nv,2*(self.Rout+1).astype(np.int),2], None)
        sH = np.full([self.nv,2*(self.Rout+1).astype(np.int),2], None)
        sV = np.full([self.nv,2*(self.Rout+1).astype(np.int),2], None)
        sI = np.full([self.nv,2*(self.Rout+1).astype(np.int),2], None)
        
        for i in tqdm(vchans):
            for vx in range(2):
                if self.surfaces[i,:,(vx+1)*vx:(vx*2)+2,:].any():
                    mid0x = mid[i,:,vx,1][np.where(mid[i,:,vx,1] != None)]
                    mid0y = mid[i,:,vx,0][np.where(mid[i,:,vx,0] != None)]
                    top0x = top[i,:,vx,1][np.where(top[i,:,vx,1] != None)]
                    top0y = top[i,:,vx,0][np.where(top[i,:,vx,0] != None)]
                    bot0x = bot[i,:,vx,1][np.where(bot[i,:,vx,1] != None)]
                    bot0y = bot[i,:,vx,0][np.where(bot[i,:,vx,0] != None)]
                    if len(mid0x) == 0:
                        continue
                    else:
                        h = abs(mid0y - self.com[0]) / np.sin(np.deg2rad(self.inc))
                        h[np.where(h < 0)] = None
                        if hdir == -1:
                            if vx == 0:
                                r = np.hypot(bot0x.astype(float) - self.com[1], (bot0y.astype(float) - mid0y.astype(float)) / np.cos(np.deg2rad(self.inc)))
                                v = (self.velocity[i] - self.vsyst) * r / ((bot0x.astype(float) - self.com[1]) * np.sin(np.deg2rad(self.inc)))
                            elif vx == 1:
                                r = np.hypot(top0x.astype(float) - self.com[1], (top0y.astype(float) - mid0y.astype(float)) / np.cos(np.deg2rad(self.inc)))
                                v = (self.velocity[i] - self.vsyst) * r / ((top0x.astype(float) - self.com[1]) * np.sin(np.deg2rad(self.inc)))
                        else:
                            if vx == 0:
                                r = np.hypot(top0x.astype(float) - self.com[1], (top0y.astype(float) - mid0y.astype(float)) / np.cos(np.deg2rad(self.inc)))
                                v = (self.velocity[i] - self.vsyst) * r / ((top0x.astype(float) - self.com[1]) * np.sin(np.deg2rad(self.inc)))
                            elif vx == 1:
                                r = np.hypot(bot0x.astype(float) - self.com[1], (bot0y.astype(float) - mid0y.astype(float)) / np.cos(np.deg2rad(self.inc)))
                                v = (self.velocity[i] - self.vsyst) * r / ((bot0x.astype(float) - self.com[1]) * np.sin(np.deg2rad(self.inc)))
                                
                        v[np.where(v < 0)] = None
                        chan = self.cube[i,:,:]
                        chan_rot = _rotate_disc(chan, angle=self.PA+90, cx=self.com[1], cy=self.com[0])
                        I = np.mean([chan_rot[top0y.astype(int),top0x.astype(int)], chan_rot[bot0y.astype(int),bot0x.astype(int)]], axis=0)

                        h[np.where(v == None)] = None
                        v[np.where(h == None)] = None
                        r[np.where(h == None)] = None
                        r[np.where(v == None)] = None
                        I[np.where(h == None)] = None
                        I[np.where(v == None)] = None

                        sR[i,:len(r),vx] = r
                        sH[i,:len(r),vx] = h
                        sV[i,:len(r),vx] = v
                        sI[i,:len(r),vx] = I
                else:
                    continue         

        self.mid = mid
        self.sR = sR
        self.sH = sH
        self.sV = sV
        self.sI = sI


    def _plots(self):
        
        print('PLOTTING FIGURES FOR GEOMETRIC PROPERTIES')
        
        pdf = matplotlib.backends.backend_pdf.PdfPages(self.filename+'_gp.pdf')

        for i in tqdm(range(3), total=3):
            fig, ax = plt.subplots(figsize=(6,6))

            if i == 0:
                # line profile for systemic velocity
                ax.plot(self.line_profile, color='black', linewidth=1.0)
                if self.gauss_fit is not None:
                    ax.plot(self.gauss_fit, color='firebrick', linestyle='--', linewidth=1.0)
                ax.plot(self.vsyst_idx, self.line_profile[self.vsyst_idx], '.', markersize=10, color='firebrick', label=f'vlsr = {self.vsyst:.3f} m/s')
                ax.legend(loc='upper right', fontsize=7)
                ax.set(xlabel = 'channel number', ylabel = f'$\Sigma$Flux [{self.iunit}]', title='systemic velocity')
                axins = inset_axes(ax, width="30%", height="30%",loc='upper left')
                axins.imshow(self.cube[self.vsyst_idx,:,:], origin='lower', cmap=cmr.swamp, vmin=0, vmax=np.nanmax(self.cube[self.vsyst_idx,:,:]))
                axins.set_yticks([])
                axins.set_xticks([])
                bounds = self.Rout*1.1
                axins.set(xlim = (self.com[1]-bounds,self.com[1]+bounds), ylim = (self.com[0]-bounds,self.com[0]+bounds))
            elif i == 1:
                # velocity map for centre of mass
                fig1 = ax.imshow(self.M1/1000, origin='lower', cmap=cm.RdBu_r, vmin=np.nanpercentile(self.M1p/1000,[1]), vmax=np.nanpercentile(self.M1p/1000,[99]))
                ax.plot(self.com[1], self.com[0], marker='.', markersize=10, color='black',
                        label=f'COM (y,x): [{self.com[0]},{self.com[1]}] pixels, \n [{(self.com[0]-self.imgcy)*self.pixelscale:.3f},{(self.com[1]-self.imgcx)*self.pixelscale:.3f}] $\Delta$arcs')
                ax.set(xlabel='pixels', ylabel='pixels', title='dynamical centre')
                ax.legend(loc='upper left', fontsize=8, framealpha=1.0)
                divider = make_axes_locatable(ax)
                colorbar_cax = divider.append_axes('right', size='4%', pad=0.05)
                cbar = fig.colorbar(fig1, shrink=0.97, aspect=70, spacing='proportional', orientation='vertical', cax=colorbar_cax, extend='both')
                cbar.ax.tick_params(labelsize=8)
                cbar.ax.xaxis.set(ticks_position = 'top', label_position = 'top')
                tick_locator = ticker.MaxNLocator(nbins=5)
                cbar.locator = tick_locator
                cbar.update_ticks()
                cbar.set_label('$\Delta$v [kms$^{-1}$]', labelpad=9, fontsize=12, weight='bold')
                axins = zoomed_inset_axes(ax, self.nx/(3*4*self.bmaj/self.pixelscale), loc='upper right')
                axins.imshow(self.M1, origin='lower', cmap=cm.RdBu_r, vmin=np.nanpercentile(self.M1p,[2]), vmax=np.nanpercentile(self.M1p,[98]))
                axins.set(xlim = (self.com[1]-2*self.bmaj/self.pixelscale,self.com[1]+2*self.bmaj/self.pixelscale),
                          ylim = (self.com[0]-2*self.bmaj/self.pixelscale,self.com[0]+2*self.bmaj/self.pixelscale))
                axins.yaxis.get_major_locator().set_params(nbins=4)
                axins.xaxis.get_major_locator().set_params(nbins=4)
                plt.xticks(visible=False)
                plt.yticks(visible=False)
                mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='black', linewidth=0.5, linestyle=':')
                axins.plot(self.com[1], self.com[0], marker='.', markersize=10, color='black')
            elif i == 2:
                # velocity map for position angle
                fig1 = ax.imshow(self.M1/1000, origin='lower', cmap=cm.RdBu_r, vmin=np.nanpercentile(self.M1p/1000,[1]), vmax=np.nanpercentile(self.M1p/1000,[99]))
                ax.set(xlabel='pixels', ylabel='pixels', title='position angle')
                divider = make_axes_locatable(ax)
                colorbar_cax = divider.append_axes('right', size='4%', pad=0.05)
                cbar = fig.colorbar(fig1, shrink=0.97, aspect=70, spacing='proportional', orientation='vertical', cax=colorbar_cax, extend='both')
                cbar.ax.tick_params(labelsize=8)
                cbar.ax.xaxis.set(ticks_position = 'top', label_position = 'top')
                tick_locator = ticker.MaxNLocator(nbins=5)
                cbar.locator = tick_locator
                cbar.update_ticks()
                cbar.set_label('$\Delta$v [kms$^{-1}$]', labelpad=9, fontsize=12, weight='bold')
                major_axis = [self.com[1]-(0.4*self.nx*np.cos(np.radians(self.PA+90))), self.com[1]+(0.4*self.nx*np.cos(np.radians(self.PA+90))),
                              self.com[0]-(0.4*self.ny*np.sin(np.radians(self.PA+90))), self.com[0]+(0.4*self.ny*np.sin(np.radians(self.PA+90)))]
                minor_axis = [self.com[1]-(0.4*self.nx*np.cos(np.radians(self.PA))), self.com[1]+(0.4*self.nx*np.cos(np.radians(self.PA))),
                              self.com[0]-(0.4*self.ny*np.sin(np.radians(self.PA))), self.com[0]+(0.4*self.ny*np.sin(np.radians(self.PA)))]
                ax.plot((major_axis[0],major_axis[1]),(major_axis[2],major_axis[3]), color='black', linestyle='--', linewidth=0.5, label=f'PA = {self.PA:.3f}$^\circ$')
                if self.nearside is not None:
                    near = [self.com[1]+(0.4*self.nx*np.cos(np.radians(self.nearside+90))), self.com[0]+(0.4*self.ny*np.sin(np.radians(self.nearside+90)))]
                    ax.plot((minor_axis[0],minor_axis[1]),(minor_axis[2],minor_axis[3]), color='darkgray', linestyle='--', linewidth=0.5, label=f'nearside = {self.nearside:.3f}$^\circ$')
                    rot = self.nearside-180 if self.nearside > 90 else self.nearside
                    ax.annotate('near', xy=(near[0],near[1]), fontsize=8, color='black', rotation=rot)
                else:
                    ax.plot((minor_axis[0],minor_axis[1]),(minor_axis[2],minor_axis[3]), color='darkgray', linestyle='--', linewidth=0.5, label='minor axis')
                ax.legend(loc='best', fontsize=7, framealpha=1.0)
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        pdf.close()
        
        
        print('PLOTTING SURFACE TRACES')

        if np.any(self.surfaces):

            pdf = matplotlib.backends.backend_pdf.PdfPages(self.filename+'_traces.pdf')
            
            for i in tqdm(self.tchans, total=len(self.tchans)):
                fig, ax = plt.subplots(figsize=(6,6))

                chan = self.cube[i,:,:]
                chan_rot = _rotate_disc(chan, angle=self.PA+90, cx=self.com[1], cy=self.com[0])

                fig1 = ax.imshow(chan_rot, origin='lower', cmap=cmr.swamp, vmin=0, vmax=np.nanmax(self.cube))
                ax.plot(self.com[1], self.com[0], marker='+', markersize=10, color='white')
                ax.set(xlabel='pixels', ylabel='pixels')

                #ax.plot(self.rsurfaces[i,:,0,1], self.rsurfaces[i,:,0,0], '.', markersize=2, color='cyan')
                #ax.plot(self.rsurfaces[i,:,1,1], self.rsurfaces[i,:,1,0], '.', markersize=2, color='cyan')
                #ax.plot(self.rsurfaces[i,:,2,1], self.rsurfaces[i,:,2,0], '.', markersize=2, color='cyan')
                #ax.plot(self.rsurfaces[i,:,3,1], self.rsurfaces[i,:,3,0], '.', markersize=2, color='cyan')
            
                ax.plot(self.surfaces[i,:,0,1], self.surfaces[i,:,0,0], '.', markersize=2, color='darkorange', label='upper surface')
                ax.plot(self.surfaces[i,:,1,1], self.surfaces[i,:,1,0], '.', markersize=2, color='darkorange')
                ax.plot(self.surfaces[i,:,2,1], self.surfaces[i,:,2,0], '.', markersize=2, color='crimson', label='lower surface')
                ax.plot(self.surfaces[i,:,3,1], self.surfaces[i,:,3,0], '.', markersize=2, color='crimson')
                ax.plot(self.mid[i,:,0,1], self.mid[i,:,0,0], '.', markersize=2, color='moccasin', label='mid upper surface')
                ax.plot(self.mid[i,:,1,1], self.mid[i,:,1,0], '.', markersize=2, color='pink', label='mid lower surface')
                ax.legend(loc='upper right', fontsize=7, markerscale=2)
                bounds = self.Rout*1.1
                ax.set(xlim = (self.com[1]-bounds,self.com[1]+bounds), ylim = (self.com[0]-bounds,self.com[0]+bounds))
            
                beam = Ellipse(xy=(self.com[1]-self.Rout,self.com[0]-self.Rout), width=self.bmin/self.pixelscale,
                               height=self.bmaj/self.pixelscale, angle=-self.bpa, fill=True, color='white')
                ax.add_patch(beam)
                
                divider = make_axes_locatable(ax)
                colorbar_cax = divider.append_axes('right', size='4%', pad=0.05)
                cbar = fig.colorbar(fig1, shrink=0.97, aspect=70, spacing='proportional', orientation='vertical', cax=colorbar_cax)
                cbar.ax.tick_params(labelsize=8)
                cbar.ax.xaxis.set(ticks_position = 'top', label_position = 'top')
                tick_locator = ticker.MaxNLocator(nbins=5)
                cbar.locator = tick_locator
                cbar.update_ticks()
                cbar.set_label(f'flux [{self.iunit}]', labelpad=9, fontsize=12, weight='bold')
        
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
        
            pdf.close()
            
        else:
            print('no valid surface traces')
        

        print('PLOTTING SURFACES')

        if np.any(self.sR):

            self.sH[np.where(self.sH != None)] *= self.pixelscale
            self.sR[np.where(self.sR != None)] *= self.pixelscale
            ylabels = ['z [arcsec]', 'v [m/s]', f'flux [{self.iunit}]']
            xlabel = ['r [arcsec]']

            if self.dist is not None:
                self.sH *= self.dist
                self.sR *= self.dist
                ylabels[0] = ['z [au]']
                xlabel = ['r [au]']
        
            surfs = {0: self.sH, 1: self.sV, 2: self.sI}            
        
            pdf = matplotlib.backends.backend_pdf.PdfPages(self.filename+'_profiles.pdf')
            for k in tqdm(range(3)):

                fig, ax = plt.subplots(figsize=(6,6))
                if self.sR[:,:,0].any():
                    ax.plot(self.sR[:,:,0].flatten(), surfs[k][:,:,0].flatten(), 'o', markersize=4, color='lightsteelblue', alpha=0.4, fillstyle='none', label='upper surface')
                    x1 = self.sR[:,:,0][self.sR[:,:,0] != None].flatten()
                    y1 = surfs[k][:,:,0][surfs[k][:,:,0] != None].flatten()
                    idx = np.argsort(x1)
                    x1, y1 = x1[idx], y1[idx]
                    win = signal.windows.hann(int(self.bmaj/self.pixelscale))
                    y1 = signal.convolve(y1, win, mode='same') / sum(win)
                    bins, _, _ = stats.binned_statistic(x1.astype(float),[x1.astype(float),y1.astype(float)], bins=np.round(self.Rout))
                    ax.plot(bins[0,:], bins[1,:], '.', markersize=8, color='navy', markeredgecolor='whitesmoke', markeredgewidth=0.3, label='avg. upper surface')

                    if k == 0:
                        with open(self.filename+'_traces_original.csv', 'w') as f:
                            output_traces = np.column_stack((self.sR[:,:,0].flatten().tolist(), self.sH[:,:,0].flatten().tolist()))
                            np.savetxt(f, output_traces, header='r, z', fmt='%s')
                        with open(self.filename+'_traces_binned.csv', 'w') as f:
                            output_traces = np.column_stack((bins[0,:].tolist(), bins[1,:].tolist()))
                            np.savetxt(f, output_traces, header='binned_r, binned_z', fmt='%s')

                    if k == 0:
                        try:
                            r = bins[0,:][np.isfinite(bins[0,:])].astype(float)
                            z = bins[1,:][np.isfinite(bins[1,:])].astype(float)
                            z0 = z[np.nanargmin(abs(r - 1.0))]
                            q, r_taper, q_taper = 1.0, 2.0, 1.0
                            p0 = [z0, q, r_taper, q_taper]
                            coeff, _ = curve_fit(tapered_powerlaw, r, z, p0=p0)
                            tpl_fit = tapered_powerlaw(r, *coeff)
                            ax.plot(r, tpl_fit, '-', color='navy', linewidth=1.0,
                                    label=f'tapered power law parameters: \n z0={coeff[0]:.2f}, q={coeff[1]:.2f}, r_tap={coeff[2]:.2f}, q_tap={coeff[3]:.2f}')
                        except:
                            pass
                
                if self.sR[:,:,1].any():
                    ax.plot(self.sR[:,:,1].flatten(), surfs[k][:,:,1].flatten(), 'o', markersize=4, color='navajowhite', alpha=0.4, fillstyle='none', label='lower surface')
                    x1 = self.sR[:,:,1][self.sR[:,:,1] != None].flatten()
                    y1 = surfs[k][:,:,1][surfs[k][:,:,1] != None].flatten()
                    idx = np.argsort(x1)
                    x1, y1 = x1[idx], y1[idx]
                    win = signal.windows.hann(int(self.bmaj/self.pixelscale))
                    y1 = signal.convolve(y1, win, mode='same') / sum(win)
                    bins, _, _ = stats.binned_statistic(x1.astype(float),[x1.astype(float),y1.astype(float)], bins=np.round(self.Rout))
                    ax.plot(bins[0,:], bins[1,:], '.', markersize=8, color='darkorange', markeredgecolor='gold', markeredgewidth=0.3, label='avg. lower surface') 

                    if k == 0:
                        try:
                            r = bins[0,:][np.isfinite(bins[0,:])].astype(float)
                            z = bins[1,:][np.isfinite(bins[1,:])].astype(float) 
                            z0 = z[np.argmin(abs(r - 1.0))]
                            q, r_taper, q_taper = 1.0, 2.0, 1.0
                            p0 = [z0, q, r_taper, q_taper]
                            coeff, _ = curve_fit(tapered_powerlaw, r, z, p0=p0)
                            tpl_fit = tapered_powerlaw(r, *coeff)
                            ax.plot(r, tpl_fit, '-', color='darkorange', linewidth=1.0,
                                    label=f'tapered power law parameters: \n z0={coeff[0]:.2f}, q={coeff[1]:.2f}, r_tap={coeff[2]:.2f}, q_tap={coeff[3]:.2f}')
                        except:
                            pass

                ylims = (np.nanmin(surfs[k][surfs[k] != None]),np.nanpercentile(surfs[k][np.isfinite(surfs[k].astype(float))].astype(float),[99.7])) if k == 1 else None
                ax.set(xlabel=xlabel[0], ylabel=ylabels[k], ylim=ylims)
                ax.legend(loc='best', fontsize=6, markerscale=1.5)
            
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            pdf.close()

        else:
            print('no valid surfaces')
        
        
#################
def _rotate_disc(channel, angle=None, cx=None, cy=None):

    padX = [channel.shape[1] - cx, cx]
    padY = [channel.shape[0] - cy, cy]
    imgP = np.pad(channel, [padY, padX], 'constant')
    imgR = ndimage.rotate(imgP, angle, reshape=False)
    im = imgR[padY[0] : -padY[1], padX[0] : -padX[1]]

    return im


def _peak_finder(profile, height=None, threshold=None, distance=None, prominence=None, width=None):

    peaks, properties = signal.find_peaks(profile, height=height, threshold=threshold, distance=distance, prominence=prominence, width=width)
    
    return peaks, properties


def _pol2cart(radius, angle, cx=None, cy=None):
    
    x = radius * np.cos(angle) + cx
    y = radius * np.sin(angle) + cy
    
    return [y,x]
    

def gauss(x, *p):
    
    amp, center, sigma = p
    
    return (amp / (np.sqrt(2.*np.pi)*sigma)) * np.exp(-abs(x-center)**2 / (2.*sigma**2))


def double_gauss(x, *p):
    
    amp1, center1, sigma1, amp2, center2, sigma2, = p

    gaus1 = (amp1 / (np.sqrt(2.*np.pi)*sigma1)) * np.exp(-abs(x-center1)**2 / (2.*sigma1**2))
    gaus2 = (amp2 / (np.sqrt(2.*np.pi)*sigma2)) * np.exp(-abs(x-center2)**2 / (2.*sigma2**2))
    
    return gaus1 + gaus2


def sine_func(x, *p):

    a, b, c = p
    
    return a * np.sin(b * x + c)


def tapered_powerlaw(r, *p):

    z0, q, r_taper, q_taper = p
    r0 = 1.0

    return (z0 * (r / r0)**q) * np.exp(-(r / r_taper)**q_taper)


def _systemic_velocity(profile, nchans=None, v0=None, dv=None):

    channels = np.arange(0,nchans,1)
    p0 = [np.nanmax(profile), nchans/2, np.nanstd(profile)]
    coeff, var_matrix = curve_fit(gauss, channels, profile, p0=p0)
    gauss_fit = gauss(channels, *coeff)

    vsyst_idx = channels[np.argmax(gauss_fit)]
    vsyst = v0 + (dv * vsyst_idx)

    return vsyst, vsyst_idx, gauss_fit


def _center_of_mass(img, beam=None):

    img_gray = np.nan_to_num(img)
    img_gray[img_gray != 0] = 1
    footprint = morphology.disk(beam)
    res = morphology.white_tophat(img_gray, footprint)
    imgp0 = img.copy()
    imgp0[res == 1] = None

    imgp = ndimage.median_filter(np.nan_to_num(imgp0), int(beam))
    imgp[imgp == 0.] = None
    
    abs_imgp = abs(imgp)
    
    normalizer = np.nansum(abs_imgp)
    grids = np.ogrid[[slice(0, i) for i in abs_imgp.shape]]

    results = [np.nansum(abs_imgp * grids[dir].astype(float)) / normalizer
               for dir in range(abs_imgp.ndim)]

    if np.isscalar(results[0]):
        centre_of_mass = tuple(results)
    else:
        centre_of_mass = [tuple(v) for v in np.array(results).T]

    x_coord0 = centre_of_mass[1]
    y_coord0 = centre_of_mass[0]

    x,y = np.meshgrid(np.arange(imgp.shape[1]) - x_coord0, np.arange(imgp.shape[0]) - y_coord0)
    R = np.hypot(x,y)
    R[np.isnan(imgp)] = None
    Rout = np.nanmax(R)
    polar = warp_polar(imgp, center=(y_coord0,x_coord0), radius=Rout)
    polar[:,np.where(np.isnan(polar).any(axis=0))] = None
    rmax = (~np.isnan(polar)).cumsum(1).argmax(1)[0]
    
    x,y = np.meshgrid(np.arange(img.shape[1]) - x_coord0, np.arange(img.shape[0]) - y_coord0)    
    radius = np.hypot(x,y)
    mask = (radius <= rmax) if rmax < 10*beam else (radius <= 10*beam)
    
    masked_img = np.nan_to_num(abs_imgp.copy())
    masked_img[~mask] = None

    dx = ndimage.sobel(masked_img, axis=1, mode='nearest')
    dy = ndimage.sobel(masked_img, axis=0, mode='nearest')
    grad_img = np.hypot(dy,dx)

    if len(imgp[mask].flatten()[np.isnan(imgp[mask].flatten())]) > np.square(beam)/3 and rmax >= int(3*beam):
        kernel = np.array(np.ones([int(3*beam),int(3*beam)])/np.square(int(3*beam)))
    else:
        kernel = np.array(np.ones([int(beam),int(beam)])/np.square(int(beam)))
    grad_imgc = ndimage.convolve(grad_img, kernel)

    if np.all(np.isnan(grad_imgc)):
        grad_imgc[int(y_coord0),int(x_coord0)] = 1
        
    y_coord, x_coord = np.unravel_index(np.nanargmax(grad_imgc), grad_imgc.shape)
    
    return [y_coord,x_coord], imgp

        
def _position_angle(img, cx=None, cy=None, beam=None):
    
    x,y = np.meshgrid(np.arange(img.shape[1]) - cx, np.arange(img.shape[0]) - cy)
    R = np.hypot(x,y)
    R[np.isnan(img)] = None
    Rout = np.nanmax(R)

    phi = np.deg2rad(np.arange(0,360,1))
    phi[phi > np.pi] -= 2*np.pi
    
    rad = np.arange(1, Rout.astype(np.int), 1)
    img0 = ndimage.median_filter(np.nan_to_num(img), int(beam))
    polar = warp_polar(img0, center=(cy,cx), radius=Rout)
    
    semimajor = []
    
    for i in rad:
        
        annuli = np.nan_to_num(polar[:,i])

        if len(annuli[np.where(annuli != 0.)]) < 0.90*len(annuli):
            continue
           
        if len(annuli[np.where(annuli == 0.)]) != 0:
            phi_idx = np.where(annuli == 0.)[0]
            fillers = np.interp(phi_idx.astype(float), phi[annuli != 0.], annuli[annuli != 0.])
            annuli[phi_idx] = fillers

        smoothed = signal.savgol_filter(annuli, window_length=11, polyorder=2, mode='interp')
        residuals = annuli - smoothed
        ss_res = np.sum(np.square(residuals))
        ss_tot = np.sum(np.square(annuli-np.mean(annuli)))
        rsquared = 1 - (ss_res / ss_tot)
        if rsquared < 0.997:
            continue
        
        red_max = phi[np.nanargmax(annuli)]
        blue_max = phi[np.nanargmin(annuli)]
        
        if red_max > blue_max:
            blue_max += np.pi
        elif red_max < blue_max:
            blue_max -= np.pi
        x = np.average([i*np.cos(red_max),i*np.cos(blue_max)])
        y = np.average([i*np.sin(red_max),i*np.sin(blue_max)])
        theta = np.rad2deg(np.arctan2(y,x)) - 90
        
        semimajor.append(theta)

    PA = np.average(semimajor)
    if PA < -90:
        PA += 360
        
    return PA, Rout


