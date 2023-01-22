import os
import sys

import numpy as np
import numpy_indexed as npi
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.backends.backend_pdf
import itertools
import math

from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy import stats
from tqdm import tqdm
from scipy import ndimage
from skimage.measure import regionprops
from skimage import color, morphology
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
from matplotlib.patches import Ellipse
from skimage.transform import warp_polar
from scipy.stats import binned_statistic
from scipy.signal import savgol_filter

import matplotlib.cm as cm
import cmasher as cmr

np.set_printoptions(threshold=np.inf)

###################################################################################################

class EXTRACT:

    def __init__(self, fits_file, distance=None, sigma=None, cx=None, cy=None, inc=None):        

        self.inc = inc
        self.distance = distance
        
        self._fits_info(fits_file)
        self._compute_geometric_parameters()
        self._trace_surface()
        self._extract_surface()
        self._plots()

        return
    

    def _fits_info(self, fits_file):

        self.filename = os.path.normpath(os.path.expanduser(fits_file))        
        hdu = fits.open(self.filename)

        # header
        #print(repr(hdu[0].header))
        self.source = hdu[0].header['OBJECT']
        
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

        if hdu[0].header['CTYPE3'] == 'VRAD':
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
        
        
    def _compute_geometric_parameters(self):
        
        print('GETTING GEOMETRIC PROPERTIES')
        
        cube = np.nan_to_num(self.image)
        rms = np.nanstd(cube)
        self.cube = cube
        self.rms = rms
        print(f'rms [{self.iunit}] =', rms)

        # systemic velocity
        print('extracting systemic velocity')

        line_profile = np.nansum(cube, axis=(1,2))
        vsyst, vsyst_idx = _systemic_velocity(line_profile, nchans=self.nv, v0=self.velocity[0], dv=self.dv)
        
        print('systemic velocity (m/s) =', vsyst)
        
        self.vsyst = vsyst
        self.vsyst_idx = vsyst_idx
        self.line_profile = line_profile
        
        # velocity map
        '''
        ## peak velocity
        peak_array = np.zeros([self.nx,self.ny], dtype='int')
        for i, k in tqdm(itertools.product(range(self.nx), range(self.ny)), total=self.nx*self.ny):
            peaks, properties = _peak_finder(cube[:,i,k], width=5, height=5*rms, prominence=5*rms)
            if len(peaks) > 0:
                max_peak = peaks[np.argmax(properties["peak_heights"])]
                peak_array[i,k] = max_peak
                
        vpeak = self.velocity[peak_array]
        vpeak[vpeak == self.velocity[0]] = None
        vpeak -= vsyst

        plt.imshow(vpeak, origin='lower', cmap=cm.RdBu_r)
        plt.show()
        sys.exit()
        self.vpeak = vpeak
        '''
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
        print('extracting center of mass')
        
        vmap = self.M1.copy()
        beam_pix = self.bmaj/self.pixelscale
        com, M1p = _center_of_mass(vmap, beam=beam_pix)
        print('center coordinates (pixels) =', com)

        self.com = com
        self.M1p = M1p

        # position angle
        print('extracting position angle')

        PA, nearside, Rout = _position_angle(self.M1p, cx=self.com[1], cy=self.com[0], beam=beam_pix)
        print('position angle (degrees) =', PA)
        print('nearside (degrees) =', nearside)

        self.PA = PA
        self.nearside = nearside
        self.Rout = Rout
        
        
    def _trace_surface(self):

        print('TRACING LAYERS')

        trms = abs(np.nansum(self.cube[0,:,:], axis=(0,1)))
        channels = np.arange(0,self.nv,1)
        tchans = channels[np.where(self.line_profile > 10*trms)]
        beam = self.bmaj/self.pixelscale

        surfaces = np.full([self.nv,(self.Rout+1).astype(np.int),4,3], None)

        phi = np.deg2rad(np.arange(0,360,1))
        phi[phi > np.pi] -= 2*np.pi
        rad = np.arange(1, self.Rout.astype(np.int), 1)
        
        for i in tchans:

            chan = self.cube[i,:,:]
            chan_rot = _rotate_disc(chan, angle=self.nearside-180, cx=self.com[1], cy=self.com[0])
            polar = warp_polar(chan_rot, center=(self.com[1],self.com[0]), radius=self.Rout)
            
            grad0 = np.full([4,2], None)
            coord0 = np.full([4,2], None)
            coord0[:,:] = [self.com[0],self.com[1]]
            
            for k in rad:
                
                annuli = polar[:,k]
              
                peaks, properties = _peak_finder(annuli, height=5*self.rms, distance=beam, prominence=3*self.rms, width=0.5*beam)
                sorted_peaks = peaks[np.argsort(annuli[peaks])][::-1]

                if len(sorted_peaks) >= 2:
                    if np.all(phi[sorted_peaks[:2]] > np.pi/2):
                        phi0 = np.min(phi[sorted_peaks[:2]])
                        phi1 = np.max(phi[sorted_peaks[:2]])
                    else:
                        phi0 = np.max(phi[sorted_peaks[:2]])
                        phi1 = np.min(phi[sorted_peaks[:2]])
                    far_up = _pol2cart(k, phi0, cx=self.com[1], cy=self.com[0])
                    near_up = _pol2cart(k, phi1, cx=self.com[1], cy=self.com[0])
                    
                    surfaces[i,k,0,:] = np.concatenate((far_up,[phi0]))
                    surfaces[i,k,1,:] = np.concatenate((near_up,[phi1]))
                    
                if len(sorted_peaks) >= 4:
                    if np.all(phi[sorted_peaks[2:4]] > np.pi/2):
                        phi2 = np.min(phi[sorted_peaks[2:4]])
                        phi3 = np.max(phi[sorted_peaks[2:4]])
                    else:
                        phi2 = np.max(phi[sorted_peaks[2:4]])
                        phi3 = np.min(phi[sorted_peaks[2:4]])
                    far_lo = _pol2cart(k, phi2, cx=self.com[1], cy=self.com[0])
                    near_lo = _pol2cart(k, phi3, cx=self.com[1], cy=self.com[0])
                    
                    surfaces[i,k,2,:] = np.concatenate((far_lo,[phi2]))
                    surfaces[i,k,3,:] = np.concatenate((near_lo,[phi3]))

                # removing discontinuous points
                '''
                for vx in range(4):
                    
                    if surfaces[i,k,vx,:].all():
                        
                        if (surfaces[i,k,vx,1] - coord0[vx,1]) == 0. or (surfaces[i,k,vx,0] - coord0[vx,0]) == 0.:
                            surfaces[i,k,vx,:] = None
                            continue
                        
                        grady = abs(surfaces[i,k,vx,0] - coord0[vx,0]) 
                        gradx = abs(surfaces[i,k,vx,1] - coord0[vx,1]) 

                        #if i == tchans[19]:
                        #    print(grady, gradx)
                        
                        if grad0[vx,:].all():

                            voly = grady / grad0[vx,0]
                            volx = gradx / grad0[vx,1]

                            if i == tchans[19]:
                                #print(grady / grad0[vx,0], gradx / grad0[vx,1])
                                print(voly,volx)
                            
                            if volx > 10 or voly > 10:
                                surfaces[i,k,vx,:] = None
                            else:
                               grad0[vx,:] = [grady,gradx]
                               coord0[vx,:] = surfaces[i,k,vx,:2]  
                        else:
                            grad0[vx,:] = [grady,gradx]
                
                if np.any(surfaces[i,k,:2,:] == None):
                    surfaces[i,k,:2,:] = None
                if np.any(surfaces[i,k,2:4,:] == None):
                    surfaces[i,k,2:4,:] = None
                '''
                
        self.surfaces = surfaces
        self.tchans = tchans


    def _extract_surface(self):

        mid = np.full([self.nv,(self.Rout+1).astype(np.int),2,2], None)
        top = np.full([self.nv,(self.Rout+1).astype(np.int),2,2], None)
        bot = np.full([self.nv,(self.Rout+1).astype(np.int),2,2], None)
        
        vdir = -1 if self.PA < self.nearside else 1
        vchans = []

        for i in self.tchans:
            for vx in range(2):
                if self.surfaces[i,:,(vx+2)*vx:(vx*2)+2,:].any():
                    phi0 = self.surfaces[i,:,(vx+2)*vx:(vx*2)+2,2][np.where(self.surfaces[i,:,(vx+2)*vx:(vx*2)+2,2] != None)]
                    if len(phi0[np.where(abs(abs(phi0)-np.pi/2) < np.deg2rad(10))]) > 0.5*len(phi0):
                        continue
                    else:
                        x0 = self.surfaces[i,:,(vx+2)*vx,1][np.where(self.surfaces[i,:,(vx+2)*vx,1] != None)]
                        x1 = self.surfaces[i,:,(vx*2)+1,1][np.where(self.surfaces[i,:,(vx*2)+1,1] != None)]
                        y0 = self.surfaces[i,:,(vx+2)*vx,0][np.where(self.surfaces[i,:,(vx+2)*vx,0] != None)]
                        y1 = self.surfaces[i,:,(vx*2)+1,0][np.where(self.surfaces[i,:,(vx*2)+1,0] != None)]
                        b0 = np.max([np.min(x0),np.min(x1)])
                        b1 = np.min([np.max(x0),np.max(x1)])
                        x = np.sort(np.arange(b0,b1,1))[::-1]
                        if len(x) == 0:
                            continue
                        else:
                            vchans.append(i)
                            idx1 = np.argsort(x0)
                            idx2 = np.argsort(x1)
                            yf0 = np.interp(x.astype(float), x0[idx1].astype(float), y0[idx1].astype(float))
                            yf1 = np.interp(x.astype(float), x1[idx2].astype(float), y1[idx2].astype(float))
                            yfm = np.mean([yf0,yf1], axis=0)

                            mid[i,:len(x),vx,0], mid[i,:len(x),vx,1] = yfm, x
                            top[i,:len(x),vx,0], top[i,:len(x),vx,1] = yf0, x
                            bot[i,:len(x),vx,0], bot[i,:len(x),vx,1] = yf1, x
                else:
                    continue

        self.mid = mid

        sR = np.full([self.nv,(self.Rout+1).astype(np.int),2], None)
        sH = np.full([self.nv,(self.Rout+1).astype(np.int),2], None)
        sV = np.full([self.nv,(self.Rout+1).astype(np.int),2], None)
        sI = np.full([self.nv,(self.Rout+1).astype(np.int),2], None)
        
        for i in vchans:
            for vx in range(2):
                if self.surfaces[i,:,(vx+2)*vx:(vx*2)+2,:].any():
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
                        r = np.hypot(top0x.astype(float) - self.com[1], (top0y.astype(float) - mid0y.astype(float)) / np.cos(np.deg2rad(self.inc)))
                        v = (self.velocity[i] - self.vsyst) * r / ((top0x.astype(float) - self.com[1]) * np.sin(np.deg2rad(self.inc)))
                        v *= vdir
                        v[np.where(v < 0)] = None
                        chan = self.cube[i,:,:]
                        chan_rot = _rotate_disc(chan, angle=self.nearside-180, cx=self.com[1], cy=self.com[0])
                        I = np.mean([chan_rot[top0y.astype(int),top0x.astype(int)], chan_rot[bot0y.astype(int),bot0x.astype(int)]], axis=0)

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
                ax.plot(self.vsyst_idx, self.line_profile[self.vsyst_idx], '.', markersize=10, color='red', label=f'systemic velocity = {self.vsyst:.3f} m/s')
                ax.legend(loc='upper right', fontsize=7)
                ax.set(xlabel = 'channel number', ylabel = f'$\Sigma$Flux [{self.iunit}]')
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
                        label=f'C.O.M.: [{self.com[0]},{self.com[1]}] pixels, \n [{(self.com[0]-self.imgcy)*self.pixelscale:.3f},{(self.com[1]-self.imgcx)*self.pixelscale:.3f}] $\Delta$arcs')
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
                ax.plot((major_axis[0],major_axis[1]),(major_axis[2],major_axis[3]), color='black', linestyle='--', linewidth=0.5, label=f'P.A. = {self.PA:.3f}$^\circ$')
                ax.plot((minor_axis[0],minor_axis[1]),(minor_axis[2],minor_axis[3]), color='darkgray', linestyle='--', linewidth=0.5, label=f'minor axis = {self.nearside:.3f}$^\circ$')
                near = [self.com[1]+(0.4*self.nx*np.cos(np.radians(self.nearside+90))), self.com[0]+(0.4*self.ny*np.sin(np.radians(self.nearside+90)))]
                rot = self.nearside-180 if self.nearside > 90 else self.nearside
                ax.annotate('near', xy=(near[0],near[1]), fontsize=8, color='black', rotation=rot)
                ax.legend(loc='best', fontsize=7, framealpha=1.0)
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        pdf.close()
        
        '''
        print('PLOTTING SURFACE TRACES')
            
        pdf = matplotlib.backends.backend_pdf.PdfPages(self.filename+'_traces.pdf')

        for i in tqdm(self.tchans, total=len(self.tchans)):      
            fig, ax = plt.subplots(figsize=(6,6))

            chan = self.cube[i,:,:]
            chan_rot = _rotate_disc(chan, angle=self.nearside-180, cx=self.com[1], cy=self.com[0])

            fig1 = ax.imshow(chan_rot, origin='lower', cmap=cmr.swamp, vmin=0, vmax=np.nanmax(self.cube))
            ax.plot(self.com[1], self.com[0], marker='+', markersize=10, color='white')
            ax.set(xlabel='pixels', ylabel='pixels')

            ax.plot(self.surfaces[i,:,0,1], self.surfaces[i,:,0,0], '.', markersize=2, color='blueviolet', label='upper far side')
            ax.plot(self.surfaces[i,:,1,1], self.surfaces[i,:,1,0], '.', markersize=2, color='darkorange', label='upper near side')
            ax.plot(self.surfaces[i,:,2,1], self.surfaces[i,:,2,0], '.', markersize=2, color='violet', label='lower far side')
            ax.plot(self.surfaces[i,:,3,1], self.surfaces[i,:,3,0], '.', markersize=2, color='gold', label='lower near side')
            ax.plot(self.mid[i,:,0,1], self.mid[i,:,0,0], '.', markersize=2, color='white', label='mid upper surface')
            ax.plot(self.mid[i,:,1,1], self.mid[i,:,1,0], '.', markersize=2, color='grey', label='mid lower side')
            ax.legend(loc='upper right', fontsize=7)
            bounds = self.Rout*1.1
            ax.set(xlim = (self.com[1]-bounds,self.com[1]+bounds), ylim = (self.com[0]-bounds,self.com[0]+bounds))
            
            beam = Ellipse(xy=(self.com[1]-self.Rout,self.com[0]-self.Rout), width=self.bmin/self.pixelscale,
                           height=self.bmaj/self.pixelscale, angle=-self.bpa, fill=True, color='white')
            ax.add_patch(beam)
            
            divider = make_axes_locatable(ax)
            colorbar_cax = divider.append_axes('right', size='4%', pad=0.05)
            cbar = fig.colorbar(fig1, shrink=0.97, aspect=70, spacing='proportional', orientation='vertical', cax=colorbar_cax)#, extend='both')
            cbar.ax.tick_params(labelsize=8)
            cbar.ax.xaxis.set(ticks_position = 'top', label_position = 'top')
            tick_locator = ticker.MaxNLocator(nbins=5)
            cbar.locator = tick_locator
            cbar.update_ticks()
            cbar.set_label(f'flux [{self.iunit}]', labelpad=9, fontsize=12, weight='bold')
        
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        pdf.close()
        

        print('PLOTTING SURFACES')

        self.sH[np.where(self.sH != None)] *= self.pixelscale
        self.sR[np.where(self.sR != None)] *= self.pixelscale
        
        surfs = {0: self.sH, 1: self.sV, 2: self.sI}
        ylabels = ['h [arcsec]', 'v [m/s]', f'Int [{self.iunit}]']
        
        pdf = matplotlib.backends.backend_pdf.PdfPages(self.filename+'_profiles.pdf')
        for k in range(3):

            fig, ax = plt.subplots(figsize=(6,6))
            if self.sR[:,:,0].any():
                ax.plot(self.sR[:,:,0].flatten(), surfs[k][:,:,0].flatten(), 'o', markersize=4, color='skyblue', fillstyle='none', label='upper surface')
                x1 = self.sR[:,:,0][self.sR[:,:,0] != None].flatten()
                y1 = surfs[k][:,:,0][surfs[k][:,:,0] != None].flatten()
                bins, _, _ = binned_statistic(x1.astype(float),[x1.astype(float),y1.astype(float)], bins=np.round(self.Rout))
                ax.plot(bins[0,:], bins[1,:], '.', markersize=4, color='navy', label='avg. upper surface')
                
            if self.sR[:,:,1].any():
                ax.plot(self.sR[:,:,1].flatten(), surfs[k][:,:,1].flatten(), 'o', markersize=4, color='navajowhite', fillstyle='none', label='lower surface')
                x1 = self.sR[:,:,1][self.sR[:,:,1] != None].flatten()
                y1 = surfs[k][:,:,1][surfs[k][:,:,1] != None].flatten()
                bins, _, _ = binned_statistic(x1.astype(float),[x1.astype(float),y1.astype(float)], bins=np.round(self.Rout))
                ax.plot(bins[0,:], bins[1,:], '.', markersize=4, color='darkorange', label='avg. lower surface')

            ylims = (0,np.nanpercentile(surfs[k].astype(float),[99.7])) if k == 1 else None
            ax.set(xlabel='r [arcsec]', ylabel=ylabels[k], ylim=ylims)
            ax.legend(loc='upper right', fontsize=10)
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
        pdf.close()
        '''
        
#################
def _rotate_disc(channel, angle=None, cx=None, cy=None):

    padX = [channel.shape[1] - cx, cx]
    padY = [channel.shape[0] - cy, cy]
    imgP = np.pad(channel, [padY, padX], 'constant')
    imgR = ndimage.rotate(imgP, angle, reshape=False)
    im = imgR[padY[0] : -padY[1], padX[0] : -padX[1]]

    return im


def _peak_finder(profile, height=None, threshold=None, distance=None, prominence=None, width=None):

    peaks, properties = find_peaks(profile, height=height, threshold=threshold, distance=distance, prominence=prominence, width=width)
    
    return peaks, properties


def _pol2cart(radius, angle, cx=0, cy=0):
    
    x = radius * np.cos(angle) + cx
    y = radius * np.sin(angle) + cy
    
    return [y,x]


def _systemic_velocity(profile, nchans=None, v0=None, dv=None):

    channels = np.arange(0,nchans,1)
    p0 = [np.nanmax(profile), nchans/2, np.nanstd(profile)]
    coeff, var_matrix = curve_fit(gauss, channels, profile, p0=p0)
    gauss_fit = gauss(channels, *coeff)

    vsyst_idx = channels[np.argmax(gauss_fit)]
    vsyst = v0 + (dv * vsyst_idx)

    return vsyst, vsyst_idx
    

def gauss(x, *p):
    
    amp, center, sigma = p
    
    return (amp / (np.sqrt(2.*np.pi)*sigma)) * np.exp(-abs(x-center)**2 / (2.*sigma**2))


def _center_of_mass(img, beam=None):

    img_gray = np.nan_to_num(img)
    img_gray[img_gray != 0] = 1
    footprint = morphology.disk(beam)
    res = morphology.white_tophat(img_gray, footprint)
    imgp = img.copy()
    imgp[res == 1] = None

    abs_imgp = abs(imgp)

    normalizer = np.nansum(abs_imgp)
    grids = np.ogrid[[slice(0, i) for i in abs_imgp.shape]]

    results = [np.nansum(abs_imgp * grids[dir].astype(float)) / normalizer
               for dir in range(abs_imgp.ndim)]

    if np.isscalar(results[0]):
        centre_of_mass = tuple(results)
    else:
        centre_of_mass = [tuple(v) for v in np.array(results).T]

    x_coord = centre_of_mass[1]
    y_coord = centre_of_mass[0]

    y,x = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
    radius = np.hypot(y-y_coord,x-x_coord)
    mask = radius <= (5*beam)

    masked_img = abs_imgp.copy()
    masked_img[~mask] = None

    dx, dy = np.gradient(masked_img, edge_order=2)
    grad_img = np.hypot(dy,dx)

    kernel = np.array(np.ones([int(beam),int(beam)])/np.square(int(beam)))
    grad_imgc = ndimage.convolve(grad_img, kernel)
   
    y_coord, x_coord = np.unravel_index(np.nanargmax(grad_imgc), grad_imgc.shape)
    
    return [y_coord,x_coord], imgp


def sine_func(x, *p):

    a, b, c = p
    
    return a * np.sin(b * x + c)

        
def _position_angle(img, cx=None, cy=None, beam=None):
    
    x,y = np.meshgrid(np.arange(img.shape[1]) - cx, np.arange(img.shape[0]) - cy)
    R = np.hypot(x,y)
    R[np.isnan(img)] = None
    Rout = np.nanmax(R)

    phi = np.deg2rad(np.arange(0,360,1))
    phi[phi > np.pi] -= 2*np.pi

    phi_deg = np.arange(0,360,1)
    
    rad = np.arange(1, Rout.astype(np.int), 1)
    polar = warp_polar(img, center=(cx,cy), radius=Rout)
    
    semimajor = []
    near = []
    
    for i in rad:
        
        annuli = np.nan_to_num(polar[:,i])

        if len(annuli[np.where(annuli != 0.)]) < 0.90*len(annuli):
            continue
           
        if len(annuli[np.where(annuli == 0.)]) != 0:
            phi_idx = np.where(annuli == 0.)[0]
            fillers = np.interp(phi_idx.astype(float), phi[annuli != 0.], annuli[annuli != 0.])
            annuli[phi_idx] = fillers

        smoothed = savgol_filter(annuli, window_length=11, polyorder=2, mode='interp')
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
        
        tred_max = phi_deg[np.nanargmax(annuli)]
        tblue_max = phi_deg[np.nanargmin(annuli)]
        
        bend = np.average([tred_max, tblue_max])
        
        if (tred_max - bend) < 90:
            if bend < 180:
                bend += 180
            elif bend > 180:
                bend -= 180
        bend -= 90
        
        near.append(bend)

    PA = np.average(semimajor)
    if PA < -90:
        PA += 360
        
    nearside = np.average(near)
    if nearside < -90:
        nearside += 360
        
    print(nearside)
    sys.exit()
    return PA, nearside, Rout


