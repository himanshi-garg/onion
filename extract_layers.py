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
from time import sleep
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy import stats
from tqdm import tqdm
from scipy import ndimage
from skimage.measure import regionprops
from skimage import color, morphology
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
from matplotlib.patches import Ellipse
from skimage.transform import warp_polar

import matplotlib.cm as cm
import cmasher as cmr

np.set_printoptions(threshold=np.inf)

###################################################################################################

class EXTRACT:

    def __init__(self, fits_file, distance=None, sigma=None, cx=None, cy=None):        

        self.inc = 21.0
        self._fits_info(fits_file)
        self._compute_geometric_parameters()
        self._trace_surface()
        self._extract_surface_info()
        self._plot_surfaces()

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
        # angle unit
        self.aunit = hdu[0].header['CUNIT1']
        # intensity unit
        self.iunit = hdu[0].header['BUNIT'] 
        self.restfreq = hdu[0].header['RESTFRQ']
        if self.aunit == 'rad':
            self.bpa = np.rad2deg(self.bpa)
        elif self.aunit == 'deg':
            self.bpa = self.bpa
        else:
            raise ValueError("unknown angle units:", self.aunit)

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
        M1 -= vsyst
        
        self.M1 = M1
        
        # center of mass
        print('extracting center of mass')
        
        vmap = self.M1.copy()
        beam_pix = beam=self.bmaj/self.pixelscale
        com = _center_of_mass(vmap, beam=beam_pix)
        print('center coordinates (pixels) =', com)

        self.com = com

        # position angle
        print('extracting position angle')

        vmap = self.M1.copy()
        PA, nearside, Rout = _position_angle(vmap, cx=self.com[1], cy=self.com[0], beam=beam_pix)
        print('position angle (degrees) =', PA)
        print('nearside (degrees) =', nearside)

        self.PA = PA
        self.nearside = nearside
        self.Rout = Rout

        # plotting figures
        
        pdf = matplotlib.backends.backend_pdf.PdfPages(self.filename+'_gp.pdf')

        # line profile for systemic velocity
        fig, ax = plt.subplots(figsize=(6,6))
        ax.plot(line_profile, color='black', linewidth=1.0)
        ax.plot(vsyst_idx, line_profile[vsyst_idx], '.', markersize=10, color='red', label=f'v_syst = {self.vsyst} m/s')
        ax.legend(loc='upper right', fontsize=7)
        ax.set(xlabel = 'channel number', ylabel = f'$\Sigma$Flux [{self.iunit}]', title='systemic velocity')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # velocity map for centre of mass
        fig, ax = plt.subplots(figsize=(6,6))
        fig1 = ax.imshow(self.M1/1000, origin='lower', cmap=cm.RdBu_r, vmin=np.nanpercentile(self.M1/1000,[2]), vmax=np.nanpercentile(self.M1/1000,[98]))
        ax.plot(self.com[1], self.com[0], marker='.', markersize=10, color='black')
        ax.set(xlabel='pixels', ylabel='pixels', title='dynamical centre')
        divider = make_axes_locatable(ax)
        colorbar_cax = divider.append_axes('right', size='4%', pad=0.05)
        cbar = fig.colorbar(fig1, shrink=0.97, aspect=70, spacing='proportional', orientation='vertical', cax=colorbar_cax, extend='both')
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.xaxis.set(ticks_position = 'top', label_position = 'top')
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.update_ticks()
        cbar.set_label('$\Delta$v [kms$^{-1}$]', labelpad=9, fontsize=12, weight='bold')
        axins = zoomed_inset_axes(ax, self.nx/(3*4*beam_pix), loc='upper right')
        axins.imshow(self.M1, origin='lower', cmap=cm.RdBu_r, vmin=np.nanpercentile(M1,[2]), vmax=np.nanpercentile(M1,[98]))
        axins.set(xlim = (com[1]-2*beam_pix,com[1]+2*beam_pix), ylim = (com[0]-2*beam_pix,com[0]+2*beam_pix))
        axins.yaxis.get_major_locator().set_params(nbins=4)
        axins.xaxis.get_major_locator().set_params(nbins=4)
        plt.xticks(visible=False)
        plt.yticks(visible=False)
        mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='black', linewidth=0.5, linestyle=':')
        axins.plot(self.com[1], self.com[0], marker='.', markersize=10, color='black')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # velocity map for position angle
        fig, ax = plt.subplots(figsize=(6,6))
        fig1 = ax.imshow(self.M1/1000, origin='lower', cmap=cm.RdBu_r, vmin=np.nanpercentile(self.M1/1000,[2]), vmax=np.nanpercentile(self.M1/1000,[98]))
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
        ax.plot((major_axis[0],major_axis[1]),(major_axis[2],major_axis[3]), color='black', linestyle='--', linewidth=0.5, label=f'PA = {self.PA}$^\circ$')
        if self.PA < 180:
            near = self.PA + 90
        else:
            near = self.PA - 90
        ax.plot((minor_axis[0],minor_axis[1]),(minor_axis[2],minor_axis[3]), color='darkgray', linestyle='--', linewidth=0.5, label=f'minor axis = {near}$^\circ$')
        near = [self.com[1]+(0.4*self.nx*np.cos(np.radians(self.nearside+90))), self.com[0]+(0.4*self.ny*np.sin(np.radians(self.nearside+90)))]
        rot = self.nearside-180 if self.nearside > 90 else self.nearside
        ax.annotate('near', xy=(near[0],near[1]), fontsize=8, color='black', rotation=rot)
        ax.legend(loc='best', fontsize=7)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        pdf.close()
        
        
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
                        grad = (surfaces[i,k,vx,0] - coord0[vx,0]) / (surfaces[i,k,vx,1] - coord0[vx,1])
                        grad *= surfaces[i,k,vx,0]
                        
                        if grad0[vx,:].all():
                            gdiff = abs(grad - grad0) / (surfaces[i,k,vx,1] - coord0[vx,1])
                            
                            meany = np.mean([grady, grad0[vx,0]])
                            meanx = np.mean([gradx, grad0[vx,1]])
                            cam = (meanx + meany) / 2.
                            d1 = meany - cam
                            d2 = meanx - cam
                            voly = np.std([grady, grad0[vx,0]]) 
                            volx = np.std([gradx, grad0[vx,1]]) 
                            volatility = np.sqrt((np.square(voly) + np.square(volx) + np.square(d1) + np.square(d2)) / 2.) 

                            if i == tchans[19]:
                                print(grady, gradx, volatility)
                            
                            if volatility > 90:
                                surfaces[i,k,vx,:] = None
                                intensity[i,k,vx] = None
                            else:
                               grad0[vx,:] = [grady,gradx]
                               coord0[vx,:] = surfaces[i,k,vx,:]  
                        else:
                            grad0[vx,:] = [grady,gradx]

                if np.any(surfaces[i,k,:2,:] == None):
                    surfaces[i,k,:2,:] = None
                if np.any(surfaces[i,k,2:4,:] == None):
                    surfaces[i,k,2:4,:] = None
                '''
                
        self.surfaces = surfaces
        self.tchans = tchans


    def _plot_surfaces(self):

        print('PLOTTING TRACED LAYERS')
            
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


    def _extract_surface_info(self):

        mid = np.full([self.nv,(self.Rout+1).astype(np.int),2,2], None)
        top = np.full([self.nv,(self.Rout+1).astype(np.int),2,2], None)
        bot = np.full([self.nv,(self.Rout+1).astype(np.int),2,2], None)
        
        r_up, h_up, v_up, I_up = [], [], [], [] 
        r_lo, h_lo, v_lo, I_lo = [], [], [], []

        vdir = -1 if self.PA < self.nearside else 1
        vchans = []

        for i in self.tchans:
            for vx in range(2):
                if self.surfaces[i,:,(vx+2)*vx:(vx*2)+2,:].any():
                    phi0 = self.surfaces[i,:,(vx+2)*vx:(vx*2)+2,2][np.where(self.surfaces[i,:,(vx+2)*vx:(vx*2)+2,2] != None)]
                    if len(phi0[np.where(abs(phi0-np.pi/2) < np.deg2rad(20))]) > 0.5*len(phi0):
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

                            #if i == self.tchans[19]:
                            #    print(x0)
                            #    print(y0)
                            #    print(x.astype(float))
                            #    print(yf0.astype(float))
                            #    sys.exit()

                            mid[i,:len(x),vx,0], mid[i,:len(x),vx,1] = yfm, x
                            top[i,:len(x),vx,0], top[i,:len(x),vx,1] = yf0, x
                            bot[i,:len(x),vx,0], bot[i,:len(x),vx,1] = yf1, x
                else:
                    continue

        self.mid = mid
        
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
                        r = np.hypot(top0x.astype(float) - self.com[1], (top0y.astype(float) - mid0y.astype(float)) / np.cos(np.deg2rad(self.inc)))
                        v = (self.velocity[i] - self.vsyst) * r / ((top0x.astype(float) - self.com[1]) * np.sin(np.deg2rad(self.inc)))
                        v *= vdir
                        chan = self.cube[i,:,:]
                        chan_rot = _rotate_disc(chan, angle=self.nearside-180, cx=self.com[1], cy=self.com[0])
                        I = np.mean([chan_rot[top0y.astype(int),top0x.astype(int)], chan_rot[bot0y.astype(int),bot0x.astype(int)]], axis=0)

                        if vx == 0:
                            r_up = np.concatenate((r_up,r))
                            h_up = np.concatenate((h_up,h))
                            v_up = np.concatenate((v_up,v))
                            I_up = np.concatenate((I_up,I))
                        elif vx == 1:
                            r_lo = np.concatenate((r_lo,r))
                            h_lo = np.concatenate((h_lo,h))
                            v_lo = np.concatenate((v_lo,v))
                            I_lo = np.concatenate((I_lo,I))
                else:
                    continue         
        
        surf_up = [h_up, v_up, I_up]
        surf_lo = [h_lo, v_lo, I_lo]
        ylabels = ['h [arcsec]', 'v [m/s]', f'Int [{self.iunit}]']
        
        pdf = matplotlib.backends.backend_pdf.PdfPages(self.filename+'_profiles.pdf')
        for k in range(len(surf_up)):
        
            fig, ax = plt.subplots(figsize=(6,6))
            
            ax.plot(r_up, surf_up[k], '.', markersize=1, color='black', label='upper surface')
            ax.plot(r_lo, surf_lo[k], '.', markersize=1, color='red', label='lower surface')
            
            ax.set(xlabel='r [arcsec]', ylabel=ylabels[k])
            ax.legend(loc='upper right', fontsize=10)
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
        pdf.close()
                        
        
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
    
    img = abs(img)

    img_gray = np.nan_to_num(img)
    img_gray[img_gray != 0] = 1
    footprint = morphology.disk(beam)
    res = morphology.white_tophat(img_gray, footprint)
    img_denoised = img
    img_denoised[res == 1] = None
        
    #img = (img - np.nanmin(img))/(np.nanmax(img) - np.nanmin(img))
    #img = img * (10 - 0.1) + 0.1
    #img = np.exp(img)
    
    normalizer = np.nansum(img_denoised)
    grids = np.ogrid[[slice(0, i) for i in img_denoised.shape]]

    results = [np.nansum(img_denoised * grids[dir].astype(float)) / normalizer
               for dir in range(img_denoised.ndim)]

    if np.isscalar(results[0]):
        centre_of_mass = tuple(results)
    else:
        centre_of_mass = [tuple(v) for v in np.array(results).T]

    x_coord = centre_of_mass[1]
    y_coord = centre_of_mass[0]

    # refine center of mass coordinates
    y,x = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
    radius = np.hypot(y-y_coord,x-x_coord)
    mask = radius <= (5*beam)

    masked_img = img.copy()
    masked_img[~mask] = None

    dx, dy = np.gradient(masked_img, edge_order=2)
    grad_img = np.hypot(dy,dx)

    kernel = np.array(np.ones([int(beam),int(beam)])/np.square(int(beam)))
    grad_imgc = ndimage.convolve(grad_img, kernel)
    y_coord, x_coord = np.unravel_index(np.nanargmax(grad_imgc), grad_imgc.shape)
    
    return [y_coord,x_coord]


def _position_angle(img, cx=None, cy=None, beam=None):

    img_gray = np.nan_to_num(img)
    img_gray[img_gray != 0] = 1
    footprint = morphology.disk(beam)
    res = morphology.white_tophat(img_gray, footprint)
    img[res == 1] = None
    '''
    x,y = np.meshgrid(np.arange(img.shape[1]) - cx, np.arange(img.shape[0]) - cy)
    R = np.hypot(x,y)
    R[np.isnan(img)] = None
    Rout = np.nanmax(R)

    phi = np.deg2rad(np.arange(0,360,1))
    
    rad = np.arange(1, Rout.astype(np.int), 1)
    polar = warp_polar(img, center=(cx,cy), radius=Rout)

    semimajor = []
    near = []
    
    for i in rad:
        
        annuli = polar[:,i]
        red_max = phi[np.nanargmax(annuli)]
        blue_max = phi[np.nanargmin(annuli)]
        #print(np.rad2deg(red_max), np.rad2deg(blue_max))
        if i == 50:
            plt.plot(annuli)
            print(np.rad2deg(red_max), np.rad2deg(blue_max))
            plt.show()
            sys.exit()

        if red_max > blue_max:
            theta = np.average([red_max, blue_max + np.pi])
        elif red_max < blue_max:
            theta = np.average([red_max, blue_max - np.pi])
        theta -= np.pi/2
        semimajor.append(theta)

        bend = np.average([red_max, blue_max])
        if abs(red_max - bend) < np.pi/2:
            if bend < np.pi:
                bend += np.pi
            elif bend > np.pi:
                bend -= np.pi
        bend -= np.pi/2
        near.append(bend)  
    '''
    
    x,y = np.meshgrid(np.arange(img.shape[1]) - cx, np.arange(img.shape[0]) - cy)
    R = np.hypot(x,y)
    R[np.isnan(img)] = None
    theta = np.arctan2(y,x) + 2*np.pi
    theta[theta > 2*np.pi] -= 2*np.pi
    
    rad = np.arange(1, np.nanmax(R), 1)
    semimajor = []
    near = []

    redside = img.copy()
    redside[redside < 0] = None
    blueside = img.copy()
    blueside[blueside > 0] = None

    for i in rad:
        mask = (np.greater(R, i - 1) & np.less(R, i + 1)) | ~np.isnan(img)
        masked_img = img[mask]
        masked_theta = theta[mask]
        masked_redside = redside[mask]
        masked_blueside = blueside[mask]

        if len(masked_redside) > 0 and len(masked_blueside) > 0:
            theta_max_redside = masked_theta[np.nanargmax(masked_redside)]
            theta_max_blueside = masked_theta[np.nanargmin(masked_blueside)]

            bend = np.average([theta_max_redside, theta_max_blueside])
            if abs(theta_max_redside - bend) < np.pi/2:
                if bend < np.pi:
                    bend += np.pi
                elif bend > np.pi:
                    bend -= np.pi
            bend -= np.pi/2
            near.append(bend)
            
            if theta_max_redside > theta_max_blueside:
                theta_max = np.average([theta_max_redside, theta_max_blueside + np.pi])
            elif theta_max_redside < theta_max_blueside:
                theta_max = np.average([theta_max_redside, theta_max_blueside - np.pi])
            theta_max -= np.pi/2
            semimajor.append(theta_max)
            
    Rout = np.nanmax(R)
   
    PA = np.rad2deg(np.average(semimajor))
    
    nearside = np.rad2deg(np.average(near))
    
    return PA, nearside, Rout


