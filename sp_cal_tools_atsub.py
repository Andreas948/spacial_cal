#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 12:08:09 2020

@author: dusch
"""


import gdal
import numpy as np
import os, sys#, datetime
home=os.getenv('HOME')
#from scipy.interpolate import griddata
import GPy
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from netCDF4 import Dataset
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
import scipy.stats
#from timeit import Timer
#from scipy.ndimage.interpolation import zoom
import scipy

linepath=home+"/home_zmaw/phd/programme/surfacetypes/"

def plotlines(m, ax=plt.gca(), path='', rotate=False):
    '''
    function to plot grounding and coast line around Antarctica on a Basemap, 
    based on the MODIS mosaik project 2009. Try not to define the Basemap more 
    often than necessary to speed things up.
    '''
    if not hasattr(m, 'Grounding_Line'):#check if it has already been loaded
        print('loading grounding line')
        if path: glpath=path
        else: glpath='/home/andi/home_zmaw/phd/programme/surfacetypes/'
        m.readshapefile(shapefile=glpath+'moa_2009_groundingline_v11', \
                        name='Grounding_Line', drawbounds=False)
        m.readshapefile(shapefile=glpath+'moa_2009_islands_v11',\
                        name='Grounding_Line_Islands', drawbounds=False)
        m.readshapefile(shapefile=glpath+'moa_2009_coastline_v11', \
                        name='Coast_Line', drawbounds=False)
    if rotate:#this takes forever!
        print('get yourself some coffee ...')
        Coast_line_r=[]
        for x_tmp, y_tmp in m.Coast_Line[0]:
            Coast_line_r.append((y_tmp,x_tmp))
        Grounding_Line_r=[]
        for x_tmp, y_tmp in m.Grounding_Line[0]:
            Grounding_Line_r.append((y_tmp,x_tmp))
        Grounding_Line_Islands_r=[]
        for x_tmp, y_tmp in m.Grounding_Line_Islands[0]:
            Grounding_Line_Islands_r.append((y_tmp,x_tmp))
        Cline = LineCollection([Coast_line_r])
        gline = LineCollection([Grounding_Line_r])
        g2line = LineCollection([Grounding_Line_Islands_r])
    else:
        Cline = LineCollection(m.Coast_Line)
        gline = LineCollection(m.Grounding_Line)
        g2line = LineCollection(m.Grounding_Line_Islands)
    Cline.set_edgecolors('k')#set to other color if you want to distinguish between coast line and grounding line
    Cline.set_linewidth(2)
    ax.add_collection(Cline)
    gline.set_edgecolors('k')
    gline.set_linewidth(2)
    ax.add_collection(gline)
    g2line.set_edgecolors('k')
    g2line.set_linewidth(2)
    ax.add_collection(g2line)
 
def plotASE(x, y, field, label, bmap, save=False, autolim=False, lims= [-8,4], cmap='plasma'):
    fig, ax=plt.subplots()
    if autolim:bmap.pcolormesh(x, y, field, cmap=cmap)
    else: bmap.pcolormesh(x, y, field, cmap=cmap, vmin=lims[0], vmax=lims[1])
    cbar=plt.colorbar()
    cbar.set_label(label, fontsize=16)
    plotlines(bmap, ax=ax, path=linepath)
    ax.set_xlim([4.4357e6, 3.907e6])
    ax.set_ylim([3.4048e6, 2.716e6])
    if save:
        fig.savefig(home+"/Dropbox/mypaper/figures/"+label.replace(" ","").replace("/",""), dpi=400)
    

def find_discr(mean_ens, mean_obs, obs_err, other_err=0.):
    '''
    returns the discrepancy which ensures that 95%(2sig_obs) of a gaussian distributed observational 
    ensemble are within 3 sig_tot=sig_disc from mean_ens. mean_ens can be the mean of the model 
    ensemble OR the closest member of the ensemble, just as you like. If other sigmas 
    are supposed to be included in sig_tot (like potentually emulator uncertainties)
    they can be provided in other_err.
    '''    
    #mean_ens=30
    #mean_obs=2000
    #obs_err=200
    
    mean_dist=abs(mean_ens-mean_obs)
    diffdis=scipy.stats.norm(-mean_dist, obs_err)#problem is symetric and ppf() needs +
    sig_95=2.#sig_95 is something between 1.645 for one sided and 2 for two sided probability tests.
    #it is the numbers of sigmas needed to cover 95% of pdf while taking into account that the dist 
    #might not be centered at 0 but mean_obs!=mean_ens. Dont forget this difference itself
    for i in range(10):
        #print(diffdis.cdf(sig_95*obs_err)-0.95)
        sig_95=0.5*abs((diffdis.ppf(diffdis.cdf(sig_95*obs_err)-0.95)+mean_dist)/obs_err)\
            +0.5*sig_95#
    #print('mean dist: {0:.2f}, obs_err: {1:.3f}'.format(mean_dist, obs_err))
    #print(sig_95)
    sig_disc=np.sqrt(((mean_dist+sig_95*obs_err)/3.)**2-other_err**2)
    return sig_disc
    
def setup_emulator(sim_in, sim_out, s='none', lscales=0.5):
    """Sets up an gaussian process emulator for a given set of n output values sim_out [n]
     for a set of input values sim_in [n x m] where m is the dimension of inputs.
    
    Hyperparameters are optimized at marginal likelyhood with 10 restarts of optimization.
    
    Output: A GPy Gaussian process model, the kernel/covarianze function and
    mean function.
    lscales can be given as scalar or array length dim. Default=0.5 for all
    """
    if np.shape(sim_in)[0]!=len(sim_out):
        if np.shape(sim_in)[1]==len(sim_out):
            sim_in=sim_in.T
        else: print('Wrong Shape!')
    dim=np.shape(sim_in)[1]
    if np.shape(np.shape(sim_out))[0]!=2:
        sim_out=sim_out.reshape([-1,1])
    if s=='none': s=1.
    lengthscales=np.ones(dim)*lscales
    #gpk=GPy.kern.RBF(input_dim=dim, variance=var, lengthscale=lengthscales, ARD=True)
    #gpk=GPy.kern.Exponential(input_dim=dim, variance=var, lengthscale=lengthscales, ARD=True)
    gpk=GPy.kern.Matern52(input_dim=dim, variance=s, lengthscale=lengthscales, ARD=True)
    meanf = GPy.mappings.Constant(input_dim=dim, output_dim=1)
    offset_prior = GPy.priors.Gaussian(mu=0, sigma=0.5)
    #meanf = GPy.mappings.Additive(GPy.mappings.Linear(input_dim=5,output_dim=1),\
    #                                    GPy.mappings.Constant(input_dim=5, output_dim=1))
    gpm=GPy.models.GPRegression(sim_in, sim_out, gpk, noise_var=0., mean_function=meanf)
    gpm.mean_function.set_prior(offset_prior)
    #gpm.constrain_positive()#are constrained anyways
    gpm.Gaussian_noise.variance.fix(0.)
    #gpm.rbf.lengthscale.constrain_bounded(lower=1e-10, upper=2)
    gpm.kern.lengthscale.constrain_bounded(1e-10,4)
    #gpm['constmap.C']=1
    gpm.optimize_restarts(5, messages=1)
    #gpm.optimize(messages=0)
    
    print(gpm[''])
    return gpm, gpk, meanf

def predictXk(vemu, emu_in):
    if np.shape(np.shape(emu_in))[0]==1:
        emu_in=emu_in.reshape([1,-1])
    if np.shape(emu_in)[1]!=5:
        print('ERROR, wrong shape')
        return
    k=len(vemu)
    v_vec, var_v_vec = [], []
    for i in range(k):
        v_tmp, var_vtmp = vemu[i][0].predict(emu_in)
        if np.shape(v_tmp)[1]==1: 
            v_vec.append(v_tmp[:,0])
            var_v_vec.append(var_vtmp[:,0])
        else: print('ERROR, wrong shape 2')
    v_vec=np.asarray(v_vec)
    var_v_vec=np.asarray(var_v_vec)
    return v_vec, var_v_vec
    
    
    
def L_5d(PC_emus, target_PC, eta_target_err, size=[11,11,11, 2, 2], plot=True, \
         log=False, Vpre='None', d3=False, nolin=False, HM=True, k=5, fakeobs=False,\
         Vs='none', central_r='none'):
    #log=False
    #eta_target_err=target_PC_err
    #Vpre='None'
    #size=[11,11,11, 2, 2]
    if Vpre=='None':
        #need to combine eta_target_error with PC consvar
        v1s=np.linspace(0.,1., size[0]) #v1: traction
        v2s=np.linspace(0.,1., size[1]) #v2: viscosity
        v3s=np.linspace(0.,1., size[2]) #v3: melt rate
        v4s=np.linspace(0.,1., size[3]) #v4: sliding
        v5s=np.linspace(0.,1., size[4]) #v5: Bed
                
        if nolin: 
            v4s[1]=0#for m=1/3
        meshv1s, meshv2s, meshv3s, meshv4s, meshv5s = np.meshgrid(v1s, v2s, v3s, v4s, v5s, indexing='ij')
        V=np.vstack([meshv1s.flatten(), meshv2s.flatten(), meshv3s.flatten(),  \
                     meshv4s.flatten(), meshv5s.flatten()]).T
    elif np.shape(Vpre)[1]==5:
        V=Vpre
        meshv1s, meshv2s, meshv3s, meshv4s, meshv5s = np.split(V,5,axis=1)
        size=np.shape(Vpre)[0]
        plot=False
    else:
        print('Something is wrong with V shape, even for 3d=True Vpre must 5d!')
        print(np.shape(Vpre))
    print(np.shape(V))
    print(size)
    PC_cons_fast, PC_cons_var_fast = predictXk(PC_emus, V)
    print(np.shape(PC_cons_fast))
    #print(np.shape(PC_cons_var_fast))
    if 0:#plot hists for each PC
        lscentral_r=np.nonzero( (V[:,0]==1.) & (V[:,1]==1.0) & (V[:,2]==0.0) & (V[:,3]==0) & (V[:,4]==0) )[0][0]
        print(lscentral_r)
        for i in range(np.shape(PC_cons_fast)[0]):
            plt.figure()
            ax=plt.gca()
            if d3: 
                n, bins, patches=ax.hist(PC_cons_fast[i,:] , density=True, label=['1/3'])
            else:
                n, bins, patches=ax.hist([PC_cons_fast[i,:][V[:,3]==1], PC_cons_fast[i,:][V[:,3]!=1]] , density=False, label=['Lin', '1/3'])
            #print(np.max(n))
            plt.plot([target_PC[i], target_PC[i]], [0,1.3*np.max(n)], c='k')
            plt.fill_betweenx([0,1.3*np.max(n)], target_PC[i]-3.*eta_target_err[i], target_PC[i]+3.*eta_target_err[i], color='gray', alpha=0.4)
            #plt.plot([vr[i, central_r], vr[i, central_r]], [0,1.3*np.max(n)], c='r', linestyle='--')#*sr[i]
            plt.plot([PC_cons_fast[i, lscentral_r], PC_cons_fast[i, lscentral_r]], [0,1.3*np.max(n)], c='r', linestyle='--')#*sr[i]
            ax.legend()
            ax.set_ylim([0,1.2*np.max(n)])
            ax.set_xlabel(r'$\omega(\theta)_{}$'.format(i+1), fontsize=16)
            ax.set_ylabel('Frequency', fontsize=16)
            ax.set_title('PC {0}'.format(i+1))
            
            
    #eta_target2d, eta_emu2d = np.meshgrid( PC_cons_var_fast.flatten(), eta_target_err**2)
    print(np.min(PC_cons_var_fast))
    #sigma2d=PC_cons_var_fast.T + eta_target_err**2
    sigma2d=eta_target_err**2
    #plt.figure()
    #plt.hist(PC_cons_var_fast.flatten(), bins=100)
    Ls_log=np.sum(-(PC_cons_fast.reshape([k,np.prod(size)]).T-target_PC.flatten())**2/(2.*sigma2d), axis=1) 
    
    if log: Ls = Ls_log.reshape(size)
    else: Ls=np.exp(Ls_log).reshape(size)

    if HM:
        if 0:#99.5
            if k==4: threesigma2=14.86 #chi-squared, 99.5%, 4DOF, 
            elif k==5: threesigma2=16.7#plt
            elif k==7: threesigma2=20.3
            elif k==8: threesigma2=21.96
            elif k==9: threesigma2=23.589
            elif k==40: threesigma2=66.77
            elif k>=200:threesigma2=320. #approx
        else:#95
            if k==4: threesigma2=9.5 #chi-squared, 99.5%, 4DOF, 
            elif k==5: threesigma2=11.#https://people.richland.edu/james/lecture/m170/tbl-chi.html
            elif k==6: threesigma2=12.6
            elif k==7: threesigma2=14.
            elif k==8: threesigma2=15.5
            elif k==40: threesigma2=55.7
            elif k>=200:threesigma2=220. #approx
        implau=np.zeros(np.prod(size))
        #print(eta_target_err**2)
        #print(sigma2d[0,:])
        for i in range(np.prod(size)):
            #print(np.shape(np.square(PC_cons_fast[:,i]-target_PC.flatten())))
            #print(np.shape(sigma2d))
            implau[i]=np.sum(np.square(PC_cons_fast[:,i]-target_PC.flatten())/sigma2d)#/sigma2d[i,:]
        frac_L=np.sum(np.exp(Ls_log)[implau<=threesigma2])/np.sum(np.exp(Ls_log))
        print('% of L in NROY: {0}'.format(frac_L))
        #print(implau)    
        print('% of Samples with Implausibility <99.5%: {0}'.format(float(np.sum(implau<=threesigma2))/float(np.shape(implau)[0])))
        Ls[implau.reshape(size)>threesigma2]=0.
        
    #print(np.shape(PC_cons_fast.reshape([k,np.prod(size)]).T-target_PC.flatten()))

    #L_5d_plot(implau.reshape(np.shape(Ls))<13.)
    #vx=vx.reshape(size)
    if plot: L_5d_plot(Ls, d3, fakeobs=fakeobs, Vs=Vs, central_r=central_r)
    #L_3d_plot(vx, Cone, bedZero)
    #in_max=np.nonzero(Ls==np.max(Ls))
    #print(V[in_max[0],:])
    #if 3d: #do marginalisation
    if d3:#marginalising
        Ls=Ls.sum(axis=3)#sliding
        Ls=Ls.sum(axis=2)#ocean melt
        Ls_4d=np.zeros([31,31,31,2,2])#carrie equal melt distribution but loose sliding as no linear
        for i in range(np.shape(Ls_4d)[2]):
            for j in range(np.shape(Ls_4d)[3]):
                Ls_4d[:,:,i,j,:]=Ls
        Ls=Ls_4d
        Ls=Ls/np.max(Ls)
        #meshv1s[:,:,0,0,:], meshv2s[:,:,0,0,:], meshv5s[:,:,0,0,:]
        #return Ls, meshv1s[:,:,:,0,:], meshv2s[:,:,:,0,:], meshv3s[:,:,:,0,:], meshv5s[:,:,:,0,:]

    #else:
    return Ls, meshv1s, meshv2s, meshv3s, meshv4s, meshv5s

def L_5d_plot(Ls, d3, mean_val=0, vmax='None', fakeobs=False, Vs='none', central_r='none'):
    #methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
    #           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
    #           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
    #if 3d:
    #    labels=['Traction', 'Viscosity', 'Bedrock']
    #    ndims=3
    #else:
    if d3:
        labels=['Traction', 'Viscosity', 'Basal Melt', 'Sliding', 'Bedrock']
    else:          
        labels=['Traction', 'Visc.', 'B.Melt', 'Sliding', 'Bedrock']
    dims=np.arange(5)
    size=np.shape(Ls)
    #Ls[np.nonzero(Ls!=np.max(Ls))]=0.#check plottig routine
    #fix this
    plt.figure()
    index5dto3di=[0,1,-1,-1]
    index5dto3dj=[0,-1,-1, 1]
    if d3:
        gs3d = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)
        Fig_dummy, ax_dummy = plt.subplots()
    else: 
        gs = gridspec.GridSpec(4, 4, wspace=0.2, hspace=0.2)
    #fig, axs = plt.subplots(ncols=4, nrows=4)
    Ls_max=np.zeros_like(Ls)
    inLsm=np.nonzero(Ls==np.max(Ls))
    Ls_max[inLsm[0][0],inLsm[1][0],inLsm[2][0],inLsm[3][0],inLsm[4][0]]=1#making sure there is only one 1
    print('index-Ls.max = '+str(np.nonzero(Ls==np.max(Ls))))
    #print('xy-central_r='+str(Vs[:,central_r]))
    #print(norm)
    #fig.subplots_adjust(hspace=0.3, wspace=0.05)
    axax1, axax2 = np.meshgrid(range(4), range (4))
    for i, j in zip(axax1.flatten(), axax2.flatten()):
        i3d, j3d = index5dto3di[i], index5dto3dj[j]
        Ls_tmp=Ls.copy()
        Ls_max_tmp=Ls_max.copy()
        yax=np.linspace(-1./(size[i]-1)/2,1.+1./(size[i]-1)/2,size[i]+1)
        xax=np.linspace(-1./(size[j+1]-1)/2,1.+1./(size[j+1]-1)/2,size[j+1]+1)
        marge_dims = dims[np.nonzero(np.logical_or(dims==i,dims ==j+1)==0)]
        for mdim in marge_dims[::-1]:
            Ls_tmp=Ls_tmp.sum(mdim)
            Ls_max_tmp=Ls_max_tmp.sum(mdim)
        if mean_val:
            Ls_tmp=Ls_tmp/(np.asarray(np.shape(Ls))[marge_dims].prod())#/n
            if vmax=='None':
                print('You need to set an maximum if mean values are plotted')
            vmin=-10
        else:
            Ls_tmp=Ls_tmp/np.max(Ls_tmp)#/norm*len(Ls_tmp.flatten())
            vmax=np.max(Ls_tmp)        
            vmin=0
            
            
        if i<=j:
            if d3: 
                #print(i,j, i3d, j3d)
                if i3d>=0 and j3d>=0:
                    ax=plt.subplot(gs3d[i3d, j3d])

                else: ax=ax_dummy
            else:
                ax=plt.subplot(gs[i, j])
            pcol=ax.pcolor(xax, yax, Ls_tmp, cmap='gnuplot', vmin=vmin, vmax=vmax)
            #plt.colorbar(pcol, ax=ax)
            #print(np.max(Ls_tmp/norm)*len(Ls_tmp.flatten()))    
            #print(np.min(Ls_tmp))
            if 1:#plot Ls.max and central_r
                
                yax_max, xax_max = np.nonzero(Ls_max_tmp==1)
                #if d3 and len(yax_max)==2:yax_max=yax_max[0]#sliding law is the same in both cases
                #if d3 and len(xax_max)==2:xax_max=xax_max[0]
                if np.shape(xax)[0]==3:
                    if xax_max==0: scx=0.25
                    elif xax_max==1: scx=0.75
                else: scx=(xax[xax_max]+xax[xax_max+1])/2.
                if np.shape(yax)[0]==3:
                    if yax_max==0: scy=0.25
                    elif yax_max==1: scy=0.75
                else: scy=(yax[yax_max]+yax[yax_max+1])/2.      
                ax.scatter(scx, scy, c='g', marker='+', s=80)
                #center_r
                if fakeobs:
                    xax_cent=Vs[j+1,central_r]
                    yax_cent=Vs[i,central_r]
                    if np.shape(xax)[0]==3:
                        if xax_cent==0.: scx=0.25
                        elif xax_cent==1.: scx=0.75
                        else:print('HM?')
                    else: scx=xax_cent
                    if np.shape(yax)[0]==3:
                        if yax_cent==0.: scy=0.25
                        elif yax_cent==1.: scy=0.75
                        else:print('HM?')
                    else: scy=yax_cent
                    ax.scatter(scx, scy, facecolor='none', edgecolor='k', s=80)
        if d3: 
            if i3d<j3d and i3d>=0: 
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                print(i3d*10+j3d)
            if i3d==j3d: 
                ax.set_ylabel(labels[i], fontsize=14)
                ax.set_xlabel(labels[j+1], fontsize=14)
                if j3d==1:     
                        ax.set_xticks([0.25, 0.75])
                        ax.set_xticklabels(['Mod', 'BM-2'])        
        else:
            
            if i<j: 
                ax.set_yticklabels([])
                ax.set_xticklabels([])
            if i==j: 
                ax.set_ylabel(labels[i], fontsize=14)
                ax.set_xlabel(labels[j+1], fontsize=14)
                if i>=3: 
                    ax.set_yticks([0.25, 0.75])
                    ax.set_yticklabels(['1/3', '1'])
                if j==2: 
                    ax.set_xticks([0.25, 0.75])
                    ax.set_xticklabels(['1/3', '1'])
                if j==3: 
                    ax.set_xticks([0.25, 0.75])
                    ax.set_xticklabels(['Mod', 'BM-2'])
                    

        #ax.set_title(interp_method)
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
    #plt.colorbar(pcol, ax=ax)
    
    if 0: ax = plt.subplot(gs[-2:, :2])
    else: 
        if d3: 
            #plt.close(Fig_dummy)
            ax=plt.subplot(gs3d[1, 0])
        else:
            fig, ax = plt.subplots()
            
    norm=np.sum(Ls)
    if d3: 
        legsize=10
        labsize=14
        postext=0.5
    else:
        legsize=14
        labsize=16
        postext=0.6
        
    for i in range(5):
        Ls_tmp=Ls.copy()
        marge_dims = dims[dims!=i]
        for mdim in marge_dims[::-1]:
            Ls_tmp=Ls_tmp.sum(mdim)
        if mean_val:
            Ls_tmp=Ls_tmp/(np.asarray(np.shape(Ls))[marge_dims].prod())#/n
        else:
            Ls_tmp=Ls_tmp/norm*size[i]#/norm*len(Ls_tmp.flatten())
            if i>=3: Ls_tmp=Ls_tmp/2.#reversing the integral=1 part to give %
            vmax=np.max(Ls_tmp)  
        if i<=1: ax.plot(np.linspace(0.,1.,size[i]), Ls_tmp, label = labels[i])
        if not d3 and i==2: ax.plot(np.linspace(0.,1.,size[i]), Ls_tmp, label = labels[i])
        if mean_val:
            if i==3:
                ax.text(0.03, 0.09, 'Sliding Law: Linear (1/3) = {:.2f} ({:.2f}) mm'\
                        .format(Ls_tmp[1], Ls_tmp[0]), transform=ax.transAxes)
            if i==4:
                ax.text(0.03, 0.04, 'Bedrock: Modified (BM-2)= {:.2f} ({:.2f}) mm'\
                        .format(Ls_tmp[0], Ls_tmp[1]), transform=ax.transAxes)
                
        else:
            if i==3 and not d3:     
                Ls_linslide=Ls_tmp[1]
                ax.text(postext, 0.86, 'L(Lin. Sliding)={:02d}%'.format\
                        (int((Ls_linslide*100.).round())), transform=ax.transAxes, fontsize=13)
            if i==4: 
                Ls_modbed=Ls_tmp[0]
                ax.text(postext, 0.93, 'L(Mod. Bed)={:02d}%'.format\
                        (int((Ls_modbed*100.).round())), transform=ax.transAxes, fontsize=13)

    #print('{0}: {1}/{2}'.format(i, Ls_linslide, Ls_modbed))
    #print(type(round(Ls_linslide[0])))

    ax.legend(loc='upper left', fontsize=legsize)
    ax.set_ylim(bottom=0)
    ax.set_xlim([0,1])
    ax.set_xlabel('Parameter Value', fontsize=labsize)
    if mean_val:
        ax.set_ylabel('Sea Level Contribution [mm]', fontsize=labsize)
    else:
        ax.set_ylabel('Likelihood', fontsize=labsize)
    #cbar=plt.colorbar(pcol, ax=ax)
    #cbar.set_label('Sea Level Contribution [mm]', fontsize=15)
    #plt.savefig(home+"/Dropbox/mypaper/figures/05050500.png", dpi=300)
    #plt.savefig(home+"/Dropbox/mypaper/figures/L3d.png", dpi=300)
    

def inflate2grid(dhdtm_test, mask_2d):
    dhdt_test=np.ma.array(np.zeros_like(mask_2d, dtype=float), mask=mask_2d)
    dhdt_test[mask_2d==0]=dhdtm_test
    return dhdt_test
    
    
