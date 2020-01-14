#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 12:05:57 2020

@author: dusch
"""

import gdal
import numpy as np
import os, sys#, datetime
home=os.getenv('HOME')
#from scipy.interpolate import griddata
genpath=home+'/Dropbox/scripts/rep/Calibration_ASE/'
sys.path.append(genpath)
import sp_cal_tools_atsub as spt

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

#%%%
#options

calyear=10.#last year of calibration period
spinnup=True #start calibration later than year 0?
if spinnup:
    styear=3. #first year of calibration period
    period=calyear-styear
else:
    period=calyear

fakeobs=False #synthetik model test
k= 5 #truncation value (=95.9%)


#%%%
#Directories of datasets
datadir= home+"/Dropbox/mypaper/"
largedatadir=home+"/outsource_Dropbox/Ice_sheet_velocitys/"

   
print('Calibartion on year {:01d}'.format(int(calyear)))

#loading prepared model data (preprocess_AMR.py)
dhdt=np.load(datadir+"dhdt_centered_{:02d}_v004.npy".format(int(calyear))) 
dhdt50=np.load(datadir+"dhdt_centered_ogrid_50_v001.npy") 
#\bold\tile{{Y}} mXn matrix with cerntered calyear and 50 year (temporal) mean model output centered by: 

dhdt_mean=np.load(datadir+"dhdt_mean_{:02d}_v004.npy".format(int(calyear)))
dhdt50_mean=np.load(datadir+"dhdt_mean_ogrid_50_v001.npy")
# \bar{Y}, m vector of mean across ensemble mebers

Vs=np.load(datadir+"Vs_{:02d}_v002.npy".format(int(calyear)))
Vs50=np.load(datadir+"Vs_50_v002.npy")
# dXn matrix of input parameters for each ensemble member

x, y = np.load(datadir+"xy_v003.npy")
x_ogrid, y_ogrid = np.load(datadir+"xyogrid_v001.npy")
#2d spatial locations for observational grid (top) and original/model grid (ogrid)

ASEcatch_ogrid = np.load(datadir+"ASEmaskogrid_v001.npy") 
ASEcatch_obsgrid = np.load(datadir+"ASEmaskobsgrid_v001.npy") 
#Amundsen chatchement mask on both grids

vaft0mb = np.load(datadir+"VAFt0mb_v001.npy") 
vaft0bm2 = np.load(datadir+"VAFt0bm2_v001.npy")
vaft0mb_ogrid = np.load(datadir+"VAFt0mbogrid_v001.npy") 
vaft0bm2_ogrid = np.load(datadir+"VAFt0bm2ogrid_v001.npy")
#2d matrix of ice volume above flotation at t=0 
#modified bed=mb ; Bedmap2=bm2, obs (top) and model grid (ogrid)

#Loading the observations
with Dataset(datadir+'dhdt_10km_sig0p5_0p0.nc') as f:
    x_obs=f.variables['x'][:]
    y_obs=f.variables['y'][:]

    year_obs=f.variables['t'][:]
    dhdt_obs=f.variables['dhdt'][:,:,:]
dhdt_obs.mask=np.isnan(dhdt_obs.data)

#getting rates of ice thickness change for calibration period if spinnup is used
if spinnup:
    stVs=np.load(datadir+"Vs_{:02d}_v002.npy".format(int(styear)))
    stdhdt_mean=np.load(datadir+"dhdt_mean_{:02d}_v004.npy".format(int(styear)))
    stdhdt=np.load(datadir+"dhdt_centered_{:02d}_v004.npy".format(int(styear))) 
    dhdt_comb=np.zeros_like(dhdt)
    for i in range(np.shape(Vs)[1]):#ensemble members
        j=0
        while (stVs[:,j]==Vs[:,i]).all()==False: j+=1
        dhdt_comb[:,i]=(dhdt[:,i]+dhdt_mean)*calyear-(stdhdt[:,j]+stdhdt_mean)*styear#rates into absolute, subtract
    dhdt_comb=dhdt_comb/period#and back to rate
    dhdt_mean=np.mean(dhdt_comb, axis=1)
    dhdt=np.subtract(dhdt_comb.T, dhdt_mean).T
    
x_obs, y_obs = np.meshgrid(x_obs*1000., y_obs*1000.)
    
#index of on specific member
central_r=np.nonzero( (Vs[0]==0.5) & (Vs[1]==0.5) & (Vs[2]==0.5) & (Vs[3]==0) & (Vs[4]==0) )[0][0]
    

if fakeobs: #fake obs from central run with noise
    nomodel=np.min(dhdt, axis=1)==np.max(dhdt, axis=1)
    mask_fobs=np.ones([14,72,54], dtype=bool)*np.logical_or(nomodel.reshape([72,54]), dhdt_obs.mask[-1,:,:] )

    if 1:#spatial variance
        var_2d= np.ma.var(dhdt_obs[-14:,:,:], axis=0)
        if 0:#plot variance spacially
            plt.figure()
            plt.imshow(var_2d)
            plt.colorbar()
            
        dhdt_obs = np.ones([14,72,54])*(dhdt[:,central_r]+dhdt_mean).reshape([72, 54])
        
        for i in range(14):
            dhdt_obs[i,:,:]+=np.random.normal(np.zeros_like(var_2d),np.sqrt(var_2d))
        dhdt_obs=np.ma.array(dhdt_obs, mask=mask_fobs)
        
    
#obs_mean will only be used to illustrate impact of dim reduction
obs_mean=np.mean(dhdt_obs[-1*int(period)*2:,:,:], axis=0)

#mask of locations with missing observations throuth the time series
noobs=np.logical_or(obs_mean.mask,ASEcatch_obsgrid==0)
sp_mask = noobs.flatten() #spatial mask in 1d

#finding n
nruns=np.shape(dhdt)[1]
mspace=np.shape(dhdt_mean)[0]

#singular value decomposition on grid of and for lacations where we have observations
u,s,v = np.linalg.svd(dhdt[sp_mask==0], full_matrices=False)
#dhdt=u.dot(v_vec*s)
u=u*s#turning U into B
#dhdt=u.dot(v_vec)

if 0: #plot s^2 as function of PCs
    s2=s**2
    fig, ax=plt.subplots(1, figsize=[4, 4.8])
    fig.subplots_adjust(left=0.15, right=0.84)
    ax.bar(np.arange(1,len(s2)+1),s2/np.sum(s2), color='gray', width=1.)
    
    ax.set_xlabel('# PC', fontsize=18)
    ax.set_ylabel('Fraction of Total Variance', fontsize=16)
    ax.set_xlim([0.5,10.5])
    ax.set_xticks(np.arange(1,11))
    ax.set_xticklabels(np.arange(1,11))
    
    ax_c=ax.twinx()
    ax_c.set_xlim([0.5,10.5])
    cums=np.array([np.sum(s2[:i]/np.sum(s2)) for i in range(len(s2))])
    ax_c.step(np.arange(0.5,len(s2)+0.5),cums, color='r',linewidth=2)
    ax_c.set_yticks([0,0.2,0.4,0.6,0.8,1])
    ax_c.set_yticklabels(['0','0.20', '0,40', '0,60', '0.80', '1.00'])
    ax_c.set_ylim([0,1])
    #ax_c.yaxis.label.set_color('g')
    ax_c.tick_params(axis='y', colors='r')
    ax_c.set_ylabel('Cumulative Fraction of Variance', fontsize=16, color='r')
    #fig.savefig(home+"/Dropbox/mypaper/figures/Variance_v2.png", dpi=300)
    
    
#tuncation
ur,sr,vr = u[:,:k], s[:k], v[:k,:]
    
    
#define basemap and center to South pole
m=Basemap(width=5400000., height=5400000., projection='stere',\
          ellps='WGS84', lon_0=180., lat_0=-90., lat_ts=-71., resolution='i')#, suppress_ticks=False)
              
#projection coordinates of south pole                
x_sp, y_sp = m(0, -90)

#the base (x,y) coordinates of both, observations and model
xbase=x_sp-x
ybase=y_sp-y

xbase_ogrid=x_sp-x_ogrid
ybase_ogrid=y_sp-y_ogrid


#finding inverse to reproject Observations to PCs

#invert ur
linv_A = np.linalg.solve(u.T.dot(u), u.T)

if 0:#test inversion
    sv=linv_A.dot(dhdt[sp_mask==0,central_r])
    dhdt_tmp_test = u.dot(sv)
    plt.figure()
    plt.hist(dhdt[sp_mask==0,central_r]-dhdt_tmp_test)#, range=[-0.5,0.5])#

   
    
#finding obs PCs contributions (and error) by comparing all timesteps
target_PCs=[]
SLE_obss=[]
nobs=np.shape(dhdt_obs)[0]
for i in range(nobs-1*int(period)*2, nobs):
    #preparing obs (removing mean field so that they are comparable to model PCs)
    obs_tmp=dhdt_obs[i,:,:].flatten()-dhdt_mean
    target_PCs.append(linv_A.dot(obs_tmp[sp_mask==0])[:k])

    mask_tmp=np.logical_or(dhdt_obs[i,:,:].mask, ASEcatch_obsgrid==0).flatten()==0
    VAF_tmp=vaft0bm2[mask_tmp]+dhdt_obs[i,:,:].flatten()[mask_tmp]*10000.*10000.*10**(-9)
    #vaft0mb
    dVAF_tmp=np.ma.sum(np.ma.array(VAF_tmp, mask=VAF_tmp<0)) - \
        np.ma.sum(np.ma.array(vaft0bm2[mask_tmp], mask=vaft0bm2[mask_tmp]<0))
    SLE_obss.append(-dVAF_tmp*0.917/361.8)
    #plotASE(xbase, ybase, obs_tmp.reshape([72,54]), label='obs')
    #target_PCs.append(linv_A_full.dot(obs_tmp[sp_mask==0]))
target_PCs=np.asarray(target_PCs)
target_PC=target_PCs.mean(axis=0)
target_PC_err=target_PCs.std(axis=0)
SLE_obs=np.mean(np.asarray(SLE_obss))
#SLE_obs=0.33
SLE_obs_err=np.std(np.asarray(SLE_obss))

    
if 1: #and some plotting
    save=0
    #PCs
    spt.plotASE(xbase, ybase, spt.inflate2grid(ur[:,0]/s[0], noobs), "PC 1", lims=[-0.25, 0.25], cmap='seismic', save=save, bmap=m)
    spt.plotASE(xbase, ybase, spt.inflate2grid(ur[:,1]/s[1], noobs), "PC 2", lims=[-0.25, 0.25], cmap='seismic', save=save, bmap=m)
    spt.plotASE(xbase, ybase, spt.inflate2grid(ur[:,2]/s[2], noobs), "PC 3", lims=[-0.25, 0.25], cmap='seismic', save=save, bmap=m)
    spt.plotASE(xbase, ybase, spt.inflate2grid(ur[:,3]/s[3], noobs), "PC 4", lims=[-0.25, 0.25], cmap='seismic', save=save, bmap=m)
    #plotASE(xbase, ybase, inflate2grid(ur[:,4]/s[4], noobs), "PC 5", lims=[-0.25, 0.25], cmap='seismic', save=save)

    
    spt.plotASE(xbase, ybase, obs_mean, "dh/dt [m/year]", lims=[-8, 8], cmap='seismic', save=save, bmap=m)
    #plotASE(xbase, ybase, dhdt_mean.reshape([72, 54])+dhdt[:,central_r].reshape([72, 54]), "dh/dt [m/year]", lims=[-8, 8], cmap='seismic', save=save)
    obs_reprojected=dhdt_mean.reshape([72,54])+spt.inflate2grid(ur.dot(target_PC), noobs)
    spt.plotASE(xbase, ybase,obs_reprojected,"dh/dt Reprojected [m/year]", lims=[-8, 8], cmap='seismic', save=save, bmap=m)
            
    print(np.ma.var(ur.dot(target_PC) - (obs_mean.flatten()[sp_mask==0]-dhdt_mean[sp_mask==0]))\
          /np.ma.var(obs_mean.flatten()[sp_mask==0]-dhdt_mean[sp_mask==0]))

    
#%%% Errors (obs and discrepancy)

var_obs=target_PC_err[0]**2/sr**2*sr[0]**2#

var_discs=np.zeros(k)
for i in range(k):
    var_discs[i]=spt.find_discr(np.mean(vr[i,:]),np.mean(target_PCs[:,i]), \
           np.sqrt(var_obs[i]), other_err=0.)**2#*sr[i]
    
var_discs=np.max(np.array([var_discs,1.*var_obs]), axis=0)

sig_tot=np.sqrt(var_discs+var_obs)#

if 0:#plot hists of pcs
    fig, axes = plt.subplots(3,2, figsize=(12,5))
    axes=axes.flatten()
    for i in range(np.min([k,6])):
        axes[i].plot([target_PC[i],target_PC[i]], [0,20], c='b')
        axes[i].plot([target_PC[i]-3.*sig_tot[i], target_PC[i]-3*sig_tot[i]], [0,10], c='gray')
        axes[i].plot([target_PC[i]+3.*sig_tot[i], target_PC[i]+3*sig_tot[i]], [0,10], c='gray')
        axes[i].hist(vr[i,:], color='orange', bins=10)
        axes[i].text(0.025, 0.9, 'PC '+str(i+1), transform=axes[i].transAxes, fontsize=14)


    
#%%% train emulator

PC_emus=[]
for i in range(k):
    PC_emus.append(spt.setup_emulator(Vs, vr[i,:], s='none'))#sr[i]*
    
    
#%%% Calculate and plot L
   
print('Calculating likelihoods')

if fakeobs:
    Ls, v1s, v2s, v3s, v4s, v5s =spt.L_5d(PC_emus, target_PC, eta_target_err=sig_tot,size=[31,31,31,2,2],\
        d3=False, nolin=False, HM=True, k=k, fakeobs=fakeobs, Vs=Vs, central_r=central_r)

else:
    Ls, v1s, v2s, v3s, v4s, v5s =spt.L_5d(PC_emus, target_PC, eta_target_err=sig_tot,size=[31,31,31,2,2],\
        d3=True, nolin=True, HM=True, k=k)

print(Vs[:,central_r])
print(np.nonzero(Ls==np.max(Ls)))
#L_5d_plot(meshv1s)
#L_PC(eta_traget, eta, eta_target_err=1., log=False)
#sys.exit()

if 0: #plot parameters vs Ls
    for vxs in [v1s, v2s, v3s]:
        plt.figure()
        plt.scatter(vxs[v4s==1.], Ls[v4s==1.], c='r')
        plt.scatter(vxs[v4s!=1.], Ls[v4s!=1.],  c='k')
        plt.xlabel('VsX')
        plt.ylabel('Ls')
#def 2dVAF(PC_emus, V, ur, dhdt_mean, VAFt0, years=50


if 0: #this fucks up uncalibrated values but speeds up/allows very large emulator ensembles
    Ls_mask=(Ls>np.sum(Ls)*10**(-7)).flatten()
    Ls, v1s, v2s, v3s, v4s, v5s = Ls.flatten()[Ls_mask], v1s.flatten()[Ls_mask], \
        v2s.flatten()[Ls_mask], v3s.flatten()[Ls_mask], v4s.flatten()[Ls_mask], v5s.flatten()[Ls_mask]
    print(np.sum(Ls_mask))

#V at which we calculated Ls, could be everything
V_Ls=np.vstack([v1s.flatten(), v2s.flatten(), v3s.flatten(), v4s.flatten(), v5s.flatten()]).T

prob=Ls.flatten() 
prob=prob/np.sum(prob)


print('Emulating dhdt and VAF for calibration period')

PC_cons, PC_cons_var = spt.predictXk(PC_emus, V_Ls)
mask_tmp=noobs.flatten()==0

    
if 1:#this is slower but avoids memory errors
    dhdt_tmp_tmp=np.zeros([np.shape(ur)[0]])
    VAF_tmp=np.zeros_like(dhdt_tmp_tmp)
    dVAF7=np.zeros(np.shape(V_Ls)[0])
    RMSE7=np.zeros(np.shape(V_Ls)[0])
    for i in range(np.shape(V_Ls)[0]):#emulator evaluations
        dhdt_tmp_tmp=ur.dot(PC_cons[:,i]) + dhdt_mean[mask_tmp]
        if V_Ls[i,4]: VAFt0_tmp=vaft0bm2#find the right volume above fllotation at t=0 as baseline
        else: VAFt0_tmp=vaft0mb
        VAF_tmp=VAFt0_tmp[mask_tmp]+dhdt_tmp_tmp/10.*period#*10e3*10e3*1e-9*7.#10km*10km;to km3, times years
        #^add total change to VAF in 2d 
        dVAF7[i]=np.ma.sum(np.ma.array(VAF_tmp, mask=VAF_tmp<0)) - \
            np.ma.sum(np.ma.array(VAFt0_tmp[mask_tmp], mask=VAFt0_tmp[mask_tmp]<0))
            #^sum all positive VAF accros area and subtract the initial
        RMSE7[i]=np.sum(np.square(dhdt_tmp_tmp - obs_mean.data.flatten()[mask_tmp]))#for (xy) calibration lateron

dSLE7=-dVAF7*0.917/361.8/period

#%%% (xy) and SLE calibrations

Ls_dSLE=np.exp(-(dSLE7-SLE_obs)**2/(2.*3.*0.035**2))#SLE_obs_err=0.035
Ls_RMSE=np.exp(-RMSE7/2./(3.*target_PC_err[0]**2*sr[0]**2))

if 0:
    print('Constraining Ls dSLE and RMSE to modfied bedrock')
    #just to see how much of the difference can be explained by the bedrock constrain.
    #Sliding law is the same as for basis calib anyways (because v4s is all 0 if nolin=True)
    Ls_dSLE[V_Ls[:,4]==1]=0.
    Ls_RMSE[V_Ls[:,4]==1]=0.
    
Ls_dSLE=Ls_dSLE.reshape([31,31,31,2,2])
Ls_RMSE=Ls_RMSE.reshape([31,31,31,2,2])
    
if fakeobs:
    spt.L_5d_plot(Ls_dSLE, d3=0)
    spt.L_5d_plot(Ls_RMSE, d3=0)
else:
    spt.L_5d_plot(Ls_dSLE, d3=1)
    spt.L_5d_plot(Ls_RMSE, d3=1)

#L_5d_plot(Ls_dSLE.reshape([31,31,31,1,2]), d3=1)
#L_5d_plot(dSLE7.reshape([11,11,11,2,2]))

prob_dSLE=Ls_dSLE.flatten()/np.sum(Ls_dSLE)
prob_RMSE=Ls_RMSE.flatten()/np.sum(Ls_RMSE)

#%%% Projections
#dSLE for BISICLES runs
dhdt_tmp_tmp=np.zeros(np.sum(ASEcatch_ogrid))
VAF_tmp=np.zeros_like(dhdt_tmp_tmp)
dVAF_BI=np.zeros(np.shape(Vs50)[1])
for i in range(np.shape(Vs50)[1]):
    dhdt_tmp_tmp=(dhdt50[:,i] + dhdt50_mean)[ASEcatch_ogrid.flatten()]
    if Vs50[4,i]: VAFt0_tmp=vaft0bm2_ogrid
    else: VAFt0_tmp=vaft0mb_ogrid
    VAF_tmp=VAFt0_tmp[ASEcatch_ogrid.flatten()]+dhdt_tmp_tmp*4000.*4000.*10**(-9)*50.#4kmres to km3*years
    dVAF_BI[i]=np.ma.sum(np.ma.array(VAF_tmp, mask=VAF_tmp<0)) - \
        np.ma.sum(np.ma.array(VAFt0_tmp[ASEcatch_ogrid.flatten()], mask=VAFt0_tmp[ASEcatch_ogrid.flatten()]<0))
dSLE_BI=-dVAF_BI*0.917/361.8/50.#to GT, to mmSLE, as yearly rate
    
#and emulate the total SLC
dSLE_emu50 = spt.setup_emulator(Vs50, dSLE_BI, s='none')
dSLE_tmp=dSLE_emu50[0].predict(V_Ls)
dSLE, dSLE_var = dSLE_tmp[0][:,0], dSLE_tmp[1][:,0]
            
if 0:#very simple plot of SLE
    fig, ax=plt.subplots()
    plt.hist(dSLE7, weights=prob, bins=30, density=True, color='b')
    plt.hist(dSLE7, density=True, alpha=0.6, bins=30, color='cornflowerblue')
    ax.set_xlabel('Mean Rate of Sea Level Contribution [mm SLE/year]')
    ax.set_ylabel('Probability')    
    
if 1:#proper plot of all calibs
    figx, axx=plt.subplots()
    if 1:#year 50
        dSLE_tmp=dSLE*50
        vmin=-20
        vmax=120.
        ymax=0.175
        varm=1.5**2
        axx.hist(dSLE_BI[Vs50[3,:]==0]*50., density=True,  alpha=0.4, bins=30, color='brown', range=(-20,120))#cornflowerblue
    else:#calibration period
        dSLE_tmp=dSLE7*period
        vmin=-10
        vmax=20.
        ymax=1.28
        varm=0.1**2
        axx.plot([SLE_obs*7., SLE_obs*7.], [0., 1.], c='gray', linestyle=':')

    hist2=axx.hist(dSLE_tmp, density=True,  alpha=0.6, bins=30, color='black', range=(vmin,vmax))#
    axx.set_xlabel('Sea Level Contribution after 50 years [mm SLE]', fontsize=14)
    axx.set_ylabel('Probability', fontsize=14)

    xm=np.linspace(np.min(dSLE_tmp)-6., np.max(dSLE_tmp)+6., 200)
    ym1=np.zeros_like(xm)
    ym2=np.zeros_like(xm)
    ym3=np.zeros_like(xm)
    ym4=np.zeros_like(xm)
   
    for i in range(len(dSLE_tmp)):
        ym1=ym1+1./len(dSLE_tmp)/np.sqrt(2.*np.pi*varm)*np.exp(-(xm-dSLE_tmp[i])**2/varm/2.)
        ym2=ym2+prob[i]/np.sqrt(2.*np.pi*varm)*np.exp(-(xm-dSLE_tmp[i])**2/varm/2.)
        ym3=ym3+prob_dSLE[i]/np.sqrt(2.*np.pi*varm)*np.exp(-(xm-dSLE_tmp[i])**2/varm/2.)
        ym4=ym4+prob_RMSE[i]/np.sqrt(2.*np.pi*varm)*np.exp(-(xm-dSLE_tmp[i])**2/varm/2.)
    axx.plot(xm, ym1, 'k', label='Prior', linewidth=2)
    axx.plot(xm, ym2, c='blue', linestyle='--', label='Calibrated (Basis)', linewidth=2)
    axx.plot(xm, ym4, c='red', linestyle=':', label='Calibrated (x,y)', linewidth=2)
    axx.plot(xm, ym3, c='green', linestyle='-.', label='Calibrated (SLE)', linewidth=2)

    print('Mode Prior: '+str(xm[ym1==np.max(ym1)]))
    print('Mode Post: '+str(xm[ym4==np.max(ym4)])) 
    axx.set_xlim([vmin, vmax])
    axx.set_ylim([0,ymax])
    axx.legend(fontsize=12)
    #figx.savefig(home+"/Dropbox/mypaper/figures/SLC50_tot.png", dpi=300)

if 1:#all the numbers (quantiles etc) you need
    print("OBSERVATIONAL PERIOD (basis):")
    dSLE_sorted7=np.sort(dSLE7)
    print('Uncal: '+str(np.mean(dSLE_sorted7)) +' +-'+str(np.std(dSLE_sorted7))+'mmSLE/a')
    quan=np.array(np.array([0.05, 0.25, 0.5, 0.75, 0.95])*len(dSLE_sorted7), dtype=int)
    print('% Uncal [5, 25, 50, 75, 95]: '+str(dSLE_sorted7[quan]))
    print('Calibrated: '+str(np.sum(dSLE7*prob))) 
    
    dSLE7inds=np.argsort(dSLE7)
    quantiles=prob[dSLE7inds].cumsum()
    quan=np.array([np.min(np.nonzero(quantiles>q)[0]) for q in\
        np.array([0.05, 0.25, 0.5, 0.75, 0.95])])
    print('% Calibrated [5, 25, 50, 75, 95]: '+str(dSLE7[dSLE7inds][quan]))
    #print('Average change in traction : {0:.2f}%'.format((2.**(2.*np.sum(v1s.flatten()*prob)-1.)-1.)*100.))
    #print('Average change in viscosity : {0:.2f}%'.format((2.**(2.*np.sum(v2s.flatten()*prob)-1.)-1.)*100.))
    
    print('')
    print("PREDICTIONS (basis):")
    dSLE_sorted=np.sort(dSLE)
    print('Uncal: '+str(np.mean(dSLE_sorted)*50.) +' +-'+str(np.std(dSLE_sorted))+'mmSLE/a')
    quan=np.array(np.array([0.05, 0.25, 0.5, 0.75, 0.95])*len(dSLE_sorted), dtype=int)
    print('% Uncal [5, 25, 50, 75, 95]: '+str(dSLE_sorted[quan]*50.))
    print('Calibrated: '+str(np.sum(dSLE*prob)*50.)) 
    
    skewness=np.sum(prob*((dSLE-np.mean(dSLE))/np.std(dSLE))**3)
    print('Skewness: {:.2f}'.format(skewness))
    dSLEinds=np.argsort(dSLE)
    quantiles=prob[dSLEinds].cumsum()
    quan=np.array([np.min(np.nonzero(quantiles>q)[0]) for q in\
        np.array([0.05, 0.25, 0.5, 0.75, 0.95])])
    print('% Calibrated [5, 25, 50, 75, 95]: '+str(dSLE[dSLEinds][quan]*50.))
    
    print('SLE year 7 for MAX(L(theta)): '+str(dSLE7[prob==np.max(prob)]))
    print('SLE year 50 for MAX(L(theta)): '+str(dSLE[prob==np.max(prob)]*50))
    
    print('')
    print('dSLE:')
    print("OBSERVATIONAL PERIOD (dSLE):")
    dSLE_sorted7=np.sort(dSLE7)
    print('Uncal: '+str(np.mean(dSLE_sorted7)) +' +-'+str(np.std(dSLE_sorted7))+'mmSLE/a')
    quan=np.array(np.array([0.05, 0.25, 0.5, 0.75, 0.95])*len(dSLE_sorted7), dtype=int)
    print('% Uncal [5, 25, 50, 75, 95]: '+str(dSLE_sorted7[quan]))
    print('Calibrated: '+str(np.sum(dSLE7*prob_dSLE))) 
    
    dSLE7inds=np.argsort(dSLE7)
    quantiles=prob_dSLE[dSLE7inds].cumsum()
    quan=np.array([np.min(np.nonzero(quantiles>q)[0]) for q in\
        np.array([0.05, 0.25, 0.5, 0.75, 0.95])])
    print('% Calibrated [5, 25, 50, 75, 95]: '+str(dSLE7[dSLE7inds][quan]))

    print('')
    print("PREDICTIONS (dSLE):")
    dSLE_sorted=np.sort(dSLE)
    print('Uncal: '+str(np.mean(dSLE_sorted)*50.) +' +-'+str(np.std(dSLE_sorted))+'mmSLE/a')
    quan=np.array(np.array([0.05, 0.25, 0.5, 0.75, 0.95])*len(dSLE_sorted), dtype=int)
    print('% Uncal [5, 25, 50, 75, 95]: '+str(dSLE_sorted[quan]*50.))
    print('Calibrated: '+str(np.sum(dSLE*prob_dSLE)*50.)) 
    dSLEinds=np.argsort(dSLE)
    quantiles=prob_dSLE[dSLEinds].cumsum()
    quan=np.array([np.min(np.nonzero(quantiles>q)[0]) for q in\
        np.array([0.05, 0.25, 0.5, 0.75, 0.95])])
    print('% Calibrated [5, 25, 50, 75, 95]: '+str(dSLE[dSLEinds][quan]*50.))
    print('SLE year 7 for MAX(L(theta)): '+str(dSLE7[prob_dSLE==np.max(prob_dSLE)]))
    print('SLE year 50 for MAX(L(theta)): '+str(dSLE[prob_dSLE==np.max(prob_dSLE)]*50))
 
    
    print('')
    print('(xy):')
        
    print("OBSERVATIONAL PERIOD (xy):")
    dSLE_sorted7=np.sort(dSLE7)
    print('Uncal: '+str(np.mean(dSLE_sorted7)) +' +-'+str(np.std(dSLE_sorted7))+'mmSLE/a')
    quan=np.array(np.array([0.05, 0.25, 0.5, 0.75, 0.95])*len(dSLE_sorted7), dtype=int)
    print('% Uncal [5, 25, 50, 75, 95]: '+str(dSLE_sorted7[quan]))
    print('Calibrated: '+str(np.sum(dSLE7*prob_RMSE))) 
    
    dSLE7inds=np.argsort(dSLE7)
    quantiles=prob_RMSE[dSLE7inds].cumsum()
    quan=np.array([np.min(np.nonzero(quantiles>q)[0]) for q in\
        np.array([0.05, 0.25, 0.5, 0.75, 0.95])])
    print('% Calibrated [5, 25, 50, 75, 95]: '+str(dSLE7[dSLE7inds][quan]))

    print('')
    print("PREDICTIONS (xy):")
    dSLE_sorted=np.sort(dSLE)
    print('Uncal: '+str(np.mean(dSLE_sorted)*50.) +' +-'+str(np.std(dSLE_sorted))+'mmSLE/a')
    quan=np.array(np.array([0.05, 0.25, 0.5, 0.75, 0.95])*len(dSLE_sorted), dtype=int)
    print('% Uncal [5, 25, 50, 75, 95]: '+str(dSLE_sorted[quan]*50.))
    print('Calibrated: '+str(np.sum(dSLE*prob_RMSE)*50.)) 
    dSLEinds=np.argsort(dSLE)
    quantiles=prob_RMSE[dSLEinds].cumsum()
    quan=np.array([np.min(np.nonzero(quantiles>q)[0]) for q in\
        np.array([0.05, 0.25, 0.5, 0.75, 0.95])])
    print('% Calibrated [5, 25, 50, 75, 95]: '+str(dSLE[dSLEinds][quan]*50.))
    print('SLE year 7 for MAX(L(theta)): '+str(dSLE7[prob_RMSE==np.max(prob_RMSE)]))
    print('SLE year 50 for MAX(L(theta)): '+str(dSLE[prob_RMSE==np.max(prob_RMSE)]*50))
            


if 0: #calculating dSLE for target PC (see how far from 0.33mmSLE/a)
    dhdt_tmp=ur.dot(target_PC)+dhdt_mean[mask_tmp]
    VAF=vaft0bm2[mask_tmp]+dhdt_tmp/10.*period 
    dVAF=np.ma.sum(np.ma.array(VAF, mask=VAF<0))-\
        np.ma.sum(np.ma.array(vaft0bm2[mask_tmp], mask=vaft0bm2[mask_tmp]<0))
    print(-dVAF*0.917/361.8/period)
    #out: 0.311



if 0:#Leave one out validation, 
    #in a first step we predict the left out member, for each member and save them, later its compared to the real
    #which period to validate?
    if 1:#calibration period, takes about 2h
        LOO_v_vec, LOO_var_v_vec = np.zeros([np.shape(dhdt)[1], k]), np.zeros([np.shape(dhdt)[1], k])
        for out in range(nruns):#this is the leave one out loop
            print(out)
            
            PC_emus_cut=[]
            Vs_cut=np.hstack([Vs[:,:out], Vs[:,out+1:]])#reduced input
            vr_cut=np.hstack([vr[:,:out], vr[:,out+1:]])#and output
            for i in range(k):#train just as before                                               
                PC_emus_cut.append(spt.setup_emulator(Vs_cut, vr_cut[i,:], s='none'))
            LOOv_tmp, LOOvarv_tmp = spt.predictXk(PC_emus_cut, Vs[:,out])#predict
            LOO_v_vec[out,:], LOO_var_v_vec[out,:] =LOOv_tmp.flatten(), LOOvarv_tmp.flatten()
        np.save(datadir+'LOOv_LOOvarv07.npy', [LOO_v_vec,LOO_var_v_vec])#and save
        
    else:#projection period
        LOO_v_vec, LOO_var_v_vec = np.zeros(np.shape(Vs50)[1]), np.zeros(np.shape(Vs50)[1])
        for out in range(np.shape(Vs50)[1]):#this is the leave one out loop
            print(out)

            PC_emus_cut=[]
            Vs_cut=np.hstack([Vs50[:,:out], Vs50[:,out+1:]])#reduced input
            vr_cut=np.hstack([dSLE_BI[:out], dSLE_BI[out+1:]])#and output
            PC_emus_cut=spt.setup_emulator(Vs_cut, vr_cut, s='none') 
            dSLE_tmp = PC_emus_cut[0].predict(Vs50[:,out].reshape([1,-1])) 
            LOO_v_vec[out], LOO_var_v_vec[out] = dSLE_tmp[0][:,0], dSLE_tmp[1][:,0]

        #np.save(datadir+'LOOv_LOOvarv50.npy', [LOO_v_vec,LOO_var_v_vec])
            
elif 1:#now look at the results
    if 1:#calibration period
        LOO_v_vec,LOO_var_v_vec = np.load(datadir+'LOOv_LOOvarv07.npy', allow_pickle=True)
        simu=vr.T
        if 1:#no linear sliding
            simu=simu[Vs[3,:]==0]
            LOO_v_vec=LOO_v_vec[Vs[3,:]==0]
            LOO_var_v_vec=LOO_var_v_vec[Vs[3,:]==0]
        
    else:#projection period
        LOO_v_vec,LOO_var_v_vec = np.load(datadir+'LOOv_LOOvarv50.npy', allow_pickle=True)
        simu=dSLE_BI
        if 1:#no linear sliding
            simu=simu[Vs50[3,:]==0]
            LOO_v_vec=LOO_v_vec[Vs50[3,:]==0]
            LOO_var_v_vec=LOO_var_v_vec[Vs50[3,:]==0]
    
    #some plotting
    fig, ax=plt.subplots()
    ax.errorbar((simu).flatten(), LOO_v_vec.flatten(), yerr=3.*np.sqrt(LOO_var_v_vec.flatten()), fmt='none', c='gray', zorder=1)
    ax.scatter((simu).flatten(), LOO_v_vec.flatten(), c='k', zorder=2, s=6)    
    ax.plot([-500, 500], [-500, 500], c='k')
    ax.set_xlim([-0.25, 0.5])
    ax.set_ylim([-0.25, 0.5])
    ax.set_xlabel('Ice Sheet Model [mmSLE/year]', fontsize=14)
    ax.set_ylabel('Emulator [mmSLE/year]', fontsize=14)
    slope, intercept, r, p, stderr = scipy.stats.linregress((simu).flatten(), LOO_v_vec.flatten())
    ax.text(0.15, -02.35, 'R2 = {0:.3f}'.format(r**2), fontsize=12)
    #fig.savefig(home+"/Dropbox/mypaper/figures/LOO_PCcon_50_v2.png", dpi=300)
    if 1:#and numbers
        LOO_mu=LOO_v_vec.flatten()
        LOO_std=np.sqrt(LOO_var_v_vec.flatten())
        data=(simu).flatten()
    
        RMSE=np.sqrt(np.mean(np.square(LOO_mu-data)))
        print(" ")
        print("RMSE(predicted-simulated):       "+str(RMSE))
        RMSE_range=RMSE/(np.max(data)-np.min(data))
        print("RMSE(predicted-simulated)/range: " + str(RMSE_range))
        sED=np.sqrt(np.sum(np.square(LOO_mu-data)/LOO_std**2))
        print("Standardised Euclidian Distance: " + str(sED))
        slope, interc, r, p, stderr = scipy.stats.linregress(data, LOO_mu)
        print("Pearson's r:                     " + str(r))
        sp_rho, sp_p = scipy.stats.spearmanr(data, LOO_mu)
        print("Spearman's rho:                  " + str(sp_rho))
        ken_tau, ken_p = scipy.stats.kendalltau(data, LOO_mu)
        print("Kendall tau:                     " + str(ken_tau))
        fin90 = np.sum(np.abs(data-LOO_mu)<1.96*LOO_std, dtype=float)/float(len(LOO_mu))
        print("Fraction in 95% range:           " + str(fin90))
        
    if 0:#does the performance change as function of input values? (no)
        fig, axes = plt.subplots(5, figsize=(5,15))
        for i in range(5):
            ax=axes[i]
            ax.scatter((np.ones([4,284])*Vs[i,:]).flatten(), (data-LOO_mu)/LOO_std, c='k', s=5)
            ax.plot([-1,2], [3,3], ls='--', c='k')
            ax.plot([-1,2], [-3,-3], ls='--', c='k')
            ax.plot([-1,2], [0,0], c='k')
            ax.set_xlim([-0.01,1.01])
            #fig.savefig(home+"/Dropbox/mypaper/figures/LOO_normerr07.png", dpi=400)
            







