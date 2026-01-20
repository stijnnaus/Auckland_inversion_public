#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Here I generate a "true" timeseries of fluxes and enhancements to be used in the inversion.
The reason that I separate it from the inversion is that doing it naively (perturbing the prior state)
doesn't work for a large number of grid cells, so I need to do something more complicated.

In addition, I want to run an ensemble of inversions that all try to reproduce the same truth
so it makes sense to have the ensemble all read in the same pre-generated truth.
The idea is I generate one truth for a whole year once, and then I can throw whatever
sensitivity test against it by running different inversion set-ups.

I generate two things:
    a) A true state (corresponding to true emissions).
    b) A timeseries of true enhancements for each site. Note that I do this per site (as opposed to per gradient,
        which is the quantity optimized in the inversion), because I might want to optimize different gradients 
        (e.g., -TKA instead of -MKH), and because it seems better to assume that the error for each site are independent, 
        rather than that errors of each gradient are independent. E.g., if all gradients are -MKH it's better to generate
        MKH synthetic observations separately and combine them with TKA,NWO,AUT, thus accounting for the overlap in gradients.
        Note that transport model errors of e.g., AUT and NWO might well be correlated since they're so close together,
        but that's not something I account for anyway.

I create the truth by generating:
    a) Scaling factors matching spatio-temporal resolution prescribed in a prior, 
        randomly perturbed based on an error covariance matrix similarly prescribed.
    b) Generate synthetic observations (H @ x)
    c) Randomly perturb synthetic observations based on R
                                                                                      
The timeseries are generated based on a prior error covariance matrix and prior prescribed in a .yaml file.

    
Created on Thu Nov 14 13:16:44 2024

@author: nauss
"""

from datetime import timedelta, datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from base_paths import path_figs
path_figs = '%s/generate_truth/'%path_figs
from postprocessing import Postprocessing
import functions_stijn as st
from inversion_main import Inversion
from construct_B import ConstructorB
from construct_x import ConstructorX

np.random.seed(1) # Reproducible results

#%%


def construct_Bspatial(cB, prior_label):
    '''
    Create the spatial correlation part of the error covariance matrix for one
    prior label (e.g., anthropogenic or biosphere).
    '''
    
    
    priors = st.invert_dictionary(cB.rc.prior_error_labels)[prior_label]
    Bs = np.zeros_like(cB.dx_2d)
    for pri1 in priors:
        
        label1 = cB.rc.prior_error_labels[pri1]
        
        # Read error settings prior 1
        error_config1  = cB.rc.prior_errors[pri1]
        Lx1            = error_config1['L_spatial']
        err1           = error_config1['rel_error']
        
        # Indices in spatial covariance matrix
        idx1 = cB.get_prior_idx_in_Bspatial(pri1) # idx1b has length (nx)
        
        for pri2 in priors:
            
            label2 = cB.rc.prior_error_labels[pri2]
            idx2   = cB.get_prior_idx_in_Bspatial(pri2)
            
            if label1==label2:
                # Priors are correlated
                
                # Read error settings prior 2
                error_config2 = cB.rc.prior_errors[pri2]
                Lx2           = error_config2['L_spatial']
                err2          = error_config2['rel_error']
                
                dx_i     = cB.dx_2d[idx1,:][:, idx2]
                Bs_i     = cB.calculate_cov_matrix(dx_i, Lx1, Lx2)
                Bs_i    *= err1*err2
                Bs[idx1[:,np.newaxis],idx2] = Bs_i
                
            else:
                # Priors are not correlated
                Bs[idx1[:,np.newaxis],idx2] = 0.
    
    # Select part that corresponds to prior error label
    idx = np.array([], dtype=int)
    Bs_priors = np.array([], dtype=object) # Which prior corresponds to which element
    for prior in priors:
        idx_i = cB.get_prior_idx_in_Bspatial(prior)
        idx = np.append(idx, idx_i)
        Bs_priors = np.append(Bs_priors, [prior]*len(idx_i))
        
    return Bs[idx][:,idx], Bs_priors


#%%

def construct_Btemporal(cB, prior):
    '''
    Create the temporal correlation part of the error covariance matrix for one prior.
    '''
        
    label = cB.rc.prior_error_labels[prior]
    
    # Read error settings
    error_config  = cB.rc.prior_errors[prior]
    Lt_long       = error_config['L_temp_long']
    Lt_diurn      = error_config['L_temp_diurnal']
    
    Bt_diurn = cB.calculate_cov_matrix(cB.dt_diurnal_2d, Lt_diurn, Lt_diurn)
    
    # Bt_long has separate treatment because of option weekend/weekday split
    Bt_long  = cB.calculate_cov_matrix_Bt_long(cB.dt_longterm_2d, Lt_long , Lt_long, label)                        
    
    Bt = np.kron(Bt_long, Bt_diurn)
                
    return Bt

def get_prior_idx_in_Btemporal(cB, prior):
    ipri  = np.where(cB.priors_all==prior)[0][0]
    ntime = len(cB.dt_diurnal_2d)*len(cB.dt_longterm_2d)
    idx = np.arange(ipri*ntime, (ipri+1)*ntime)
    return idx
    

#%%

template_name = 'baseAKLNWP_truth2'

kwargs = {}

inv = Inversion(template_name, kwargs)

#%%

# prep work
t0 = time.time()
cx = ConstructorX(template_name, kwargs)
cx.remove_file_emis_on_invgrid()
cx.construct_x()

t1 = time.time()
cB = ConstructorB(template_name, kwargs)
cB.construct_dx_dt()
t2 = time.time()

print('Construct x takes %2.0f // Construct dx_dt takes %2.0f'%(t1-t0, t2-t1))

#%%

# We generate a truth "per label" (i.e., anthropogenic and biospheric), because we
# can only deconstruct B into temporal and spatial per label (because both labels have
# their own spatial and temporal error structure)

# Which prior error settings to use for the temporal error matrix, per label
prior_labels = [0,1]
prior_to_use_Bt = {0:'MahuikaAuckland', 1:'UrbanVPRM'}

# Force a bias per prior? If not prescribed stick to the random perturbation
bias_per_prior  = {'MahuikaAuckland':1.05, 'EDGARv8':1.05}
prior_per_label = st.invert_dictionary(cB.rc.prior_error_labels)

t0 = time.time()

xpert_dict = {}
Bs_priors = {}
for prior_label in prior_labels:
    
    # Spatial part
    Bs, Bs_priors[prior_label] = construct_Bspatial(cB, prior_label)
    Bs_chol =  np.linalg.cholesky(Bs)
    nx = len(Bs)
    
    # Temporal part
    # In the inversion I can have different temporal correlations per prior, but
    # because I do my analysis here per label, only Bs OR Bt can be per-prior
    # (and Bs has to be per-prior, because different priors cover different domains)
    # (and I have to do analysis per-label because the label is internally correlated)  
    Bt  = construct_Btemporal(cB, prior_to_use_Bt[prior_label])
    Bt_chol = np.linalg.cholesky(Bt)
    nt = len(Bt)
    
    rand_vec = np.random.randn(nt, nx)
    
    xpert_dict[prior_label] = Bt_chol @ rand_vec @ Bs_chol.T

# Combine the different grid cells / priors in the same order as they are in the inversion,
# which is (prior, time, space)
# Note that domain order should be automatically conserved since I use inversion indexing in constructing Bs
xpert = np.zeros((0),dtype=float)
priornames = []
for prior in cB.priors_all: # In inversion this is the ordering
    label = cB.rc.prior_error_labels[prior]
    idx_pri = np.where(Bs_priors[label]==prior)[0]
    
    xpert_i = xpert_dict[label][:,idx_pri].flatten()
    
    xpert = np.append(xpert, xpert_i)
    priornames += [prior]*len(xpert_dict[label][:,idx_pri].flatten())
    
# Finally, we have to scale the perturbation in the same way that we scale the prior error matrix
# These scalings for example cover where I impose a minimum absolute uncertainty, or an uncertainty
# based on GPP/Resp instead of NEE
cB.get_emissions_on_inversion_grid()
scalars = cB.get_Bmatrix_error_scalings([])
xpert_sc = np.zeros((0), dtype=float)
for prior in cB.priors_all:
    idx_pri = np.where(cx.xpriornames==prior)[0]
    xpert_sc_i = xpert[idx_pri]*np.sqrt(scalars[prior]) # Scaling analogous to how we scale B
    xpert_sc = np.append(xpert_sc, xpert_sc_i)
    
    

# Default is prior is 1, so the perturbation has to be applied around 1
xtrue = 1.0 + xpert_sc

# Force anthropogenic fluxes to be positive
for prior in cB.priors_all: 
    label = cB.rc.prior_error_labels[prior]
    if label==0:
        mask_1 = cx.xpriornames==prior
        mask_2 = xtrue<0
        mask = mask_1 & mask_2
        
        sum1 = (xtrue*cB.emis_inv_vec)[mask_1].sum()
        xtrue[mask] = 0
        sum2 = (xtrue*cB.emis_inv_vec)[mask_1].sum()
        
        # Conserve total emissions
        xtrue[mask_1] *= (sum1/sum2)
        
# Optionally force biases in truth relative to prior
for prior,bias in bias_per_prior.items():
    mask_1 = cx.xpriornames==prior
    sum_true  = (xtrue*cB.emis_inv_vec)[mask_1].sum()
    sum_prior = (cB.emis_inv_vec)[mask_1].sum()
    scale     = (bias*sum_prior)/sum_true
    xtrue[mask_1] *= scale
    
    print( '%s %2.2f'%(prior, (np.sum(xtrue[mask_1]*cB.emis_inv_vec[mask_1]) / np.sum(cB.emis_inv_vec[mask_1]))) )
    

print("Generate truth takes : %2.2f"%(time.time()-t0))
    
#%%

# So now I need to save it in a format that I can use 

#  a) In the inversion -> I only really need the true enhancements ( = H @ x )

#  b) In postprocessing. In principle, since I have the truth .yaml, I only
#      need xtrue, and can always recreate all emission information, using 
#      existing postprocessing scripts.

# So just save enhancements + true state, and then I can always re-run this script
# if I want more info


# 1) Calculate true enhancements. I do it stepwise because otherwise H can get very
#     big for a whole year, and in principle all days are independent.


def select_timeperiod_in_state(cx_all, cx_sel, x, start, end):
    tsteps_all = cx_all.get_timesteps_opt_full()
    tsteps_sel = cx_sel.get_timesteps_opt_full()
    mask = [t in tsteps_sel for t in tsteps_all]
    
    xsel = np.zeros(0, dtype=float)
    for prior in cx_all.priors_all: 
        
        # Select part of the state for this prior
        idx_pri = np.where(cx_all.xpriornames==prior)[0]
        xsel_i = x[idx_pri]
        
        # Select relevant timesteps
        xsel_i = xsel_i.reshape(len(tsteps_all), -1) # Reshape so time is its own dimension
        xsel = np.append(xsel, xsel_i[mask].flatten())
        
    return xsel
    

from construct_y_synthetic import ConstructorY_synthetic
from construct_H import ConstructorH
from construct_R import ConstructorR

dt_step = timedelta(days=10) # how big are the chunks we create the truth in

cy = ConstructorY_synthetic(template_name, kwargs, barebones=True)
cy.construct_y()

startdate_all = cy.rc.date_start
enddate_all   = cy.rc.date_end

ytrue_all           = np.zeros_like(cy.yvector) # Container for the full truth
ytrue_all[:]        = np.nan                  # So I can check after if everything's been filled
ytrue_all_wnoise    = np.zeros_like(cy.yvector)
ytrue_all_wnoise[:] = np.nan
yori_all            = np.zeros_like(cy.yvector)
ynot_scaled         = np.zeros_like(cy.yvector)
ydates_all = cy.ydates
ysites_all = cy.ysites
usites = np.unique(ysites_all)

t0 = time.time()
# Loop over chunks we construct truth in
startdate_i = startdate_all
while startdate_i <= enddate_all:
    print(startdate_i)
    
    enddate_i = startdate_i + dt_step
    if enddate_i>enddate_all:
        # Last window might be shorter than dt_step
        enddate_i = enddate_all
    
    kwargs_i = {'date_start':['direct',startdate_i], 'date_end':['direct',enddate_i]}
    
    cx_i = ConstructorX(template_name, kwargs_i)
    cx_i.construct_x()
    
    cy_i = ConstructorY_synthetic(template_name, kwargs_i, barebones=True)
    cy_i.construct_y()
    
    cH_i = ConstructorH(template_name, kwargs_i)
    cH_i.construct_H()
    
    ydates_i = cy_i.ydates
    ysites_i = cy_i.ysites
    
    xtrue_sel = select_timeperiod_in_state(cx, cx_i, xtrue, startdate_i, enddate_i)
    xtrue_nosc_sel = select_timeperiod_in_state(cx, cx_i, xpert+1, startdate_i, enddate_i)
    ytrue_i  = (cH_i.Hmatrix @ xtrue_sel.flatten())
    ytrue_i_nosc = (cH_i.Hmatrix @ xtrue_nosc_sel.flatten())
    
    yori_i = (cH_i.Hmatrix @ np.ones_like(xtrue_sel.flatten()))
    
    # Add observational noise - I need H for R so it all needs to be in the same loop
    cR_i = ConstructorR(template_name, kwargs_i)
    cR_i.construct_R()
    
    ytrue_i_wnoise = np.random.multivariate_normal(ytrue_i, cR_i.Rmatrix, 1)[0]
    
    # Insert partial ytrue_i into whole ytrue
    for site in usites:
        mask_sel_site = (ysites_i==site)
        mask_sel_date = np.array([d in ydates_all for d in ydates_i])
        mask_sel      = mask_sel_site & mask_sel_date
        
        mask_all_site = (ysites_all==site)
        mask_all_date = (ydates_all >= startdate_i) & (ydates_all <= enddate_i)
        mask_all      = mask_all_site & mask_all_date
        
        ytrue_all_wnoise[mask_all] = ytrue_i_wnoise[mask_sel]
        ytrue_all[mask_all]        = ytrue_i[mask_sel]
        yori_all[mask_all]         = yori_i[mask_sel]
        ynot_scaled[mask_all]      = ytrue_i_nosc[mask_sel]
        
    
    startdate_i += dt_step
    
    
print("No of nans %i"%(np.isnan(ytrue_all_wnoise).sum()))
print("Calc ytrue  takes %2.2f"%(time.time()-t0))

# Finally, we add a somewhat arbitrary threshold to filter out extreme values,
# which I think sometimes pop up because of interaction between an extreme 
# perturbation, high footprint sensitivity and either large GPP or Respiration
# I do it based on 3-sigma now because it sounds better

yfinal = np.copy(ytrue_all_wnoise)
dates_final = np.copy(cy.ydates)
sites_final = np.copy(cy.ysites)
mask_threshold = np.array([False],dtype=bool)

i = 0
maxiter = 1

while np.sum(~mask_threshold)>0 and i<maxiter:
    threshold = 3*np.std(yfinal)
    low  = np.median(yfinal) - threshold
    high = np.median(yfinal) + threshold
    mask_threshold = (yfinal>low) & (yfinal<high)
    yfinal      = yfinal[mask_threshold]
    dates_final = dates_final[mask_threshold]
    sites_final = sites_final[mask_threshold]
    
    i+=1
    
    print("Treshold filters only %2.2f%% of data"%((~mask_threshold).sum()/len(mask_threshold)*100))

#%%

# Save state
# Note that priornames and domainnames are already saved when we set up the prior

path_out = cx.get_path_inversion_input()
np.save('%s/xtrue.npy'%path_out, xtrue.flatten())

#%%

# Save enhancements

from netCDF4 import Dataset
import os

path_out = cx.get_path_inversion_input()

fname_out = '%s/true_obs.nc4'%path_out
print(fname_out)
if os.path.isfile(fname_out):
    os.remove(fname_out)

with Dataset(fname_out, 'w') as d:
    d.createDimension('date_len', 4)
    for site in np.unique(cy.ysites):
        dsite = d.createGroup(site)
        
        mask = (sites_final==site)
        nobs = mask.sum()
        
        dsite.createDimension("nobs_%s"%site)
        vdate = dsite.createVariable('dates', 'i4', ('nobs_%s'%site, 'date_len'))
        vco2  = dsite.createVariable('co2'  , 'f8', ('nobs_%s'%site, ))
        
        vdate[:] = np.array([[date.year, date.month, date.day, date.hour] for date in dates_final[mask]], dtype=int)
        vco2[:]  = yfinal[mask]

#%%


#%%

# fig,ax = plt.subplots(2,2, figsize=(20,15))

# cp0 = ax[0,0].pcolormesh(Bs, vmax=1.0)
# plt.colorbar(cp0, ax=ax[0,0])

# cp0 = ax[0,1].pcolormesh(Bs_chol, vmax=1.0)
# plt.colorbar(cp0, ax=ax[0,1])

# cp1 = ax[1,0].pcolormesh(Bt,vmax=1.0)
# plt.colorbar(cp1, ax=ax[1,0])

# cp1 = ax[1,1].pcolormesh(Bt_chol,vmax=1.0)
# plt.colorbar(cp1, ax=ax[1,1])

fig,ax = plt.subplots(1,2,figsize=(12,5))

nt = len(cx.get_timesteps_opt_full())
nx,ny = 12,12
ngrid = nx*ny
istep = 40
for i,prior in enumerate(['MahuikaAuckland','UrbanVPRM']):
    ax[i].set_title(prior)
    
    xsel = xtrue[cx.xpriornames==prior].reshape(nt,-1)
    xsel = xsel[istep][:ngrid].reshape(nx,ny)
    
    cp = ax[i].pcolormesh(xsel,)#vmin=-0.0, vmax=+2.0)
    plt.colorbar(cp, ax=ax[i])

fig,ax = plt.subplots(1,1,figsize=(12,5))
for i,prior in enumerate(['MahuikaAuckland','UrbanVPRM']):
    xt = xpert[cx.xpriornames==prior].reshape(nt,-1).mean(axis=-1)
    xt = xt[3::8]
    ax.plot(xt, label=prior)
ax.legend(loc='best')

#%%


def get_wind_mask(dates_i):
    '''
    Function to get the standard windspeed / wind direction mask for a list of dates
    '''
    
    from read_meteo_NAME import RetrievorWinddataNAME
    
    retr = RetrievorWinddataNAME(template_name)
    
    udates = np.sort(np.unique(dates_i))
    meteo = retr.retrieve_winddata_NAME(udates, ['mangere','mkh','skytower'], sitename_version='obs')
    ws, wd = meteo[0], meteo[1]
    
    mask_ws    = (ws[0]>3) & (ws[1]>3)
    mask_wd_p1 = (wd[1]>200) & (wd[1]<320) & (wd[2]>200) & (wd[2]<300)
    mask_wd_p2 = (wd[1]>0) & (wd[1]<90)    & (wd[2]>0)   & (wd[2]<90)
    mask_wd    = mask_wd_p1 | mask_wd_p2
    
    afternoon_hours = np.arange(12,18)
    afternoon = [d.hour in afternoon_hours for d in udates]
    mask_w = (mask_ws & mask_wd) | afternoon
    
    # This is for the unique dates, now we project on the original date list
    # which can have duplicates
    mask_w_full = np.zeros(len(dates_i), dtype=bool)
    for i,date in enumerate(udates):
        mask_w_full[dates_i==date] = mask_w[i]
    
    return mask_w_full

# usites='AUT',
afternoon = np.arange(12,18)
night = np.arange(0,8)
fig,ax = plt.subplots(len(usites),2, figsize=(18,5*len(usites)))
for i,site in enumerate(usites):
    
    
    mask = sites_final==site
    mask1 = mask & np.array([d.hour in afternoon for d in dates_final])
    mask2 = mask & np.array([d.hour in night for d in dates_final])
    
    dates_i = dates_final[mask]
    mask_w = get_wind_mask(dates_i)
    
    ax[i,0].set_title(site)
    ax[i,0].plot(dates_i[mask_w], yfinal[mask][mask_w], label='No sc')
    # ax[i,0].plot(dates_i[mask_w], yori_all[mask][mask_w], label='Prior')
    # ax[i,0].plot(dates_i[mask_w], ytrue_all[mask][mask_w], label='No noise')
    # ax[i,0].plot(dates_i[mask_w], ytrue_all_wnoise[mask][mask_w], label='With noise')
    bins = np.linspace(-15,15,30)
    alpha = 0.5
    # ax[i,0].hist(yori_all[mask1], bins=bins, alpha=alpha, label='day')
    # ax[i,0].hist(yori_all[mask2], bins=bins, alpha=alpha, label='night')
    ax[i,0].legend(loc='best')
    ax[i,0].set_ylabel("CO2 [ppm]")
    
    
    # ax[i,1].set_title(site)
    # ax[i,1].plot(dates_i[mask_w], (ynot_scaled-ytrue_all)[mask][mask_w], label='Diff')
    
#%%



prior = 'UrbanVPRM'
NEE = cB.get_emis_invgrid(prior, 'NEE')
GEE = cB.get_emis_invgrid(prior, 'GEE')
Re  = cB.get_emis_invgrid(prior,  'Re')

# sc2 = cB.get_scalars_for_GPP_Resp_one('UrbanVPRM')
# sc3 = cB.get_Bmatrix_error_scalings([])['UrbanVPRM']

#%%

# plt.plot(np.abs(GEE)*0.25 + np.abs(Re)*0.5)
# plt.plot(sc3*np.abs(NEE))
# # plt.plot(scalars['UrbanVPRM']*np.abs(NEE))

#%%


# scalars1 = cB.get_scalars_for_minimum_uncertainty_per_timestep()
# scalars2 = cB.get_scalars_for_unc_total_per_prior([], scalars1)
# scalars3 = cB.get_scalars_for_GPP_Resp(scalars2)

#%%

# Investigate the highest enhancement
site = 'MKH'
mask1 = (cy.ysites==site)
mask_w = get_wind_mask(cy.ydates[mask1])
keys = np.argsort( np.abs(ytrue_all[mask1][mask_w]) )[::-1]

nt_x = len(cx.get_timesteps_opt_full())
n = 5
for i in range(n):
    iy = keys[i]
    
    site = cy.ysites[mask1][mask_w][iy]
    date = cy.ydates[mask1][mask_w][iy]
    val  = ytrue_all[mask1][mask_w][iy]
    
    print("Highest observation on: ", site,date,val)
    
    # ivprm = np.where(cx.get_timesteps_opt_full()==date)[0][0]
    mask  = cx.xpriornames=='UrbanVPRM'
    
    print_stats_date_site(date,site)





#%%

def print_stats_date_site(date,site):


    day  = datetime(date.year, date.month, date.day)
    startdate_i = day + timedelta(seconds=25*3600) - timedelta(days=2)
    enddate_i   = day + timedelta(days=1)
    
    
    
    kwargs_i = {'date_start':['direct',startdate_i], 'date_end':['direct',enddate_i]}
    
    cx_i = ConstructorX(template_name, kwargs_i)
    cx_i.construct_x()
    
    cy_i = ConstructorY_synthetic(template_name, kwargs_i, barebones=True)
    cy_i.construct_y()
    
    cH_i = ConstructorH(template_name, kwargs_i)
    cH_i.construct_H()
    
    cB_i = ConstructorB(template_name, kwargs_i)
    # NEE_i = cB_i.get_emis_invgrid('UrbanVPRM', 'NEE')
    # GEE_i = cB_i.get_emis_invgrid('UrbanVPRM', 'GEE')
    # Re_i  = cB_i.get_emis_invgrid('UrbanVPRM', 'Re')
    
    ydates_i = cy_i.ydates
    ysites_i = cy_i.ysites
    
    xtrue_sel = select_timeperiod_in_state(cx, cx_i, xtrue, startdate_i, enddate_i)
    xtrue_nosc_sel = select_timeperiod_in_state(cx, cx_i, xpert+1, startdate_i, enddate_i)
    ytrue_i  = (cH_i.Hmatrix @ xtrue_sel.flatten())
    ytrue_i_nosc = (cH_i.Hmatrix @ xtrue_nosc_sel.flatten())
    
    yori_i = (cH_i.Hmatrix @ np.ones_like(xtrue_sel.flatten()))
    
    # Add observational noise - I need H for R so it all needs to be in the same loop
    cR = ConstructorR(template_name, kwargs_i)
    cR.construct_R()
    
    ytrue_i_wnoise = np.random.multivariate_normal(ytrue_i, cR.Rmatrix, 1)[0]
    
    idx_y = np.where( (ydates_i==date) & (ysites_i==site) )[0][0]
    Hx = np.abs(cH_i.Hmatrix[idx_y]*xtrue_sel)
    idx_x = np.where(Hx==Hx.max())[0][0]
    
    _, xpriornames, xpriordomains = cB_i.read_state()
    xpriordates   = np.array([], dtype=object)
    for prior in cB_i.priors_all:
        domains_i = cB_i.get_domains_for_prior(prior)
        tsteps = cB_i.get_timesteps_opt_full()
        nstep = len(tsteps)
        for itime in range(nstep):
            for domain in domains_i:
                nx,ny = cB_i.rc.nx_inv[domain],cB_i.rc.ny_inv[domain]
                xpriordates   = np.append(xpriordates, [tsteps[itime]]*nx*ny)
    
    
    print("This high enhancement comes from:",xpriornames[idx_x], xpriordomains[idx_x], xpriordates[idx_x])
    print()



#%%

# How many obs are above a certain threshold?
thresholds = np.linspace(0,15,100)
above_true       = np.zeros_like(thresholds)
above_prior      = np.zeros_like(thresholds)
above_not_scaled = np.zeros_like(thresholds)
mask_w = get_wind_mask(cy.ydates)
for i,t in enumerate(thresholds):
    above_true[i]       = (np.abs(ytrue_all[mask_w])>t).mean()*100
    above_not_scaled[i] = (np.abs(ynot_scaled[mask_w])>t).mean()*100
    above_prior[i]      = (np.abs(yori_all[mask_w])>t ).mean()*100


sns.set_context('talk')
fig,ax = plt.subplots(1,1, figsize=(10,5))

ax.plot(thresholds, above_true,       label='True enhancements')
ax.plot(thresholds, above_not_scaled, label='True enhancements, not scaled')
ax.plot(thresholds, above_prior,      label='Prior enhancements')
ax.set_ylabel("Percentage higher than")
ax.set_xlabel("Threshold (ppm)")
ax.legend(loc='best')

#%%

# Quick look at wind filter
afternoon = np.arange(12,18)
mask_night = [d.hour not in afternoon for d in cy.ydates]
mask_w_night = mask_w[mask_night]
dates_night = cy.ydates[mask_night]



perc_monthly_filtered = np.zeros(12)
for i,month in enumerate(np.arange(1,13)):
    mask_month = np.array([d.month == month for d in dates_night])
    perc_monthly_filtered[i] = 100*np.mean(mask_w_night[mask_month])

fig,ax = plt.subplots(1,1,figsize=(10,5))
ax.plot([datetime(2022,i,1) for i in range(1,13)], perc_monthly_filtered)
ax.set_ylabel("Percentage of night-time\ndata not filtered")
ax.set_ylim(0,50)
st.adjust_xticks_dates_to_m(ax, month_fmt='%b')



#%%

err = cR_i.rc.obs_errors
ydiff = np.abs(yori_all[mask_threshold]-yfinal)
err_all = 0.5 + 0.3*np.abs(yori_all[mask_threshold]) + 0.3*ydiff

cost_per_obs = (np.abs(yfinal-yori_all[mask_threshold])/err_all)


plt.plot(np.cumsum(np.sort(cost_per_obs)[::-1])/np.sum(cost_per_obs))










