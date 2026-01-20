#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper figures of prior, posterior, true emission maps

Created on Wed Feb 12 10:14:39 2025

@author: nauss
"""


from datetime import timedelta, datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from base_paths import path_figs
from base_paths import path_base
path_figs = '%s/osse_longterm/'%path_figs
from postprocessing import Postprocessing
import functions_stijn as st
from inversion_main import Inversion
import inversion as inv
from postprocessing_truth import Postprocess_truth
from postpostprocess_multi import Postpostprocess_multi
import time
import read_obs as ro

import cartopy.crs as ccrs
import seaborn as sns
from cartopy.io import shapereader
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.gridspec import GridSpec
    

path = '/%s/observations/aerial/'%path_base
coastline_filename = '%s/nz-coastlines-and-islands-polygons-topo-1250k.shp'%path
shp = shapereader.Reader(coastline_filename)

#%%

t0 = time.time()

inversion_names = 'baseAKLNWP_base',#'baseAKLNWP_mahk_bias',#'baseAKLNWP_odiac'
domain_inv = 'Mah0p3_in'
startdate = datetime(2022,1,1)
enddate = datetime(2022,12,30)
dt_inversion = timedelta(days=28)
dt_spinup    = timedelta(days=7)
dt_spindown  = timedelta(days=7)

postpr_m = {}
for invname in inversion_names:
    print(invname)
    postpr_m[invname] = Postpostprocess_multi(invname, startdate, enddate, dt_inversion, dt_spinup, dt_spindown)
    postpr_m[invname].read_all_postprocessed_data()
    


#%%


pp = postpr_m['baseAKLNWP_base']
lon_inv, lat_inv = pp.inversion_example.get_lonlat_invgrid(domain_inv)
lonb_inv, latb_inv = pp.inversion_example.get_lonlat_invgrid(domain_inv, bounds=True)

true_label = postpr_m[invname].rc.osse_truth_label

startdate = postpr_m[invname].dates_all['daily'][0] + timedelta(seconds=25*3600)
enddate   = postpr_m[invname].dates_all['daily'][-1]

inventories = ['UrbanVPRM','MahuikaAuckland']
emimap_true = {inventory:np.zeros((pp.ninv, 12, 12)) for inventory in inventories}
emimap_pri_from_true = {inventory:np.zeros((pp.ninv, 12, 12)) for inventory in inventories}
etot_true = {inventory:np.zeros((pp.ninv, 2)) for inventory in inventories}
etot_pri  = {inventory:np.zeros((pp.ninv, 2)) for inventory in inventories}
scalars_true = {inventory:np.zeros((pp.ninv, 12, 12)) for inventory in inventories}
for i in range(pp.ninv):
    start = startdate + i*dt_inversion
    end   = startdate + (i+1)*dt_inversion - timedelta(days=1)
    
    pp_true = Postprocess_truth(true_label, startdate=start, enddate=end)
    pp_true.run_standard_postprocessing()
    
    for inventory in inventories:
        scale = postpr_m[invname].inversion_example.get_scaling_factor_prior(inventory)
        
        emimap_true_i = pp_true.emis_agg['per_gridcell']['one_timestep'][inventory][:,:144].reshape(12,12)
        emimap_pri_i  = pp_true.emis_prior['per_gridcell']['one_timestep'][inventory][:,:144].reshape(12,12)*scale
        
        emimap_true_i = np.swapaxes(emimap_true_i, -1,-2)
        emimap_pri_i  = np.swapaxes(emimap_pri_i, -1,-2)
        
        scalars_true[inventory][i] = emimap_true_i / emimap_pri_i
        scalars_true[inventory][i][emimap_pri_i==0] = np.nan
        
        emimap_true[inventory][i]          = emimap_true_i 
        emimap_pri_from_true[inventory][i] = emimap_pri_i
        etot_true[inventory][i] = pp_true.emis_agg['per_domain']['one_timestep'][inventory]/1e6
        etot_pri[inventory][i]  = pp_true.emis_prior['per_domain']['one_timestep'][inventory]/1e6*scale


lon_true, lat_true = pp_true.inversion.get_lonlat_invgrid(domain_inv)
print('%2.2f s'%(time.time()-t0))

#%%

fig,ax = plt.subplots(1,3,figsize=(30,8))
cp = ax[0].pcolormesh(emimap_true['UrbanVPRM'].mean(axis=0))
plt.colorbar(cp,ax=ax[0])

cp = ax[1].pcolormesh(emimap_pri_from_true['UrbanVPRM'].mean(axis=0))
plt.colorbar(cp,ax=ax[1])

cp = ax[2].pcolormesh(scalars_true[inventory][0]*emimap_true['UrbanVPRM'][0] - emimap_pri_from_true['UrbanVPRM'][0])
plt.colorbar(cp,ax=ax[2])

#%%

from read_priors import read_preprocessed_1inventory

# Read high-res fluxes per 4-week interval (= per inversion)
inventories = ['UrbanVPRM','MahuikaAuckland']
domainNAME = pp.rc.domains_NAME[domain_inv]

lons_NAME_full, lats_NAME_full = inv.getLonLatNAME(domainNAME, bounds=False)

mask_lon = ((lons_NAME_full>lonb_inv.min()) & (lons_NAME_full<lonb_inv.max()))
mask_lat = (lats_NAME_full>latb_inv.min()) & (lats_NAME_full<latb_inv.max())
mask_dom = np.outer(mask_lat,mask_lon)

lons_NAME = lons_NAME_full[mask_lon]
lats_NAME = lats_NAME_full[mask_lat]

emis_prior = {}
emis_prior_full = {}
for inventory in inventories:
    scale = postpr_m[invname].inversion_example.get_scaling_factor_prior(inventory)
    emis_prior_full[inventory] = np.zeros((pp.ninv, len(lats_NAME_full), len(lons_NAME_full)))
    emis_prior[inventory] = np.zeros((pp.ninv, len(lats_NAME), len(lons_NAME)))
    
    for i in range(pp.ninv):
        days_to_read = [startdate + i*dt_inversion + timedelta(days=iday) for iday in range(dt_inversion.days)]
        emii = read_preprocessed_1inventory(inventory, days_to_read, domainNAME, cats='all') # kg/m2/s
        emii = np.sum(emii*3600, axis=(0,1)) # kg/m2/period
        emis_prior[inventory][i] = emii[mask_dom].reshape(len(lats_NAME), len(lons_NAME))*scale
        emis_prior_full[inventory][i] = emii*scale

#%%

# Reading fluxes on inversion grid and calculating scalars

scalars_pos = {}

sns.set_context('talk')

nx, ny = pp.rc.nx_inv[domain_inv], pp.rc.ny_inv[domain_inv]
ngrid = nx*ny

epri_map_lr, epos_map_lr = {}, {}
for inventory in inventories:
    scale = pp.inversion_example.get_scaling_factor_prior(inventory)
    epri = pp.emis_all['prior']['per_gridcell']['one_timestep'][inventory]
    epos = pp.emis_all['posterior']['per_gridcell']['one_timestep'][inventory]
    epri = epri[:,:ngrid].reshape(-1, nx,ny)
    epos = epos[:,:ngrid].reshape(-1, nx,ny)
    
    # Easier to have lat,lon, even though in inversion I use lon,lat
    epri = np.swapaxes(epri, -1,-2)
    epos = np.swapaxes(epos, -1,-2)
    
    scalars_pos[inventory] = epos/epri
    
    scalars_pos[inventory][epri==0] = np.nan
    
    epri_map_lr[inventory] = epri
    epos_map_lr[inventory] = epos

#%%

# Regrid scaling factors to the highresolution emission grid

mask_lon = ((lons_NAME_full>lonb_inv.min()) & (lons_NAME_full<lonb_inv.max()))
mask_lat = (lats_NAME_full>latb_inv.min()) & (lats_NAME_full<latb_inv.max())
mask_dom = np.outer(mask_lon,mask_lat)

lons_NAME = lons_NAME_full[mask_lon]
lats_NAME = lats_NAME_full[mask_lat]

regr_inv  = st.make_xesmf_regridder(lon_inv, lat_inv, lons_NAME, lats_NAME, method='nearest_s2d')
regr_true = st.make_xesmf_regridder(lon_true, lat_true, lons_NAME, lats_NAME, method='nearest_s2d')

scalars_pos_regr = {}
scalars_true_regr = {}
for inventory in inventories:
    scalars_pos_regr[inventory]  = regr_inv(scalars_pos[inventory])
    scalars_true_regr[inventory] = regr_true(scalars_true[inventory])

#%%

colors= sns.color_palette('Set2')

fig,ax = plt.subplots(2,1,figsize=(12,12))

# Compare emission totals - double-checking that regridding conserves emission totals
for i,inventory in enumerate(inventories):
    ax[i].set_title(inventory)
    
    area_pgridcell = st.calc_area_per_gridcell(lats_NAME, lons_NAME)

    epri_tot_hr = np.nansum(emis_prior[inventory]*area_pgridcell, axis=(-1,-2)) / 1e6 # kton / timestep
    epri_tot_lr = np.nansum(epri_map_lr[inventory], axis=(-1,-2)) / 1e6 # kton / timestep
    
    epos_tot_hr = np.nansum(scalars_pos_regr[inventory]*emis_prior[inventory]*area_pgridcell, axis=(-1,-2)) / 1e6 # kton / timestep
    epos_tot_lr = np.nansum(epos_map_lr[inventory], axis=(-1,-2)) / 1e6 # kton / timestep
    
    etru_tot_hr = np.nansum(scalars_true_regr[inventory]*emis_prior[inventory]*area_pgridcell, axis=(-1,-2)) / 1e6 # kton / timestep
    etru_tot_lr = np.nansum(emimap_true[inventory], axis=(-1,-2)) / 1e6 # kton / timestep
    
    ax[i].plot(epri_tot_hr, color=colors[0], linestyle='-', label='pri, hr')
    ax[i].plot(epri_tot_lr, color=colors[0], linestyle='--', label='pri, lr')
    ax[i].plot(epos_tot_hr, color=colors[1], linestyle='-', label='pos, hr')
    ax[i].plot(epos_tot_lr, color=colors[1], linestyle='--', label='pos, lr')
    ax[i].plot(etru_tot_hr, color=colors[2], linestyle='-', label='tru, hr')
    ax[i].plot(etru_tot_lr, color=colors[2], linestyle='--', label='tru, lr')
    
    ax[i].legend(loc='best',ncol=2)

#%%

regr_inv = st.make_xesmf_regridder(lons_NAME, lats_NAME, lon_inv, lat_inv, method='conservative')
regr_true = st.make_xesmf_regridder(lons_NAME, lats_NAME, lon_true, lat_true, method='conservative')

# Regrid highres to lowres to see what it looks like
epri_regr_to_inv  = {}
epri_regr_to_true = {}
for i,inventory in enumerate(inventories):
    epri_regr_to_inv[inventory]  = regr_inv(emis_prior[inventory])
    epri_regr_to_true[inventory] = regr_true(emis_prior[inventory])

#%%

# Plot true emissions highres and lowres

fig,ax = plt.subplots(2,2,figsize=(24,12))

for i,inventory in enumerate(inventories):
    
    ratio = epos_map_lr[inventory].sum(axis=0).T / epri_map_lr[inventory].sum(axis=0).T
    
    ax[i,0].set_title('%s, lowres'%inventory)
    # cp = ax[i,0].pcolormesh(emimap_true[inventory].sum(axis=0))
    # cp = ax[i,0].pcolormesh(emimap_pri_from_true[inventory].sum(axis=0))
    # cp = ax[i,0].pcolormesh(epos_map_lr[inventory].sum(axis=0).T)
    cp = ax[i,0].pcolormesh(ratio)
    # cp = ax[i,0].pcolormesh(epri_regr_to_inv[inventory].sum(axis=0))
    cb = plt.colorbar(cp, ax=ax[i,0])
        
    ax[i,1].set_title('%s, highres'%inventory)
    # cp = ax[i,1].pcolormesh((scalars_true_regr[inventory]*emis_prior[inventory]*area_pgridcell).sum(axis=0))
    # cp = ax[i,1].pcolormesh((scalars_true[inventory]*emimap_pri_from_true[inventory]).mean(axis=0))
    cp = ax[i,1].pcolormesh((emis_prior[inventory]).sum(axis=0))
    plt.colorbar(cp, ax=ax[i,1])

#%%

inventory = 'MahuikaAuckland'
scale_pri = postpr_m[invname].inversion_example.get_scaling_factor_prior(inventory)

def setup_maps_auckland(lonlims, latlims, nx_plot, ny_plot, xticks=[], yticks=[]):
    
    fig = plt.figure(figsize=(nx_plot*4,ny_plot*4))
    
    # Plot Mahuika-Auckland
    ax = np.zeros((nx_plot,ny_plot), dtype=object)
    for i in range(nx_plot):
        for j in range(ny_plot):
            ax[i,j] = fig.add_subplot(ny_plot,nx_plot,ny_plot*i+j+1, projection=ccrs.Mercator())
    
    # Plot NZ
    for axi in ax.flatten():
        axi.set_extent([lonlims[0],lonlims[1],latlims[0],latlims[1]], crs=ccrs.PlateCarree())
        for record, geometry in zip(shp.records(), shp.geometries()):
            axi.add_geometries([geometry], ccrs.Mercator(), facecolor='none',
                              linewidth = 0.5, edgecolor='k',zorder=10)
        
        gl = axi.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='silver', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(xticks)
        gl.ylocator = mticker.FixedLocator(yticks)
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 15, 'color': 'k'}
        gl.ylabel_style = {'size': 15, 'color': 'k'}
    
    return fig, ax
    

lonlims = lons_NAME.min(), lons_NAME.max()
latlims = lats_NAME.min(), lats_NAME.max()

x,y = np.meshgrid(lons_NAME,lats_NAME)
for i,sc_pos in enumerate(scalars_pos_regr[inventory]):
    
    fig, ax = setup_maps_auckland(lonlims, latlims, 2,1)
    
    sc_tru = scalars_true_regr[inventory][i]
    
    diff_pri = emis_prior[inventory][i]*(sc_tru-1)
    bias_pri = np.nanmean(diff_pri)
    rms_pri = np.sqrt(np.nanmean(diff_pri**2))
    ax[0,0].set_title("Bias = %2.2f\nRMS = %2.2f"%(bias_pri, rms_pri), fontsize=24)
    cp = ax[0,0].pcolormesh(x,y, diff_pri, vmin=-2, vmax=2, cmap = 'coolwarm', transform=ccrs.PlateCarree())
    
    diff_pos = emis_prior[inventory][i]*(sc_tru-sc_pos)
    bias_pos = np.nanmean(diff_pos)
    rms_pos = np.sqrt(np.nanmean(diff_pos**2))
    ax[1,0].set_title("Bias = %2.2f\nRMS = %2.2f"%(bias_pos, rms_pos), fontsize=24)
    cp = ax[1,0].pcolormesh(x,y, diff_pos, vmin=-2, vmax=2, cmap = 'coolwarm', transform=ccrs.PlateCarree())
    
    # plt.colorbar(cp,ax=ax[i,1], label="kg/m2/4 weeks", extend='both')
    
plt.tight_layout()

#%%

inventory = 'MahuikaAuckland'

fig, ax = setup_maps_auckland(lonlims, latlims, 2,1, xticks=[174.6,174.9], yticks=[-37.0,-36.8])
vmin,vmax = -50,50

x,y = np.meshgrid(lons_NAME,lats_NAME)

epri = emis_prior[inventory]
etru = scalars_true_regr[inventory]*emis_prior[inventory]
epos = scalars_pos_regr[inventory]*emis_prior[inventory]

nday = len(epri)*7
epri = np.sum(epri,axis=0) / nday
epos = np.sum(epos,axis=0) / nday
etru = np.sum(etru,axis=0) / nday

diff_pri = (epri-etru)*1000
bias_pri = np.nanmean(diff_pri)
rms_pri = np.sqrt(np.nanmean(diff_pri**2))
ax[0,0].set_title("Bias = %2.2f\nRMS = %2.2f"%(bias_pri, rms_pri), fontsize=15)
cp = ax[0,0].pcolormesh(x,y, diff_pri, vmin=vmin, vmax=vmax, cmap = 'coolwarm', transform=ccrs.PlateCarree())

diff_pos = (epos-etru)*1000
bias_pos = np.nanmean(diff_pos)
rms_pos = np.sqrt(np.nanmean(diff_pos**2))
ax[1,0].set_title("Bias = %2.2f\nRMS = %2.2f"%(bias_pos, rms_pos), fontsize=15)
cp = ax[1,0].pcolormesh(x,y, diff_pos, vmin=vmin, vmax=vmax, cmap = 'coolwarm', transform=ccrs.PlateCarree())

# Plot measurement sites
for a in ax.flatten():
    for site,(x,y) in st.getSiteCoords().items():
        a.scatter(x,y, s=120, edgecolor='k', facecolor='none', linewidth=1.5, alpha=0.1,transform=ccrs.PlateCarree())
    

# Plot inversion gridlines
# for a in ax.flatten():
#     for ix,lon in enumerate(lonb_inv):
#         a.plot([lon,lon],latlims, 'k-',alpha=0.3, transform=ccrs.PlateCarree())
        
#     for iy,lat in enumerate(latb_inv):
#         a.plot(lonlims,[lat,lat], 'k-',alpha=0.3, transform=ccrs.PlateCarree())
    

# plt.colorbar(cp,ax=ax[1,0], label="mg/m$^2$/day", extend='both')
    
plt.tight_layout()

plt.savefig("%s/emimaps_diff_total"%path_figs, dpi=300)


#%%

# RMS map per day? Or 3-hours?



#%%

# Calculate emission totals from spatial distributions and scalars
etot_from_map = {'pri':{}, 'pos':{}, 'tru':{}}
for inventory in inventories:
    area_pgridcell = st.calc_area_per_gridcell(lats_NAME, lons_NAME)
    etot_from_map['pri'][inventory] = np.nansum(emis_prior[inventory]*area_pgridcell, axis=(-1,-2)) / 1e6 # kton / timestep
    etot_from_map['pos'][inventory] = np.nansum((scalars_pos_regr[inventory] * emis_prior[inventory])*area_pgridcell, axis=(-1,-2)) / 1e6 # kton / timestep
    etot_from_map['tru'][inventory] = np.nansum((scalars_true_regr[inventory] * emis_prior[inventory])*area_pgridcell, axis=(-1,-2)) / 1e6

#%%

fig,ax = plt.subplots(2,1,figsize=(15,10))
    

colors=sns.color_palette('Set2')
for i,inventory in enumerate(inventories):
    ax[i].set_title(inventory)
    ax[i].plot(etot_true[inventory][:,0], color=colors[0], linestyle='-', label='True, from totals')
    ax[i].plot(etot_from_map['tru'][inventory], color=colors[0], linestyle='--', label='True, from map')
    
    etot_prii = postpr_m[invname].emis_all['prior']['per_domain']['one_timestep'][inventory][:,0] / 1e6
    ax[i].plot(etot_prii, color=colors[1], linestyle='-', label='Prior, from totals')
    ax[i].plot(etot_from_map['pri'][inventory], color=colors[1], linestyle='--', label='Prior, from map')

    etot_posi = postpr_m[invname].emis_all['posterior']['per_domain']['one_timestep'][inventory][:,0] / 1e6
    ax[i].plot(etot_posi, color=colors[2], linestyle='-', label='Posterior, from totals')
    ax[i].plot(etot_from_map['pos'][inventory], color=colors[2], linestyle='--', label='Posterior, from map')
    
    ax[i].legend(loc='best', ncol=2)


#%%

# "Paper figure", with 3 panels for true, prior, posterior fluxes, 
# and then below 2 panels with difference with the truth
inventory = 'MahuikaAuckland'

def setup_plot_56panel(npanel=5):
    fig = plt.figure(figsize=(25,12))
    
    gs = GridSpec(100, 100)
    
    axes_top = [fig.add_subplot(gs[:44,i*25+i*10:(i+1)*25+i*10], projection=ccrs.Mercator()) for i in range(3)]
    if npanel == 5:
        axes_bot = [fig.add_subplot(gs[56:,i*25+i*10:(i+1)*25+i*10], projection=ccrs.Mercator()) for i in range(1,3)]
    elif npanel == 6:
        axes_bot = [fig.add_subplot(gs[56:,i*25+i*10:(i+1)*25+i*10], projection=ccrs.Mercator()) for i in range(3)]
    axes = axes_top + axes_bot
    
    # Plot NZ
    for axi in axes:
        axi.set_extent([lonlims[0],lonlims[1],latlims[0],latlims[1]], crs=ccrs.PlateCarree())
        for record, geometry in zip(shp.records(), shp.geometries()):
            axi.add_geometries([geometry], ccrs.Mercator(), facecolor='none',
                              linewidth = 0.5, edgecolor='k',zorder=10)
        
        gl = axi.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='silver', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        xticks = [174.7, 174.9]
        yticks = [-37, -36.9, -36.8]
        gl.xlocator = mticker.FixedLocator(xticks)
        gl.ylocator = mticker.FixedLocator(yticks)
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 15, 'color': 'k'}
        gl.ylabel_style = {'size': 15, 'color': 'k'}
        
    # Plot measurement locations
    for axi in axes:
        for site in ['MKH','TKA','AUT','NWO']:
            loni,lati = ro.get_site_coords(site)
            
            axi.scatter(loni,lati,facecolor='none',edgecolor='k', transform=ccrs.PlateCarree(), zorder=11, linewidth=2.0, s=300)
        
        
    return fig, axes_top, axes_bot

def plot_5panel(inventory, cmap_base, vmin_base, vmax_base, cmap_diff, vmin_diff, vmax_diff, months=np.arange(13)):
    fig, axes_top, axes_bot = setup_plot_56panel()
    
    
    epri = np.nansum((emis_prior[inventory][months]), axis=0)
    epos = np.nansum((scalars_pos_regr[inventory][months] * emis_prior[inventory][months]), axis=0)
    etru = np.nansum((scalars_true_regr[inventory][months] * emis_prior[inventory][months]), axis=0)
    
    for i,(emi,title) in enumerate( zip([etru, epri, epos],['True','Prior','Posterior']) ):
        axes_top[i].set_title(title)
        cp = axes_top[i].pcolormesh(lons_NAME, lats_NAME, emi, vmin=vmin_base, vmax=vmax_base, cmap=cmap_base, transform=ccrs.PlateCarree())
        plt.colorbar(cp,ax=axes_top[i])
    
    
    for i,(emi,title) in enumerate(zip([epri, epos],['Prior - True', 'Posterior - True'])):
        bias = np.nanmean(emi-etru)
        rms  = np.sqrt( np.nanmean((emi-etru)**2) )
        # rms2 = np.sqrt( np.nanmean((emi-etru-bias)**2) ) # bias-corrected1
        
        axes_bot[i].set_title('%s\nBias=%2.2f; RMS=%2.2f'%(title, bias,rms))
        cp = axes_bot[i].pcolormesh(lons_NAME, lats_NAME, emi-etru, vmin=vmin_diff, vmax=vmax_diff, cmap=cmap_diff, transform=ccrs.PlateCarree())
        plt.colorbar(cp,ax=axes_bot[i])
        
    return fig, axes_top, axes_bot
    
#%%

cmap_base = 'Reds'
vmin_base = 0
vmax_base = 150
cmap_diff = 'RdBu_r'
vmin_diff = -10,10
vmax_diff = None
fig1, axes_top1, axes_bot1 = plot_5panel('MahuikaAuckland', cmap_base, vmin_base, vmax_base, cmap_diff, vmin_diff, vmax_diff)

cmap_base = 'PiYG_r'
vmin_base = -10
vmax_base = 10
vmin_diff = -10
vmax_diff = 10
fig2, axes_top2, axes_bot2 = plot_5panel('UrbanVPRM', cmap_base, vmin_base, vmax_base, cmap_diff, vmin_diff, vmax_diff)



#%%

# Where does the weird stuff in UrbanVPRM come from?
# a) Is it one specific month?

cmap_base = 'PiYG_r'
vmin_base = -2
vmax_base = 2
vmin_diff = -2
vmax_diff = 2
for month in range(13):
    fig, axes_top, axes_bot = plot_5panel('UrbanVPRM', cmap_base, vmin_base, vmax_base, cmap_diff, vmin_diff, vmax_diff, months=[month])
    axes_top[0].set_title(month)


#%%

# b) Does it disappear when we regrid to coarse inversion resolution?

regr_hr_to_inv = st.make_xesmf_regridder(lons_NAME, lats_NAME, lon_inv, lat_inv)



months = [4]
for inventory in inventories:
    
    if inventory=='MahuikaAuckland':
        vmin,vmax=0,1
        cmap = 'Reds'
    elif inventory=='UrbanVPRM':
        vmin=-0.5
        vmax=+0.5
        cmap='PiYG_r'
    
    fig,axes_top,axes_bot = setup_plot_56panel()
    
    emis_true_hr = (scalars_true_regr[inventory] * emis_prior[inventory])
    emis_pos_hr  = (scalars_pos_regr[inventory] * emis_prior[inventory])
    
    einv_tru = np.nanmean(regr_hr_to_inv(emis_true_hr)[months],axis=0)
    einv_pri = np.nanmean(regr_hr_to_inv(emis_prior[inventory])[months],axis=0)
    einv_pos = np.nanmean(regr_hr_to_inv(emis_pos_hr)[months],axis=0)
    
    for a,e,title in zip(axes_top, [einv_tru, einv_pri, einv_pos],['True','Prior','Posterior']):
        a.set_title(title)
        cp = a.pcolormesh(lon_inv,lat_inv,e, transform=ccrs.PlateCarree(), 
                          vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar(cp,ax=a)

    for a,e,title in zip(axes_bot, [einv_pri, einv_pos], ['Prior-True', 'Posterior-True']):
        bias = np.nanmean(e-einv_tru)*100
        rms  = np.sqrt( np.nanmean((100*(e-einv_tru))**2) )
        # rms2 = np.sqrt( np.nanmean((emi-etru-bias)**2) ) # bias-corrected1
        
        a.set_title('%s\nBias=%2.2f; RMS=%2.2f'%(title, bias,rms))
        cp = a.pcolormesh(lon_inv,lat_inv,e-einv_tru, transform=ccrs.PlateCarree(), 
                          cmap='RdBu_r', vmin=-0.5,vmax=+0.5)
        plt.colorbar(cp,ax=a)


#%%

# Paper figure
# TRUE     (PRIOR-TRUE)    (POSTERIOR-TRUE)

fig, axes_top, axes_bot = setup_plot_56panel(npanel=6)

months = np.arange(12)
fs = 22

unit = 'mg m$^{-2}$ s$^{-1}$'

inventory_labels = {'MahuikaAuckland':'Anthropogenic', 'UrbanVPRM':'Biosphere'}
for inventory in inventories:
    
    if inventory=='MahuikaAuckland':
        ax = axes_top
        vmin,vmax=0,5.0
        cmap = 'Purples'
        cbar_ticks = np.arange(5)
    elif inventory=='UrbanVPRM':
        ax = axes_bot
        vmin=-2.0
        vmax=+2.0
        cmap='PiYG_r'
        cbar_ticks = np.arange(-2,2.1)
        
    emis_true_hr = (scalars_true_regr[inventory] * emis_prior[inventory])
    emis_pos_hr  = (scalars_pos_regr[inventory] * emis_prior[inventory])
    
    einv_tru = np.nanmean(regr_hr_to_inv(emis_true_hr)[months],axis=0) # [kg/m2/4weeks]
    einv_pri = np.nanmean(regr_hr_to_inv(emis_prior[inventory])[months],axis=0)
    einv_pos = np.nanmean(regr_hr_to_inv(emis_pos_hr)[months],axis=0)
    
    # [kg/m2/4weeks] to [mg/m2/s]
    einv_tru *= (4*7*24*3600) / 1e6
    einv_pri *= (4*7*24*3600) / 1e6
    einv_pos *= (4*7*24*3600) / 1e6
    
        
    ax[0].set_title("True fluxes\n%s"%(inventory_labels[inventory]), fontsize=fs)
    cp = ax[0].pcolormesh(lon_inv,lat_inv,einv_tru, transform=ccrs.PlateCarree(), 
                      vmin=vmin, vmax=vmax, cmap=cmap)
    cb = plt.colorbar(cp,ax=ax[0], label='[%s]'%unit)
    cb.set_ticks(cbar_ticks)

    for a,e,title in zip(ax[1:], [einv_pri, einv_pos], ['Prior $-$ True', 'Posterior $-$ True']):
        bias = np.nanmean(e-einv_tru)*100
        rms  = np.sqrt( np.nanmean((100*(e-einv_tru))**2) )
        # rms2 = np.sqrt( np.nanmean((emi-etru-bias)**2) ) # bias-corrected1
        
        a.set_title('%s\n%s'%(title,inventory_labels[inventory]), fontsize=fs)
        props = dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='white', alpha=0.5)
        a.text(0.05, 0.98, 'Bias = %1.1f %s\nRMS = %1.1f %s'%(bias, unit, rms, unit), 
               transform=a.transAxes, va='top', bbox=props, zorder=99)
        cp = a.pcolormesh(lon_inv,lat_inv,e-einv_tru, transform=ccrs.PlateCarree(), 
                          cmap='RdBu_r', vmin=-1.0,vmax=+1.0)
        cb = plt.colorbar(cp,ax=a, label='[%s]'%unit)
        cb.set_ticks([-1,0,1])

fig.tight_layout()  

# Panel labels bottom right
bbox=dict(boxstyle='square', fc="w", ec="k", alpha=0.9)
for a,panel_label in zip(axes_top+axes_bot, ['A','B','C','D','E','F']):
    a.text(0.972, 0.025, panel_label, va='bottom', ha='right', fontsize=24, zorder=12, transform=a.transAxes, bbox=bbox)
    


plt.savefig('%s/paperfig_emimaps_true_diff'%path_figs, dpi=300)


#%%

import inversion as invh

# Spatial distribution prior and posterior errors and error reductions
fig, axes_top, axes_bot = setup_plot_56panel(npanel=6)
for inventory in inventories:
    if inventory=='MahuikaAuckland':
        ax = axes_top
        vmax = 1.5
        ticks = [0,0.5,1.0,1.5]
    elif inventory=='UrbanVPRM':
        ax = axes_bot
        vmax = 1.5
        ticks = [0,0.5,1.0,1.5]
        
    unc = {}
    for i,label in enumerate(['prior','posterior']):
            unci = np.sqrt(pp.unc_all_abs[label]['per_gridcell']['one_timestep'][inventory][inventory][:,:81])
            
            # Add up timesteps
            tsteps = pp.dates_all['one_timestep']
            corlen = pp.rc.prior_errors[inventory]['L_temp_long']
            nx = unci.shape[1]
            unci_tot = np.zeros(nx)
            for ix in range(nx):
                unci_tot[ix] = invh.add_up_correlated_uncertainties(tsteps, unci[:,ix], timedelta(days=corlen))
                
            # Reshape to lat,lon
            unci = unci_tot.reshape(9,9).T 
    
            # [kg/year] -> [mg/m2/s]
            areas = st.calc_area_per_gridcell(lat_inv, lon_inv)
            ntime = 13*4*7*3600
            unci = unci / (areas*ntime) * 1e6
            unc[label] = unci
    
    cmap = 'Purples'
    cp0 = ax[0].pcolormesh(lon_inv, lat_inv, unc['prior'], transform=ccrs.PlateCarree(),vmax=vmax,cmap=cmap)
    cp1 = ax[1].pcolormesh(lon_inv, lat_inv, unc['posterior'], transform=ccrs.PlateCarree(),vmax=vmax,cmap=cmap)
    reduction = 100*( unc['prior'] - unc['posterior'] ) / unc['prior']
    reduction[unc['prior']==0] = 0.0
    cp2 = ax[2].pcolormesh(lon_inv, lat_inv, reduction, transform=ccrs.PlateCarree(),vmax=50,cmap=cmap)
    
    ax[0].set_title('Prior errors\n%s'%(inventory_labels[inventory]), fontsize=fs)
    ax[1].set_title('Posterior errors\n%s'%(inventory_labels[inventory]), fontsize=fs)
    ax[2].set_title('Error reduction\n%s'%(inventory_labels[inventory]), fontsize=fs)
    
    cb0 = plt.colorbar(cp0,ax=ax[0], label='[%s]'%unit)
    cb0.set_ticks(ticks)
    cb1 = plt.colorbar(cp1,ax=ax[1], label='[%s]'%unit)
    cb1.set_ticks(ticks)
    cb2 = plt.colorbar(cp2,ax=ax[2], label='[%]')
    


plt.savefig('%s/sfig_map_error_reductions.png'%path_figs)










#%%































