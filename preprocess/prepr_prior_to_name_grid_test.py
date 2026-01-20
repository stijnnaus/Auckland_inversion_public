#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing whether the preprocessing of the priors worked: does the output
look reasonable on a grid, and are emission totals conserved?
"""

import functions_stijn as st
from netCDF4 import Dataset
import numpy as np
import read_priors as rp
import xarray as xar
import matplotlib.pyplot as plt
import seaborn as sns
import inversion as inv
from datetime import datetime,timedelta
import os,glob,sys,time
from base_paths import path_inv_base
sns.set_context('talk')
colors = sns.color_palette('Set2')


def create_lonlat_mask_bounds(lons, lats, lonbounds, latbounds):
    masklon = (lons>=lonbounds[0]) & (lons<=lonbounds[1])
    masklat = (lats>=latbounds[0]) & (lats<=latbounds[1])
    return np.outer(masklon,masklat)

domain = 'Mah0p3'
path_prepr = '%s/priors/preprocessed/'%path_inv_base
lons_name, lats_name = inv.getLonLatNAME(domain)
area_pgrid_proc = st.calc_area_per_gridcell(lats_name, lons_name)
kgs_to_ktyr = 366*24*3600*1e-6 # kg/s to kg/year

# ------- 1. MahuikaAuckland -------
#%%
path_prepr_mah = path_prepr + 'MahuikaAuckland/%s'%domain
cats = rp.get_all_mahuika_cats()

#%%
# Read the original emission files
lons_ori, lats_ori, emi_ori = {}, {}, {}
etot_ori, area_pgrid_ori = {},{}
for cat in cats:
    lons_ori[cat], lats_ori[cat], emi_ori[cat] = rp.read_mahuika_onecat(cat, unit='kg/m2/s')
    
    area_pgrid_ori[cat] = st.calc_area_per_gridcell(lats_ori[cat], lons_ori[cat])
    if   rp.get_freq_from_cat_mah(cat)=='weekday1':
        # Zero on weekends
        emi_ori[cat] = (5/7.)*emi_ori[cat]
    elif rp.get_freq_from_cat_mah(cat)=='weekday2':
        # Explicit data on weekends
        emi_ori[cat] = (2/7.)*emi_ori[cat][:,:,:,0] + (5/7.)*emi_ori[cat][:,:,:,1]
    
    
#%%
# Read preprocessed emissions
emi_proc = {}
dates = np.array([datetime(2016,1,1) + timedelta(days=i) for i in range(366)])
etot_proc = {}
for cat in cats:
    print(cat)
    
    # Read in kg/m2/s
    emi_proc[cat] = rp.read_preprocessed_mahuika_onecat(path_prepr_mah, cat, dates, lats_name, lons_name)
    
    
    
#%%
# Calculate emission totals in overlapping grid

etot_ori, etot_proc = {}, {}
for cat in cats:
    epri = emi_ori[cat]
    epos = emi_proc[cat]
    
    mask_ori  = create_lonlat_mask_bounds(lons_ori[cat], lats_ori[cat], [lons_name.min(), lons_name.max()], [lats_name.min(), lats_name.max()]).T
    mask_proc = create_lonlat_mask_bounds(lons_name, lats_name, [lons_ori[cat].min(), lons_ori[cat].max()], [lats_ori[cat].min(), lats_ori[cat].max()]).T
    
    etot_ori_i  = (emi_ori[cat].mean(axis=2)*area_pgrid_ori[cat])[mask_ori].sum() # kg/s
    etot_proc_i = (emi_proc[cat].mean(axis=(0,1))*area_pgrid_proc)[mask_proc].sum() # kg/s
    
    etot_ori[cat]  = etot_ori_i*kgs_to_ktyr
    etot_proc[cat] = etot_proc_i*kgs_to_ktyr


#%%
# Test the two against each other

# Emission totals
for cat in cats:
    print("%20.20s: Orig=%5.0f; Regr=%5.0f kt/yr"%(cat, etot_ori[cat], etot_proc[cat]))
    
# Make some maps
for cat in cats:
    fig, ax = plt.subplots(1,2, figsize=(20,7))
    
    emi_ori_i = emi_ori[cat].mean(axis=-1)
    ax[0].set_title("Original %s"%cat)
    x1,y1 = np.meshgrid(lons_ori[cat],lats_ori[cat])
    emi_ori_i = np.log(emi_ori_i)
    lo,hi = np.nanmin(emi_ori_i[emi_ori_i!=-np.inf]), np.nanmax(emi_ori_i)
    cp = ax[0].contourf(x1,y1, emi_ori_i, 100, cmap='viridis',vmin=lo,vmax=hi)
    plt.colorbar(cp,ax=ax[0])
    
    ax[1].set_title("Regridded %s"%cat)
    emi_proc_i = emi_proc[cat].mean(axis=(0,1))
    x2,y2 = np.meshgrid(lons_name,lats_name)
    cp = ax[1].contourf(x2,y2, np.log(emi_proc_i), 100, cmap='viridis',vmin=lo,vmax=hi)
    ax[1].set_xlim(ax[0].get_xlim())
    ax[1].set_ylim(ax[0].get_ylim())
    plt.colorbar(cp,ax=ax[1])
    
    plt.savefig("Figures/Preprocessing_tests/MahuikaAuckland/comparison_%s_%s.png"%(domain, cat))


#%%
# ------- 2. UrbanVPRM -------

# Read original emissions
dates = [datetime(2018,1,1) + timedelta(days=i) for i in range(365)]
lons_ori, lats_ori, emi_ori = rp.read_UrbanVPRM(dates)
area_pgrid_ori = st.calc_area_per_gridcell(lats_ori, lons_ori)

# Read preprocessed emissions
emi_proc = rp.read_preprocessed_urbanvprm(dates, domain)
# Monthly averaged
emi_proc_m = rp.read_preprocessed_urbanvprm_monthly(dates, domain)

# Calculate emission totals
mask_ori  = create_lonlat_mask_bounds(lons_ori, lats_ori, [lons_name.min(), lons_name.max()], [lats_name.min(), lats_name.max()]).T
mask_proc = create_lonlat_mask_bounds(lons_name, lats_name, [lons_ori.min(), lons_ori.max()], [lats_ori.min(), lats_ori.max()]).T

etot_ori    = kgs_to_ktyr*(emi_ori.mean(axis=(0,1))*area_pgrid_ori)[mask_ori].sum() # kg/s
etot_proc   = kgs_to_ktyr*(emi_proc.mean(axis=(0,1))*area_pgrid_proc)[mask_proc].sum() # kg/s
etot_proc_m = kgs_to_ktyr*(emi_proc_m.mean(axis=(0,1))*area_pgrid_proc)[mask_proc].sum() # kg/s

#%%
# Compare the two

# Emission totals
print("UrbanVPRM NEE: Orig=%5.0f; Regr=%5.0f kt/yr ; Regr_m=%5.0f kt/yr"%(etot_ori, etot_proc, etot_proc_m))

# Make some maps
fig, ax = plt.subplots(1,3, figsize=(30,7))

ax[0].set_title("Original")
x1,y1 = np.meshgrid(lons_ori,lats_ori)
emi_ori_i = np.log(emi_ori.mean(axis=(0,1)))
lo,hi = np.nanmin(emi_ori_i[emi_ori_i!=-np.inf]), np.nanmax(emi_ori_i)
cp = ax[0].contourf(x1,y1, emi_ori_i, 100, cmap='viridis',vmin=lo,vmax=hi)
plt.colorbar(cp,ax=ax[0])

ax[1].set_title("Regridded")
x2,y2 = np.meshgrid(lons_name,lats_name)
cp = ax[1].contourf(x2,y2, np.log(emi_proc.mean(axis=(0,1))), 100, cmap='viridis',vmin=lo,vmax=hi)
ax[1].set_xlim(ax[0].get_xlim())
ax[1].set_ylim(ax[0].get_ylim())
plt.colorbar(cp,ax=ax[1])

ax[2].set_title("Regridded, monthly")
x2,y2 = np.meshgrid(lons_name,lats_name)
cp = ax[2].contourf(x2,y2, np.log(emi_proc_m.mean(axis=(0,1))), 100, cmap='viridis',vmin=lo,vmax=hi)
ax[2].set_xlim(ax[0].get_xlim())
ax[2].set_ylim(ax[0].get_ylim())
plt.colorbar(cp,ax=ax[2])

plt.savefig("Figures/Preprocessing_tests/UrbanVPRM/comparison_NEE_%s.png"%(domain))

#%%
# ------- 3. BiomeBGC -------

varb = 'GEE'

# Read original emissions
dates = [datetime(2022,1,1) + timedelta(days=i) for i in range(365)]
lons_ori, lats_ori, emi_ori = rp.read_BiomeBGC_year(2022, varb=varb)
area_pgrid_ori = st.calc_area_per_gridcell(lats_ori, lons_ori)

# Read preprocessed emissions
emi_proc = rp.read_preprocessed_biomebgc(dates, domain, varb=varb)
# Monthly averaged
# emi_proc_m = rp.read_preprocessed_biomebgc_monthly(dates, domain, varb='NEE')

# Calculate emission totals
mask_ori  = create_lonlat_mask_bounds(lons_ori, lats_ori, [lons_name.min(), lons_name.max()], [lats_name.min(), lats_name.max()]).T
mask_proc = create_lonlat_mask_bounds(lons_name, lats_name, [lons_ori.min(), lons_ori.max()], [lats_ori.min(), lats_ori.max()]).T

etot_ori    = kgs_to_ktyr*(emi_ori.mean(axis=(0,1))*area_pgrid_ori)[mask_ori].sum() # kg/s
etot_proc   = kgs_to_ktyr*(emi_proc.mean(axis=(0,1))*area_pgrid_proc)[mask_proc].sum() # kg/s
# etot_proc_m = kgs_to_ktyr*(emi_proc_m.mean(axis=(0,1))*area_pgrid_proc)[mask_proc].sum() # kg/s

#%%
# Compare the two

# Emission totals
print("BiomeBGC NEE: Orig=%5.0f; Regr=%5.0f kt/yr ; Regr_m=%5.0f kt/yr"%(etot_ori, etot_proc, etot_proc))

# Make some maps
fig, ax = plt.subplots(1,3, figsize=(30,7))

ax[0].set_title("Original")
x1,y1 = np.meshgrid(lons_ori,lats_ori)
emi_ori_i = np.log(emi_ori.mean(axis=(0,1)))
lo,hi = np.nanmin(emi_ori_i[emi_ori_i!=-np.inf]), np.nanmax(emi_ori_i)
cp = ax[0].pcolormesh(x1,y1, emi_ori_i,  cmap='viridis',vmin=lo,vmax=hi)
plt.colorbar(cp,ax=ax[0])
ax[0].set_xlim(lons_name.min(), lons_name.max())
ax[0].set_ylim(lats_name.min(), lats_name.max())

ax[1].set_title("Regridded")
x2,y2 = np.meshgrid(lons_name,lats_name)
cp = ax[1].contourf(x2,y2, np.log(emi_proc.mean(axis=(0,1))), 100, cmap='viridis',vmin=lo,vmax=hi)
ax[1].set_xlim(ax[0].get_xlim())
ax[1].set_ylim(ax[0].get_ylim())
plt.colorbar(cp,ax=ax[1])

ax[2].set_title("Regridded, monthly")
x2,y2 = np.meshgrid(lons_name,lats_name)
# cp = ax[2].contourf(x2,y2, np.log(emi_proc_m.mean(axis=(0,1))), 100, cmap='viridis',vmin=lo,vmax=hi)
ax[2].set_xlim(ax[0].get_xlim())
ax[2].set_ylim(ax[0].get_ylim())
plt.colorbar(cp,ax=ax[2])

plt.savefig("Figures/Preprocessing_tests/BiomeBGC/comparison_NEE_%s.png"%(domain))

#%%

def read_BiomeBGC_NEE(year, timezone='NZST'):
    '''
    These are Daemon's processed files. Annual files with hourly data in UTC
    '''
    
    path = '%s/BiomeBGC/'%rp.get_path_priors()
    fname = '%s/prior_%4.4i.nc'%(path, year)
    d = Dataset(fname)
    print(d,d['ER'])
    ER  = d['ER'][:]  # (time, nlat, nlon) / gCO2 m-2 s-1
    GPP = d['GPP'][:]
    NEE = np.array(ER - GPP)
    NEE[NEE==1e20] = 0. # Fill values to zero
    NEE /= 1e3          # kgCO2 m-2 s-1
    
    lon, lat = d['lon'][:], d['lat'][:]
    
    if timezone=='UTC':
        pass
    elif timezone=='NZST':
        NEE = np.roll(NEE,12)
    
    # (nhour) to (nday,24)
    nhour,nlat,nlon = NEE.shape
    nday = int(nhour/24)
    NEE = NEE.reshape(nday, 24, nlat, nlon)
    
    return lon, lat, NEE

lons_ori, lats_ori, emi_ori = read_BiomeBGC_NEE(2020)
area_pgrid_ori = st.calc_area_per_gridcell(lats_ori, lons_ori)

#%%
etot_ori1    = kgs_to_ktyr*(emi_ori.mean(axis=(0,1))*area_pgrid_ori)[mask_ori].sum() # kg/s
etot_ori2    = kgs_to_ktyr*(emi_ori.mean(axis=(0,1))*area_pgrid_ori).sum() # kg/s
print('%2.2f kt/yr ; %2.2f kt/yr'%(etot_ori1,etot_ori2))

lonlo,lonhi = lons_name.min(),lons_name.max()
latlo,lathi = lats_name.min(),lats_name.max()

lonmask = ((lons_ori<=lonhi) & (lons_ori>=lonlo))
latmask = ((lats_ori<=lathi) & (lats_ori>=latlo))
nx,ny = lonmask.sum(), latmask.sum()


fig,ax = plt.subplots(2,1,figsize=(10,15))


emean   = np.mean(emi_ori, axis=(0,1))
emasked = emean[mask_ori].reshape(ny,nx)

x,y = np.meshgrid(lons_ori,lats_ori)
ax[0].contourf(x,y,emean==0)
st.plotSquare(ax[0], [lonlo,lonhi], [latlo,lathi])

x,y = np.meshgrid(lons_ori[lonmask], lats_ori[latmask])
ax[1].contourf(x,y,emasked==0)

#%%

# d = Dataset('/nesi/nobackup/niwa03154/kennettd/INPUT/priors/priors/land/Biome_CenW/prior_2020.nc')
d = Dataset('%s/priors/BiomeBGC/prior_2020.nc'%path_inv_base)
lon,lat = d['lon'][:], d['lat'][:]
ER, GPP = d['ER'][:], d['GPP'][:]

x,y = np.meshgrid(lon,lat)
#%%

NEE = ER- GPP
NEE[NEE==1e20] = 0. # Fill values to zero
NEE = np.roll(NEE,12, axis=0)

# (nhour) to (nday,24)
nhour,nlat,nlon = NEE.shape
nday = int(nhour/24)
NEE = NEE.reshape(nday, 24, nlat, nlon)

plt.figure()
NEE_plot = np.mean(NEE, axis=(0,1))==0
plt.contourf(x,y, NEE_plot)
plt.plot(174.56, -37.05, 'r.')

#%%
# ------- 3. EDGARv7   -------
path_prepr_edg = path_prepr + '/EDGARv7/%s/'%domain

# Read original emissions
lons_ori, lats_ori, emi_ori = rp.read_EDGARv7()
area_pgrid_ori = st.calc_area_per_gridcell(lats_ori, lons_ori)

# Read preprocessed emissions
dates = [datetime(2021,1,1)]
emi_proc = rp.read_preprocessed_edgarv7(path_prepr_edg, dates, lats_name, lons_name)

# Calculate emission totals
mask_ori  = create_lonlat_mask_bounds(lons_ori, lats_ori, [lons_name.min(), lons_name.max()], [lats_name.min(), lats_name.max()]).T
mask_proc = create_lonlat_mask_bounds(lons_name, lats_name, [lons_ori.min(), lons_ori.max()], [lats_ori.min(), lats_ori.max()]).T

etot_ori  = kgs_to_ktyr*(emi_ori*area_pgrid_ori)[mask_ori].sum() # kg/s
etot_proc = kgs_to_ktyr*(emi_proc.mean(axis=(0,1))*area_pgrid_proc)[mask_proc].sum() # kg/s


#%%
# Compare the two

# Emission totals
print("EDGARv7 tot: Orig=%5.0f; Regr=%5.0f kt/yr"%(etot_ori, etot_proc))

xlim = [lons_name.min(),lons_name.max()]
ylim = [lats_name.min(),lats_name.max()]

# Make some maps
fig, ax = plt.subplots(1,2, figsize=(20,7))

ax[0].set_title("Original")
x1,y1 = np.meshgrid(lons_ori,lats_ori)
emi_ori_i = np.log(emi_ori)
lo,hi = -23,-14
cp = ax[0].contourf(x1,y1, emi_ori_i, 100, cmap='viridis',vmin=lo,vmax=hi)
plt.colorbar(cp,ax=ax[0])

ax[1].set_title("Regridded")
x2,y2 = np.meshgrid(lons_name,lats_name)
cp = ax[1].contourf(x2,y2, np.log(emi_proc.mean(axis=(0,1))), 100, cmap='viridis',vmin=lo,vmax=hi)
[a.set_xlim(xlim) for a in ax]
[a.set_ylim(ylim) for a in ax]
plt.colorbar(cp,ax=ax[1])

plt.savefig("Figures/Preprocessing_tests/EDGARv7/comparison_tot_%s.png"%(domain))



#%%
# ------- 4. ODIAC   -------
path_prepr_odiac = path_prepr + '/ODIAC/%s/'%domain

# Read original emissions
dates = np.array([datetime(2021,i,1) for i in range(1,13)])
lonb = lons_name.min()-0.1, lons_name.max()+0.1
latb = lats_name.min()-0.1, lats_name.max()+0.1
lons_ori, lats_ori, emi_ori = rp.read_ODIAC(dates,lonb,latb)
area_pgrid_ori = st.calc_area_per_gridcell(lats_ori, lons_ori)

# Read preprocessed emissions
emi_proc = rp.read_preprocessed_odiac(path_prepr_odiac, dates, lats_name, lons_name)

# Calculate emission totals
mask_ori  = create_lonlat_mask_bounds(lons_ori, lats_ori, [lons_name.min(), lons_name.max()], [lats_name.min(), lats_name.max()]).T
mask_proc = create_lonlat_mask_bounds(lons_name, lats_name, [lons_ori.min(), lons_ori.max()], [lats_ori.min(), lats_ori.max()]).T

etot_ori  = kgs_to_ktyr*(emi_ori.mean(axis=(0,1))*area_pgrid_ori)[mask_ori].sum() # kg/s
etot_proc = kgs_to_ktyr*(emi_proc.mean(axis=(0,1))*area_pgrid_proc)[mask_proc].sum() # kg/s


#%%
# Compare the two

# Emission totals
print("ODIAC tot: Orig=%5.0f; Regr=%5.0f kt/yr"%(etot_ori, etot_proc))

xlim = [lons_name.min(),lons_name.max()]
ylim = [lats_name.min(),lats_name.max()]

# Make some maps
fig, ax = plt.subplots(1,2, figsize=(20,7))

ax[0].set_title("Original")
x1,y1 = np.meshgrid(lons_ori,lats_ori)
emi_ori_i = np.log(emi_ori.mean(axis=(0,1)))
lo,hi = -23,-14
cp = ax[0].contourf(x1,y1, emi_ori_i, 100, cmap='viridis',vmin=lo,vmax=hi)
plt.colorbar(cp,ax=ax[0])

ax[1].set_title("Regridded")
x2,y2 = np.meshgrid(lons_name,lats_name)
cp = ax[1].contourf(x2,y2, np.log(emi_proc.mean(axis=(0,1))), 100, cmap='viridis',vmin=lo,vmax=hi)
[a.set_xlim(xlim) for a in ax]
[a.set_ylim(ylim) for a in ax]
plt.colorbar(cp,ax=ax[1])

plt.savefig("Figures/Preprocessing_tests/ODIAC/comparison_tot_%s.png"%(domain))








