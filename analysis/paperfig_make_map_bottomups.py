#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:01:50 2024

@author: nauss
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 08:48:14 2024

@author: nauss
"""
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import inversion as inv
import cartopy.crs as ccrs
import seaborn as sns
from cartopy.io import shapereader
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
import read_priors as rp
import functions_stijn as st
import os
from base_paths import path_figs
from base_paths import path_base

from matplotlib.gridspec import GridSpec

#%%

path_figs = '%s/priors/'%path_figs
if not os.path.exists(path_figs):
    os.makedirs(path_figs)


# I need to know number of holidays/weekends vs weekdays in a year
year = 2022
nday_in_year = st.get_nday_in_year(year)
idx_weekday, idx_weekend = st.get_idx_weekday_weekend([datetime(year,1,1) + timedelta(days=i) for i in range(nday_in_year)])
nday_weekday, nday_weekend = len(idx_weekday), len(idx_weekend)

domain = 'mah'
lons_b, lats_b = inv.getLonLatNAME(domain, bounds=True)
lons_c, lats_c = inv.getLonLatNAME(domain, bounds=False)

lonlo,lonhi = 174.05,175.32
latlo,lathi = -37.3, -36.108

path = '/%s/observations/aerial/'%path_base
coastline_filename = '%s/nz-coastlines-and-islands-polygons-topo-1250k.shp'%path

inversion_name = 'baseAKLNWP_base'

shp = shapereader.Reader(coastline_filename)


#%%

# Read Mahk
cats = rp.get_all_mahuika_cats()

# Since sea transport is on a different grid, we read the base grid set-up so we can
# regrid to that grid for easy plotting
cat_example = 'air_transport_CO2ff'
lons_mahk, lats_mahk, _ = rp.read_mahuika_onecat(cat_example, unit='kg/m2/s')

emi_mahk = np.zeros( (len(cats), 24, len(lats_mahk), len(lons_mahk)) )
emi_mahk_week = np.zeros( (len(cats), 24, len(lats_mahk), len(lons_mahk)) )
emi_mahk_wknd = np.zeros( (len(cats), 24, len(lats_mahk), len(lons_mahk)) )
for icat,cat in enumerate(cats):
    lons_cat, lats_cat, emi_mahk_i = rp.read_mahuika_onecat(cat, unit='kg/m2/s')
    
    if cat=='sea_transport_CO2ff':
        # Regrid to default grid
        regridder = st.make_xesmf_regridder(lons_cat, lats_cat, lons_mahk, lats_mahk, 'conservative')
        
        ntime = emi_mahk_i.shape[-1]
        emis_regr = np.zeros((len(lats_mahk), len(lons_mahk), ntime))
        for i in range(ntime):
            emis_regr[:,:,i] = regridder(emi_mahk_i[:,:,i])
        
        emi_mahk_i = emis_regr
        
    freq = rp.get_freq_from_cat_mah(cat)
    
    emii = np.moveaxis(emi_mahk_i, 2, 0) # Move hours to the front
    
    if  freq=='weekday1': 
        # Only weekday data is prescribed ; weekend is zero
        emi_mahk[icat] = (nday_weekday/nday_in_year)*emii + (nday_weekend/nday_in_year)*np.zeros_like(emii)
        
        emi_mahk_week[icat] = emii 
        emi_mahk_wknd[icat] = 0. 
        
    elif freq=='weekday2':
        # Weekend / weekday explicitly prescribed
        emi_mahk[icat] = ( (nday_weekend/nday_in_year)*emii[:,:,:,0] + (nday_weekday/nday_in_year)*emii[:,:,:,1] ) # 0 is weekend, 1 is weekday
        
        emi_mahk_week[icat] = emii[:,:,:,1]
        emi_mahk_wknd[icat] = emii[:,:,:,0]
        
    elif freq=='daily':
        emii = emii.reshape(366,24,len(lats_mahk),len(lons_mahk))
        
        # Remove Feb 29, 60th day of year, since inventory is for 2020 but we look at 2022
        idx_no_feb29 = list(range(0,59)) + list(range(60,366))
        emii_sel = emii[idx_no_feb29]
        
        print(emii.shape,emii_sel.shape)
        
        # We average over days
        emi_mahk[icat] = emii_sel.mean(axis=0)
        
        emi_mahk_week[icat] = emii[idx_weekday].mean(axis=0)
        emi_mahk_wknd[icat] = emii[idx_weekend].mean(axis=0)
        
        
    elif freq=='fixed':
        # One fixed diurnal cycle
        emi_mahk[icat] = emii
        
        emi_mahk_week[icat] = emii
        emi_mahk_wknd[icat] = emii
        
        
#%%

# Check emission totals

areas_mahk = st.calc_area_per_gridcell(lats_mahk, lons_mahk, bounds=False)
etot = 0.
for icat,cat in enumerate(cats):
    etot_i = np.sum(emi_mahk[icat].mean(axis=0)*areas_mahk*3600*24*365)/1e6
    etot += etot_i
    print( '%15.15s: %8.2f kt/year'%(cat,etot_i ) )
print("%15.15s: %8.2f kt/year"%('Total', etot))

#%%

# Read UrbanVPRM NEE
dates = np.array([datetime(2022,1,1) + timedelta(days=i) for i in range(365)])
lons_vprm, lats_vprm, emis_vprm = rp.read_UrbanVPRM(dates, varb='NEE') # kg/s/m2

#%%

sns.set_context('talk',font_scale=1.1)

plt.rc('legend',fontsize=16)

fig = plt.figure(figsize=(16,10.5))

# Plot Mahuika-Auckland
def setup_map_auckland(ax):
    ax.set_extent([lonlo,lonhi,latlo,lathi], crs=ccrs.PlateCarree())
    for record, geometry in zip(shp.records(), shp.geometries()):
        ax.add_geometries([geometry], ccrs.Mercator(), facecolor='none',
                          linewidth = 0.5, edgecolor='k',zorder=10)
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='silver', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([174.5,175])
    gl.ylocator = mticker.FixedLocator([-37,-36.5])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 15, 'color': 'k'}
    gl.ylabel_style = {'size': 15, 'color': 'k'}
    
    # Plot square for inner inversion domain
    args = {'linewidth':2.0, 'color':'k', 'alpha':1.0, 'linestyle':'-', 'zorder':10, 'transform':ccrs.PlateCarree()}
    xx,yy = [174.54, 174.98], [-37.1,-36.72]
    [ax.plot([xx[0],xx[1]], [yy[i],yy[i]], **args) for i in [0,1]]
    [ax.plot([xx[i],xx[i]], [yy[0],yy[1]], **args) for i in [0,1]]
    
    # Plot square for mid inversion domain
    xx,yy = [174.2, 175.1], [-37.25,-36.3]
    [ax.plot([xx[0],xx[1]], [yy[i],yy[i]], **args) for i in [0,1]]
    [ax.plot([xx[i],xx[i]], [yy[0],yy[1]], **args) for i in [0,1]]

gs = GridSpec(100, 100)

axl_top = fig.add_subplot(gs[:60,:43], projection=ccrs.Mercator())
axl_bot = fig.add_subplot(gs[70:,:43])
axr_top = fig.add_subplot(gs[:60,57:], projection=ccrs.Mercator())
axr_bot = fig.add_subplot(gs[70:,57:])
axes = np.array([[axl_top, axl_bot], [axr_top, axr_bot]])

# ax0 = axl_top
# ax1 = axr_top

# ax0 = fig.add_subplot(1,2,1, projection=ccrs.Mercator())
# ax1 = fig.add_subplot(1,2,2, projection=ccrs.Mercator())

# Plot NZ
for ax in [axl_top,axr_top]:
    setup_map_auckland(ax)
    
    
# Plot Mahuika Auckland

emi_mahk_tot = emi_mahk.sum(axis=0).mean(axis=0) # Sum over categories, mean over hours
emi_mahk_tot[emi_mahk_tot*1e6<0.001] = np.nan

cp0 = axl_top.pcolormesh(lons_mahk, lats_mahk, emi_mahk_tot*1e6, vmax=1.0, cmap='cividis', transform=ccrs.PlateCarree())
plt.colorbar(cp0, ax=axl_top, extend='max',location='right',label='Anthropogenic emissions\n[mg m$^{-2}$ s$^{-1}$]')

# Mark the Glenbrook Steel mill
axl_top.scatter(174.729,-37.207, marker='x', color='red', zorder=12, linewidths=6.0, s=180, transform=ccrs.PlateCarree())
axl_top.scatter(174.729,-37.207, marker='x', color='white', zorder=12, linewidths=2.0, s=110, transform=ccrs.PlateCarree())

# Plot UrbanVPRM
    
emis_vprm_av = emis_vprm.mean(axis=(0,1)) # average over day, hour
emis_vprm_av[emis_vprm_av==0] = np.nan

cp1 = axr_top.pcolormesh(lons_vprm, lats_vprm, emis_vprm_av*1e6, vmin=-0.2, vmax=0.2, cmap='PRGn_r', transform=ccrs.PlateCarree())
plt.colorbar(cp1, ax=axr_top, extend='both',label='NEE\n[mg m$^{-2}$ s$^{-1}$]', ticks=[-0.2,-0.1,0,0.1,0.2])
    

# Diurnal cycle for innermost domain

latlims = -37.1, -36.72
lonlims = 174.54, 174.98

mask_mahk_lat = (lats_mahk>=latlims[0]) & (lats_mahk<=latlims[1])
mask_mahk_lon = (lons_mahk>=lonlims[0]) & (lons_mahk<=lonlims[1])
mask_mahk = np.outer(mask_mahk_lat, mask_mahk_lon)

lw = 4.
# Mahuika we distinguish weekend weekday
areas_mahk = st.calc_area_per_gridcell(lats_mahk, lons_mahk)
emis_mahk_av_hourly   = np.nansum( (emi_mahk*areas_mahk)[:,:,mask_mahk]     , axis=(0,-1))*3600 # kg/hour
emis_mahk_week_hourly = np.nansum( (emi_mahk_week*areas_mahk)[:,:,mask_mahk], axis=(0,-1))*3600 # kg/hour
emis_mahk_wknd_hourly = np.nansum( (emi_mahk_wknd*areas_mahk)[:,:,mask_mahk], axis=(0,-1))*3600 # kg/hour


colors = sns.color_palette('Set2')
axl_bot.plot(0.5+np.arange(24), emis_mahk_wknd_hourly/1e6, color=colors[1], linewidth=lw, label='Weekend')
axl_bot.plot(0.5+np.arange(24), emis_mahk_week_hourly/1e6, color=colors[2], linewidth=lw, label='Weekday')
axl_bot.plot(0.5+np.arange(24), emis_mahk_av_hourly/1e6  , color='k', linewidth=lw, label='Average')
axl_bot.set_xticks(np.arange(0,25,6))
axl_bot.legend(loc='best', ncol=1)
axl_bot.set_yticks([0,0.5,1])



# UrbanVPRM we distinguish winter/summer

mask_winter = [d.month in [5,6,7] for d in dates]
mask_summer = [d.month in [0,1,11] for d in dates]

mask_vprm_lat = (lats_vprm>=latlims[0]) & (lats_vprm<=latlims[1])
mask_vprm_lon = (lons_vprm>=lonlims[0]) & (lons_vprm<=lonlims[1])
mask_vprm = np.outer(mask_vprm_lat, mask_vprm_lon)

areas_vprm = st.calc_area_per_gridcell(lats_vprm, lons_vprm)
emis_vprm_hourly = np.nansum( (emis_vprm*areas_vprm)[:,:,mask_vprm], axis=(-1))*3600 # kg/hour

emis_vprm_winter = emis_vprm_hourly[mask_winter].mean(axis=0)
emis_vprm_summer = emis_vprm_hourly[mask_summer].mean(axis=0)

axr_bot.plot(0.5+np.arange(24), emis_vprm_summer/1e6, color=colors[0], linewidth=lw, label='Summer')
axr_bot.plot(0.5+np.arange(24), emis_vprm_winter/1e6, color=colors[3], linewidth=lw, label='Winter')
axr_bot.plot(0.5+np.arange(24), emis_vprm_hourly.mean(axis=0)/1e6, color='k', linewidth=lw, label='Average')
# Plot zero-line for NEE
axr_bot.set_xlim(axr_bot.get_xlim())
axr_bot.plot(axr_bot.get_xlim(), [0,0], 'k-', linewidth=1.0, alpha=0.5,zorder=-5)

axr_bot.set_xticks(np.arange(0,25,6))
axr_bot.legend(loc='best')

[a.set_xlabel("Hour in day") for a in [axl_bot,axr_bot]]
axl_bot.set_ylabel("Anth emissions\n[kton/hour]")
axr_bot.set_ylabel("NEE\n[kton/hour]")



# Panel labels
bbox=dict(boxstyle='square', fc="w", ec="k", alpha=0.9)
zorder=12
axl_top.text(0.02, 0.98, 'A', ha='left', va='top', fontsize=22, zorder=zorder, transform=axl_top.transAxes, bbox=bbox)
axr_top.text(0.02, 0.98, 'B', ha='left', va='top', fontsize=22, zorder=zorder, transform=axr_top.transAxes, bbox=bbox)
axl_bot.text(0.018, 0.96, 'C', ha='left', va='top', fontsize=22, zorder=zorder, transform=axl_bot.transAxes, bbox=bbox)
axr_bot.text(0.018, 0.96, 'D', ha='left', va='top', fontsize=22, zorder=zorder, transform=axr_bot.transAxes, bbox=bbox)
    
fig.savefig('%s/mah_vprm_maps_invdoms+diurn.png'%path_figs, dpi=300)

#%%
# Plot specific time

date = datetime(2022,1,22,6)

day = datetime(date.year,date.month,date.day)
iday = np.where(dates==day)[0][0]
ihour = date.hour

fig,ax = plt.subplots(1,1,figsize=(8,6.5))

cp = ax.pcolormesh(lons_vprm, lats_vprm, emis_vprm[iday,ihour], cmap='PRGn_r')
plt.colorbar(cp, ax=ax)






#%%

latlims = -37.1, -36.72
lonlims = 174.54, 174.98

mask_mahk_lat = (lats_mahk>=latlims[0]) & (lats_mahk<=latlims[1])
mask_mahk_lon = (lons_mahk>=lonlims[0]) & (lons_mahk<=lonlims[1])
mask_mahk = np.outer(mask_mahk_lat, mask_mahk_lon)

lw = 4.
# Mahuika we distinguish weekend weekday
areas_mahk = st.calc_area_per_gridcell(lats_mahk, lons_mahk)
emis_mahk_av_hourly   = np.nansum( (emi_mahk*areas_mahk)[:,:,mask_mahk]     , axis=(0,-1))*3600 # kg/hour
emis_mahk_week_hourly = np.nansum( (emi_mahk_week*areas_mahk)[:,:,mask_mahk], axis=(0,-1))*3600 # kg/hour
emis_mahk_wknd_hourly = np.nansum( (emi_mahk_wknd*areas_mahk)[:,:,mask_mahk], axis=(0,-1))*3600 # kg/hour


fig, ax = plt.subplots(1,2,figsize=(18,3.5))


colors = sns.color_palette('Set2')
ax[0].plot(0.5+np.arange(24), emis_mahk_wknd_hourly/1e3, color=colors[1], linewidth=lw, label='Weekend')
ax[0].plot(0.5+np.arange(24), emis_mahk_week_hourly/1e3, color=colors[2], linewidth=lw, label='Weekday')
# emis_mahk_av_hourly = (2/7)*emis_mahk_wknd_hourly + (5/7)*emis_mahk_week_hourly
ax[0].plot(0.5+np.arange(24), emis_mahk_av_hourly/1e3  , color='k', linewidth=lw, label='Average')
ax[0].set_xticks(np.arange(0,25,6))
ax[0].legend(loc='best', ncol=1)
ax[0].set_yticks([0,500,1000])



# UrbanVPRM we distinguish winter/summer

mask_winter = [d.month in [5,6,7] for d in dates]
mask_summer = [d.month in [0,1,11] for d in dates]

mask_vprm_lat = (lats_vprm>=latlims[0]) & (lats_vprm<=latlims[1])
mask_vprm_lon = (lons_vprm>=lonlims[0]) & (lons_vprm<=lonlims[1])
mask_vprm = np.outer(mask_vprm_lat, mask_vprm_lon)

areas_vprm = st.calc_area_per_gridcell(lats_vprm, lons_vprm)
emis_vprm_hourly = np.nansum( (emis_vprm*areas_vprm)[:,:,mask_vprm], axis=(-1))*3600 # kg/hour

emis_vprm_winter = emis_vprm_hourly[mask_winter].mean(axis=0)
emis_vprm_summer = emis_vprm_hourly[mask_summer].mean(axis=0)

ax[1].plot(0.5+np.arange(24), emis_vprm_summer/1e3, color=colors[0], linewidth=lw, label='Summer (DJF)')
ax[1].plot(0.5+np.arange(24), emis_vprm_winter/1e3, color=colors[3], linewidth=lw, label='Winter (JJA)')
ax[1].plot(0.5+np.arange(24), emis_vprm_hourly.mean(axis=0)/1e3, color='k', linewidth=lw, label='Average')
ax[1].set_xticks(np.arange(0,25,6))
ax[1].legend(loc='best')

[a.set_xlabel("Hour in day") for a in ax]
ax[0].set_ylabel("Flux [ton/hour]")

fig.savefig('%s/diurnal_cycle_priors.png'%path_figs, dpi=300)



#%%

# Same for ODIAC, since we also use it for the inner inventory

lonbounds_in = lonlo,lonhi
latbounds_in = latlo,lathi
dates_months = [datetime(2021,i,1) for i in range(1,13)]

lon_odc, lat_odc, emis_odc = rp.read_ODIAC(dates_months, lonbounds=lonbounds_in, latbounds=latbounds_in) # kg/m2/s

#%%

# Map ODIAC


fig = plt.figure(figsize=(16,10.5))
gs = GridSpec(100, 100)

ax_top = fig.add_subplot(gs[:60,:43], projection=ccrs.Mercator())
ax_bot = fig.add_subplot(gs[70:,:43])


setup_map_auckland(ax_top)

emis_odc_plot = np.copy(emis_odc)
emis_odc_plot[emis_odc_plot<0.001e-6] = np.nan

cp = ax_top.pcolormesh(lon_odc, lat_odc, np.nanmean(emis_odc_plot,axis=(0,1))*1e6, cmap='cividis', transform=ccrs.PlateCarree(), vmin=0, vmax=1.0)
plt.colorbar(cp, ax=ax_top, extend='max',location='right',label='Anthropogenic emissions\n[mg m$^{-2}$ s$^{-1}$]')


# Diurnal cycle
# Same as for Mahuika we want it only for inner domain

mask_odc_lat = (lat_odc<=latlims[1]) & (lat_odc>=latlims[0])
mask_odc_lon = (lon_odc<=lonlims[1]) & (lon_odc>=lonlims[0])
mask_odc = np.outer(mask_odc_lat, mask_odc_lon)

area_odc = st.calc_area_per_gridcell(lat_odc, lon_odc)

emis_odc_tot = np.nansum((emis_odc*area_odc)[:,:,mask_odc], axis=(2))*3600/1e3 # t/hour

for i in range(12):
    label = 'Per-month' if i==0 else None
    ax_bot.plot(np.arange(24)+0.5, emis_odc_tot[i], 'gray', linewidth=1.0, label=label)
    
ax_bot.plot(np.arange(24)+0.5, np.nanmean(emis_odc_tot,axis=0), 'k', linewidth=lw, label='Average')
ax_bot.set_ylim(0,900)
ax_bot.set_xticks(np.arange(0,25,6))
ax_bot.set_xlabel("Hour in day")
ax_bot.set_ylabel("Anth emissions\n[kton/hour]")
ax_bot.legend(loc='best')


fig.savefig("%s/map+diurnal_ODIAC.png"%path_figs, dpi=300)











    