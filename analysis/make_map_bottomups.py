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

path = '/nesi/nobackup/niwa03154/nauss/Data/observations/aerial'
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

emi_mahk = np.zeros( (len(cats), len(lats_mahk), len(lons_mahk)) )
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
    
    if  freq=='weekday1': 
        # Only weekday data is prescribed ; weekend is zero
        emii = emi_mahk_i.mean(axis=2)
        emi_mahk[icat] = (nday_weekday/nday_in_year)*emii + (nday_weekend/nday_in_year)*np.zeros_like(emii)
        
    elif freq=='weekday2':
        # Weekend / weekday explicitly prescribed
        emii = emi_mahk_i.mean(axis=2)
        emi_mahk[icat] = ( (nday_weekend/nday_in_year)*emii[:,:,0] + (nday_weekday/nday_in_year)*emii[:,:,1] ) # 0 is weekend, 1 is weekday
        
    else:
        # Hourly or daily, we just take the average
        emi_mahk[icat] = emi_mahk_i.mean(axis=2)


#%%

# Check emission totals

areas_mahk = st.calc_area_per_gridcell(lats_mahk, lons_mahk, bounds=False)
etot = 0.
for icat,cat in enumerate(cats):
    etot_i = np.sum(emi_mahk[icat]*areas_mahk*3600*24*365)/1e6
    etot += etot_i
    print( '%15.15s: %8.2f kt/year'%(cat,etot_i ) )
print("%15.15s: %8.2f kt/year"%('Total', etot))

#%%

# Read UrbanVPRM NEE
dates = np.array([datetime(2022,1,1) + timedelta(days=i) for i in range(365)])
lons_vprm, lats_vprm, emis_vprm = rp.read_UrbanVPRM(dates, varb='NEE') # kg/s/m2

#%%

sns.set_context('talk',font_scale=1.1)

fig = plt.figure(figsize=(16,6.5))

# Plot Mahuika-Auckland


ax0 = fig.add_subplot(1,2,1, projection=ccrs.Mercator())
ax1 = fig.add_subplot(1,2,2, projection=ccrs.Mercator())

# Plot NZ
for ax in [ax0,ax1]:
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
    
    
# Plot Mahuika Auckland

emi_mahk_tot = emi_mahk.sum(axis=0)
emi_mahk_tot[emi_mahk_tot*1e6<0.001] = np.nan

cp0 = ax0.pcolormesh(lons_mahk, lats_mahk, emi_mahk_tot*1e6, vmax=1.0, cmap='cividis', transform=ccrs.PlateCarree())
plt.colorbar(cp0, ax=ax0, extend='max',label='[mg m$^{-2}$ s$^{-1}$]')

# Mark the Glenbrook Steel mill
ax0.scatter(174.729,-37.207, marker='x', color='red', zorder=12, linewidths=6.0, s=180, transform=ccrs.PlateCarree())
ax0.scatter(174.729,-37.207, marker='x', color='white', zorder=12, linewidths=2.0, s=110, transform=ccrs.PlateCarree())

# Plot UrbanVPRM
    
emis_vprm_av = emis_vprm.mean(axis=(0,1)) # average over day, hour
emis_vprm_av[emis_vprm_av==0] = np.nan

cp1 = ax1.pcolormesh(lons_vprm, lats_vprm, emis_vprm_av*1e6, vmin=-0.2, vmax=0.2, cmap='PRGn_r', transform=ccrs.PlateCarree())
plt.colorbar(cp1, ax=ax1, extend='both',label='[mg m$^{-2}$ s$^{-1}$]', ticks=[-0.2,-0.1,0,0.1,0.2])
    

fig.tight_layout()
fig.savefig('%s/mah_vprm_maps_invdoms.png'%path_figs, dpi=320)

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

# Apply inversion scaling factors to emission maps
# def combine_state_scaling_with_emission_maps









    