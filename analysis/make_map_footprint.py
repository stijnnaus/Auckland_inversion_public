#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 08:48:14 2024

@author: nauss
"""

from enhancement_calculator import CalculatorEnhancementsForInversion
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import inversion as inv
import cartopy.crs as ccrs
from cartopy.io import shapereader
import rioxarray as rio
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
import glob


dates = [datetime(2022,2,9,6) + timedelta(seconds=3600*i) for i in range(1)]

isite = 1
domain = 'In1p5'
lons_b, lats_b = inv.getLonLatNAME(domain, bounds=True)
lons_c, lats_c = inv.getLonLatNAME(domain, bounds=False)

lonlo,lonhi = lons_b.min(), lons_b.max()
latlo,lathi = lats_b.min(), lats_b.max()

path = '/nesi/nobackup/niwa03154/nauss/Data/observations/aerial/'
coastline_filename = '%s/nz-coastlines-and-islands-polygons-topo-1250k.shp'%path

inversion_name = 'baseAKLNWP_base'

shp = shapereader.Reader(coastline_filename)

for date in dates:
    plt.figure(figsize=(8,6.3))
    ax = plt.axes(projection=ccrs.Mercator())
    ax.set_extent([lonlo,lonhi,latlo,lathi], crs=ccrs.PlateCarree())
    for record, geometry in zip(shp.records(), shp.geometries()):
        ax.add_geometries([geometry], ccrs.Mercator(), facecolor='none',
                          linewidth = 0.5, edgecolor='k',zorder=1)
        
    ax.set_title(date.strftime('%Y-%m-%d %H:00'))
    
    # Footprint inner domain
    domain = 'Mah0p3'
    lons_b, lats_b = inv.getLonLatNAME(domain, bounds=True)
    lons_c, lats_c = inv.getLonLatNAME(domain, bounds=False)
    lonlo, lonhi, latlo, lathi = lons_b.min(), lons_b.max(), lats_b.min(), lats_b.max()

    calci = CalculatorEnhancementsForInversion(inversion_name)
    tsteps, footprint_333 = calci.read_footprint(date, domain)
    
    vmin, vmax = -14, -10
    cp0 = ax.pcolormesh(lons_c, lats_c, np.log(footprint_333[isite].sum(axis=0)), transform=ccrs.PlateCarree(), vmax=vmax, vmin=vmin, zorder=10)
    
    # Footprint medium domain
    domain = 'In1p5'
    lons_b, lats_b = inv.getLonLatNAME(domain, bounds=True)
    lons_c, lats_c = inv.getLonLatNAME(domain, bounds=False)

    calci = CalculatorEnhancementsForInversion(inversion_name)
    tsteps, footprint_1p5 = calci.read_footprint(date, domain)
    
    # Mask out inner domain
    lonmask = (lons_c>lonlo) & (lons_c<lonhi)
    latmask = (lats_c>latlo) & (lats_c<lathi)
    mask_inner = np.outer(latmask,lonmask)
    footprint_1p5 = footprint_1p5[1].sum(axis=0)
    footprint_1p5[mask_inner] = np.nan
    
    cp1 = ax.pcolormesh(lons_c, lats_c, np.log(footprint_1p5), transform=ccrs.PlateCarree(), vmax=vmax, vmin=vmin, zorder=10)
    
    cticks = [-14, -12, -10]
    cbar = plt.colorbar(cp0, ax=ax, extend='both', ticks=cticks, label="[ppm g$^{-1}$ m$^{2}$ s]")
    cbar.ax.set_yticklabels(['10$^{%i}$'%ctick for ctick in cticks])  # vertically oriented colorbar
    
    # Plot AKL0p333 domain
    lons_b, lats_b = inv.getLonLatNAME('akl_0p3', bounds=True)
    
    args = {'linewidth':2.0, 'color':'k', 'alpha':1.0, 'linestyle':'-', 'zorder':10, 'transform':ccrs.PlateCarree()}
    xx,yy = [lons_b.min(), lons_b.max()], [lats_b.min(), lats_b.max()]
    [ax.plot([xx[0],xx[1]], [yy[i],yy[i]], **args) for i in [0,1]]
    [ax.plot([xx[i],xx[i]], [yy[0],yy[1]], **args) for i in [0,1]]
    
    # for aerial_filename in glob.glob('%s/9*.tif'%path)[:10]:
    #     image = rio.open_rasterio(aerial_filename)
    #     image.plot.imshow(rgb='band', ax=ax, zorder=20)
    
    
#%%

plt.figure(figsize=(8,6.3))
ax = plt.axes(projection=ccrs.Mercator())
ax.set_extent([lonlo-20,180,latlo-20,lathi+20], crs=ccrs.PlateCarree())

for aerial_filename in glob.glob('%s/lds/*.tif'%path)[:10]:
    
    image = rio.open_rasterio(aerial_filename)
    image.plot.imshow(rgb='band', ax=ax,alpha=0.5)
    


#%%




    
plt.show()
    
    
    
    
    
    
    