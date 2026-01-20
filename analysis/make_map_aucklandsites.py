###############################################################################
# This script plots Raukumara CO2 fluxes from the Biome-BGC Model
#
###############################################################################

print('------------------------ Plot Auckland Sites -------------------------')

# ---------------------------- Import Packages --------------------------------
import cartopy.crs as ccrs
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib as mpl
import matplotlib.pyplot as plt   
import matplotlib.ticker as mticker
import numpy as np
from base_paths import path_base, path_figs
path_figs = '%s/maps/'%path_figs
import rioxarray as rio
import read_obs as ro
import xarray as xr

sites = ['MKH', 'AUT', 'NWO', 'TKA', ]#'MOTAT','AklAirp','Mangere','Sky']
lon_sites, lat_sites = np.zeros(len(sites)), np.zeros(len(sites))
for i,site in enumerate(sites):
    lon_sites[i], lat_sites[i] = ro.get_site_coords(site)
    
    
path = '%s/observations/aerial/'%path_base
coastline_filename = '%s/nz-coastlines-and-islands-polygons-topo-1250k.shp'%path
aerial_filename = '%s/92L4J-92L1S.tif'%path

#%%

# Read yearly averaged footprint
fname = '%s/footprints_processed/mean_footprint_filtered.nc'%path_base

from netCDF4 import Dataset
with Dataset(fname,'r') as d:
    lon_fp = d['lon'][:]
    lat_fp = d['lat'][:]
    footprint_per_site = d['mixing_ratio'][:]

#%%

# ---------------------------- Plot Auckland Sites ----------------------------
# Create grid axis
plt.figure()
ax = plt.axes(projection=ccrs.Mercator())
ax.set_extent([174.5, 175, -36.6, -37.1], crs=ccrs.PlateCarree())

# Plot coastlines

shp = shapereader.Reader(coastline_filename)
for record, geometry in zip(shp.records(), shp.geometries()):
    ax.add_geometries([geometry], ccrs.Mercator(), facecolor='none',
                      linewidth = 0.7, edgecolor='k',zorder=9)
    
# Plot gridlines
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='silver', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlocator = mticker.FixedLocator([174.5, 174.6, 174.7, 174.8, 174.9, 175])
gl.ylocator = mticker.FixedLocator([-36.6, -36.7, -36.8, -36.9, -37, -37.1])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 6, 'color': 'k'}
gl.ylabel_style = {'size': 6, 'color': 'k'}
# Plot aerial imagery
image = rio.open_rasterio(aerial_filename)
image.plot.imshow(rgb='band', ax=ax)
ax.set_title('')
# Plot station locations


ax.scatter(lon_sites, lat_sites, transform=ccrs.PlateCarree(), c='r', s=4, zorder=11)

# Plot average footprint
vmin,vmax = -13,-7
fp_tot = footprint_per_site.sum(axis=0)
x,y = np.meshgrid(lon_fp,lat_fp)
fp_tot = np.log(fp_tot)
fp_tot[fp_tot<vmin] = np.nan
fp_tot[fp_tot>vmax] = vmax
cp = ax.contourf(x,y, fp_tot, zorder=8, transform=ccrs.PlateCarree(), extend='max', alpha=0.7, vmin=vmin, vmax=vmax)
cb = plt.colorbar(cp, ticks=np.arange(-13,-6), label='Footprint sensitivity\n[log(ppm g$^{-1}$ m$^2$ s$^1$)]')
# cb.set_ylim(-12.-7)
# cb.set_yticks()

bbox = {'facecolor':'white', 'edgecolor':'none', 'alpha':0.4, 'pad':0.2}
fontsize, textcolor, textweight = 8, 'k', 'bold'
for i, site in enumerate(sites):
    xoffset = len(site)/2.*0.014
    yoffset = -0.017 #if site=='AUT' else 0.007
    ax.text(lon_sites[i]-xoffset, lat_sites[i]+yoffset, site, weight=textweight, 
            fontsize=fontsize, color=textcolor,
            transform=ccrs.PlateCarree(), bbox=bbox,zorder=10)
# Save plot
plt.tight_layout()
plt.savefig('%s/auckland_sites+footprint.png'%path_figs, dpi=320)

#%%


sites = ['MKH', 'AUT', 'NWO', 'TKA']
lon_sites, lat_sites = np.zeros(len(sites)), np.zeros(len(sites))
for i,site in enumerate(sites):
    lon_sites[i], lat_sites[i] = ro.get_site_coords(site)

# ---------------------------- Plot Auckland Sites ----------------------------
# Create grid axis
fig = plt.figure()

ax = plt.axes(projection=ccrs.Mercator())
extent = [174.3, 175.2, -36.6, -37.2]
ax.set_extent(extent, crs=ccrs.PlateCarree())
# Plot coastlines

shp = shapereader.Reader(coastline_filename)
for record, geometry in zip(shp.records(), shp.geometries()):
    ax.add_geometries([geometry], ccrs.Mercator(), facecolor='none',
                      linewidth = 0.3, edgecolor='k')
# Plot gridlines
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='silver', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlocator = mticker.FixedLocator([174.5,175])
gl.ylocator = mticker.FixedLocator([-37,-36.8])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 10, 'color': 'k'}
gl.ylabel_style = {'size': 10, 'color': 'k'}
# Plot aerial imagery
image = rio.open_rasterio(aerial_filename)
image.plot.imshow(rgb='band', ax=ax)
ax.set_title('')
# Plot station locations


ax.scatter(lon_sites, lat_sites, transform=ccrs.PlateCarree(), facecolor='r', edgecolor='k', s=45, zorder=10)


bbox = {'facecolor':'white', 'edgecolor':'none', 'alpha':0.4, 'pad':0.2}
fontsize, textcolor, textweight = 14, 'k', 'bold'

kwargs = {'weight':textweight, 'fontsize':fontsize, 'color':textcolor, 'transform':ccrs.PlateCarree(), 'bbox':bbox}
ax.text(lon_sites[0]-0.05, lat_sites[0]-0.035, 'MKH', **kwargs)
ax.text(lon_sites[1]-0.1 , lat_sites[1]-0.01 , 'AUT', **kwargs)
ax.text(lon_sites[2]+0.015, lat_sites[2]-0.01, 'NWO', **kwargs)
ax.text(lon_sites[3]-0.05, lat_sites[3]+0.015, 'TKA', **kwargs)


# Save plot
fig.tight_layout()
fig.savefig('/home/nauss/Code/maps/Figures/auckland_sites_zoomout2.png', dpi=320)
plt.close()

#%%
# ---------------------------- Plot Auckland Sites ----------------------------


sites = ['AUT', 'NWO', 'TKA', 'MOTAT','Sky']
lon_sites, lat_sites = np.zeros(len(sites)), np.zeros(len(sites))
for i,site in enumerate(sites):
    lon_sites[i], lat_sites[i] = ro.get_site_coords(site)

# Create grid axis
plt.figure()
ax = plt.axes(projection=ccrs.Mercator())
ax.set_extent([174.7, 174.9, -36.8, -36.9], crs=ccrs.PlateCarree())
# Plot coastlines
shp = shapereader.Reader(coastline_filename)
for record, geometry in zip(shp.records(), shp.geometries()):
    ax.add_geometries([geometry], ccrs.Mercator(), facecolor='none',
                      linewidth = 0.5, edgecolor='k')
# Plot gridlines
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.3, color='silver', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlocator = mticker.FixedLocator([174.7, 174.75, 174.8, 174.85, 174.9])
gl.ylocator = mticker.FixedLocator([-36.8, -36.85, -36.9])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 10, 'color': 'k'}
gl.ylabel_style = {'size': 10, 'color': 'k'}

# Plot aerial imagery
image = rio.open_rasterio(aerial_filename)
image.plot.imshow(rgb='band', ax=ax)
ax.set_title('')

# Plot station locations
bbox = {'facecolor':'white', 'edgecolor':'none', 'alpha':0.4, 'pad':0.2}
fontsize, textcolor, textweight = 10, 'k', 'bold'
# ax.scatter(lon_sites, lat_sites, transform=ccrs.PlateCarree(), c='r', s=10, zorder=10)
for i, site in enumerate(sites):
    xoffset = len(site)/2.*0.0038
    yoffset = -0.005 if site=='AUT' else 0.002
    
    # ax.text(lon_sites[i]-xoffset, lat_sites[i]+yoffset, site, weight=textweight, 
    #         fontsize=fontsize, color=textcolor, transform=ccrs.PlateCarree(), bbox=bbox)
# Save plot
plt.tight_layout()
plt.savefig('Figures/auckland_nosites_zoom.png', dpi=320)
plt.close()
# -----------------------------------------------------------------------------
