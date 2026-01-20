#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 12:14:20 2024

@author: stijn
"""

import iris
import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from base_paths import path_figs, path_base
import matplotlib
import seaborn as sns
sns.set_context('talk')
path_figs = '%s/topo/'%path_figs
if not os.path.exists(path_figs):
    os.makedirs(path_figs)

path = '%s/topo/'%path_base

fname_333m = os.path.join(path, 'TopogUMAKLNWP0p333.pp')
alt_333m = iris.load(fname_333m)[0]

fname_1p5km = os.path.join(path, 'TopogUMNZCSM.pp')
alt_1p5km = iris.load(fname_1p5km)[-1]

def plot_iris_cube(cube, xname='grid_longitude', yname='grid_latitude', vmin=None, vmax=None, ax=None, cmap='viridis'):
    
    proj_x = cube.coord(xname).points
    proj_y = cube.coord(yname).points
    xx, yy = np.meshgrid(proj_x, proj_y)
    
    
    if ax is None:
        return_fig = True
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    else:
        return_fig = False
    
    
    cs_nat = cube.coord_system()
    cs_nat_cart = cs_nat.as_cartopy_projection()
    
    
    data = np.clip(cube.data, vmin, vmax)
    cp = ax.pcolormesh(xx,yy, data, transform=cs_nat_cart, vmin=vmin, vmax=vmax, cmap=cmap)
    
    
    cs_tgt = iris.coord_systems.GeogCS(iris.analysis.cartography.DEFAULT_SPHERICAL_EARTH_RADIUS)
    # Again, convert it to a cartopy projection
    cs_tgt_cart = cs_tgt.as_cartopy_projection()
    
    lons, lats, _ = cs_tgt_cart.transform_points(cs_nat_cart, xx, yy).T
    
    if return_fig:
        return fig,ax, cp, lons,lats
    else:
        return ax, cp, lons, lats


vmin, vmax = alt_333m.data.min(), alt_333m.data.max()
vmin,vmax = 0,100

#%%

fig = plt.figure(figsize=(5.2,11))

gs = matplotlib.gridspec.GridSpec(50, 1)
ax_333 = fig.add_subplot(gs[:19, 0], aspect="equal", projection=ccrs.PlateCarree())
ax_1p5 = fig.add_subplot(gs[23:, 0], aspect="equal", projection=ccrs.PlateCarree())

cmap = 'cividis'

ax_333, cp_333, lons_333, lats_333 = plot_iris_cube(alt_333m, xname='longitude', yname='latitude', vmin=vmin, vmax=vmax, ax=ax_333, cmap=cmap)
ax_1p5, cp_1p5, lons_1p5, lats_1p5 = plot_iris_cube(alt_1p5km, vmin=vmin, vmax=vmax, ax=ax_1p5, cmap=cmap)

ax_1p5.set_xlim(ax_333.get_xlim())
ax_1p5.set_ylim(ax_333.get_ylim())

ax_1p5.set_title('1.5-km topography')
ax_333.set_title('333-m topography')

lon_tka, lat_tka = 174.7993, -36.8265
for a in [ax_333,ax_1p5]:
    a.scatter(lon_tka,lat_tka, s=200, facecolor='r', edgecolor='white')

fig.savefig('%s/topo_nozoom.png'%path_figs, dpi=280)

dlon, dlat = 0.1,0.1
for a in [ax_333,ax_1p5]:
    a.set_xlim(lon_tka-dlon, lon_tka+dlon)
    a.set_ylim(lat_tka-dlat, lat_tka+dlat)
    
    a.set_xticks([174.7,174.8,174.9])
    a.set_yticks([-36.9,-36.8])

ax_333.xaxis.set_visible(False)

fig.colorbar(cp_1p5, ax=ax_1p5, label='Altitude (m asl)', extend='max', location='bottom')
fig.tight_layout()
fig.savefig('%s/topo_zoom_TKA.png'%path_figs, dpi=280, bbox_inches='tight')

























