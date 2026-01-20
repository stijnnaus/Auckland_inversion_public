#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:40:39 2024

@author: nauss
"""

import functions_stijn as st
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk',font_scale=1.2)
import inversion as inv
import numpy as np
import read_obs as ro
import os

from base_paths import path_figs
path_figs = '%s/maps/'%path_figs
if not os.path.exists(path_figs):
    os.makedirs(path_figs)
    
#%%

domains = 'Mah0p3','In1p5', 'Out7p0'
ndomain = len(domains)
lonb = np.zeros((ndomain, 2))
latb = np.zeros((ndomain, 2))
for i,domain in enumerate(domains):
    lons, lats = inv.getLonLatNAME(domain, bounds=True)
    lonb[i] = lons.min(),lons.max()
    latb[i] = lats.min(),lats.max()


    
fig, ax, cl = st.makeMapCartopy(lonb[-1],latb[-1], fig_scale=2.0)

for i,dom in enumerate(domains):
    xx,yy = lonb[i]-cl, latb[i]
    st.plotSquare(ax, xx, yy, color='k')

fig.savefig('%s/map_NAME_domains.png'%path_figs)

#%%

domains = 'akl_0p3','Out7p0'
ndomain = len(domains)
lonb = np.zeros((ndomain, 2))
latb = np.zeros((ndomain, 2))
for i,domain in enumerate(domains):
    lons, lats = inv.getLonLatNAME(domain, bounds=True)
    lonb[i] = lons.min(),lons.max()
    latb[i] = lats.min(),lats.max()


    
fig, ax, cl = st.makeMapCartopy(lonb[-1],latb[-1], fig_scale=2.0)

for i,dom in enumerate(domains):
    xx,yy = lonb[i]-cl, latb[i]
    st.plotSquare(ax, xx, yy, color='k')

fig.savefig('%s/map_resolution_domains.png'%path_figs)

#%%

from inversion_base import InversionBase

configname = 'baseAKLNWP_base_1month'
invb = InversionBase(configname)


domains = 'Mah0p3_in', 'Mah0p3_out','In1p5', 'Out7p0'
ndomain = len(domains)
lonb = np.zeros((ndomain, 2))
latb = np.zeros((ndomain, 2))
for i,domain in enumerate(domains):
    bounds = invb.rc.outer_bounds[domain]
    if bounds is None:
        lons, lats = inv.getLonLatNAME(domain, bounds=True)
        lonb[i] = lons.min(),lons.max()
        latb[i] = lats.min(),lats.max()
        
    else:
        lonb[i] = bounds['lon']
        latb[i] = bounds['lat']


    
fig, ax, cl = st.makeMapCartopy(lonb[-1],latb[-1], fig_scale=2.0)

for i,dom in enumerate(domains):
    xx,yy = lonb[i]-cl, latb[i]
    st.plotSquare(ax, xx, yy, color='k')

fig.savefig('%s/map_inversion_domains.png'%path_figs)

#%%

# Plot the different grid resolutions
rc = invb.rc

domain_inv, domain_NAME = 'Mah0p3_in', 'Mah0p3'

lons_NAME, lats_NAME = inv.getLonLatNAME(domain_NAME, bounds=True)
lonb_inv , latb_inv = invb.rc.outer_bounds[domain_inv]['lon'], invb.rc.outer_bounds[domain_inv]['lat']

nx_inv, ny_inv   = rc.nx_inv[domain_inv], rc.ny_inv[domain_inv]
nx_base, ny_base = rc.nx_base[domain_inv], rc.ny_base[domain_inv]

lons_inv = np.linspace(lonb_inv[0], lonb_inv[1], nx_inv+1)
lats_inv = np.linspace(latb_inv[0], latb_inv[1], ny_inv+1)

lons_base = np.linspace(lonb_inv[0], lonb_inv[1], nx_base+1)
lats_base = np.linspace(latb_inv[0], latb_inv[1], ny_base+1)

gridress = [[lons_NAME, lats_NAME], [lons_base, lats_base], [lons_inv, lats_inv]]

for i,(lons,lats) in enumerate(gridress):
    print(len(lons)*len(lats))
    fig, ax, cl = st.makeMapCartopy(lonb_inv,latb_inv, fig_scale=2.0)
    
    for lon in lons:
        ax.plot([lon-cl,lon-cl], [lats.min(), lats.max()], 'k-')
    for lat in lats:
        ax.plot([lons.min()-cl,lons.max()-cl], [lat,lat], 'k-')
































