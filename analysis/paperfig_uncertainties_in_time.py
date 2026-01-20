#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:49:31 2025

@author: nauss
"""



from datetime import timedelta, datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from base_paths import path_figs
path_figs = '%s/osse_longterm/'%path_figs
from postprocessing import Postprocessing
import functions_stijn as st
from inversion_main import Inversion
import inversion as inv
from postprocessing_truth import Postprocess_truth
from postpostprocess_multi import Postpostprocess_multi
import time

import cartopy.crs as ccrs
import seaborn as sns
from cartopy.io import shapereader
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.gridspec import GridSpec
    

path = '/nesi/nobackup/niwa03154/nauss/Data/observations/aerial'
coastline_filename = '%s/nz-coastlines-and-islands-polygons-topo-1250k.shp'%path
shp = shapereader.Reader(coastline_filename)

#%%

t0 = time.time()

inversion_names = 'baseAKLNWP_base',#'baseAKLNWP_mahk_bias',#'baseAKLNWP_odiac'
domain_inv = 'Mah0p3_in'
startdate = datetime(2022,1,1)
enddate = datetime(2022,3,1)
dt_inversion = timedelta(days=28)
dt_spinup    = timedelta(days=7)
dt_spindown  = timedelta(days=7)

postpr_m = {}
for invname in inversion_names:
    print(invname)
    postpr_m[invname] = Postpostprocess_multi(invname, startdate, enddate, dt_inversion, dt_spinup, dt_spindown)
    postpr_m[invname].read_all_postprocessed_data()
    

pp = postpr_m['baseAKLNWP_base']
lon_inv, lat_inv = pp.inversion_example.get_lonlat_invgrid(domain_inv)
lonb_inv, latb_inv = pp.inversion_example.get_lonlat_invgrid(domain_inv, bounds=True)

#%%

true_label = postpr_m[invname].rc.osse_truth_label

startdate = postpr_m[invname].dates_all['daily'][0] + timedelta(seconds=25*3600)
enddate   = postpr_m[invname].dates_all['daily'][-1]

pp_true = Postprocess_truth(true_label, startdate=startdate, enddate=enddate)
pp_true.run_standard_postprocessing()
    
#%%

postpr = postpr_m['baseAKLNWP_base']

inventories = ['MahuikaAuckland','UrbanVPRM']
nplot = len(inventories)


fig,ax = plt.subplots(nplot, 1, figsize=(10,10))
for i,inventory in enumerate(inventories):
    ax[i].set_title(inventory)
    
    etru = pp_true.emis_agg['per_domain']['per_timestep'][inventory][:,0]
    epri = postpr.emis_all['prior']['per_domain']['per_timestep'][inventory][:,0]
    epos = postpr.emis_all['posterior']['per_domain']['per_timestep'][inventory][:,0]
    
    unc_pri = np.sqrt(postpr.unc_all_abs['prior']['per_domain']['per_timestep'][inventory][inventory][:,0])
    unc_pos = np.sqrt(postpr.unc_all_abs['posterior']['per_domain']['per_timestep'][inventory][inventory][:,0])
    
    nums  = [1, 8, 8*7, 8*28, len(etru)]
    rms   = np.zeros((2,len(nums)))
    unc_B = np.zeros((2,len(nums)))
    for j,n in enumerate(nums):
        
        nday = (n/8)
        etru_i = etru.reshape(-1,n).sum(axis=1) / nday
        
        for k,(emi,unc) in enumerate(zip([epri,epos],[unc_pri, unc_pos])):
            emi_i = emi.reshape(-1,n).sum(axis=1) / nday
            unc_i = unc.reshape(-1,n).sum(axis=1) / nday
            
            rms[k,j]   = np.sqrt(np.mean( (emi_i-etru_i)**2 ))
            unc_B[k,j] = np.sqrt(np.mean( (unc_i)**2 ))
            
    x = np.arange(len(nums))
    xticks = ['3-hr','day','week','month','total']
    for k,label in enumerate(['prior','posterior']):
        ax[i].plot(x, rms[k]/1e6, label='RMS w truth %s'%label)
        ax[i].plot(x, unc_B[k]/1e6, label='%s unc in B'%label)
    
    ax[i].legend(loc='best')
    ax[i].set_ylabel("Flux error [Gg/day]")
    ax[i].set_xticks(x)
    ax[i].set_xticklabels(xticks)
    
    
    
plt.tight_layout()
    
























    