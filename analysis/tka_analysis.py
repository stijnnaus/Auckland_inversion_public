#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 13:58:49 2024

@author: stijn
"""

from read_meteo_NAME import RetrievorWinddataNAME
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from base_paths import path_figs
import os
path_figs = '%s/meteo/'%path_figs

if not os.path.exists(path_figs):
    os.makedirs(path_figs)
    
sns.set_context('talk', font_scale=1.5)

sites = 'Takarunga', 'ManukauHeads', 'AucklandUni', 'Pourewa'

startdate = datetime(2022,2,3)
nday = 332
dates_days = np.array([startdate + timedelta(days=i) for i in range(nday)])
dates_hours = np.array([startdate + timedelta(seconds=3600*i) for i in range(nday*24)])

retr_333 = RetrievorWinddataNAME('baseAKLNWP_base')
meteo_333 = retr_333.retrieve_winddata_NAME(dates_hours, sites, overwrite=False)

retr_1p5 = RetrievorWinddataNAME('baseNZCSM_base')
meteo_1p5 = retr_1p5.retrieve_winddata_NAME(dates_hours, sites, overwrite=False)

labels = '333m', '1.5km'

fig,ax = plt.subplots(3, len(sites), figsize=(3*10,4*5))
for i,site in enumerate(sites):
    ax[0,i].set_title(site)
    for j,meteo in enumerate([meteo_333,meteo_1p5]):
        pbl = meteo[2][i].reshape(nday,24)
        t   = meteo[3][i].reshape(nday,24)
        rh  = meteo[4][i].reshape(nday,24)
        
        for l,v in enumerate([t,rh,pbl]):
            mn,sd = v.mean(axis=0), v.std(axis=0)
            ax[l,i].plot(mn, label=labels[j])
            ax[l,i].fill_between(range(24), mn-sd, mn+sd, alpha=0.2)

[ a.legend(loc='best') for a in ax[:,0]]
[ a.set_xlabel('Hour in day') for a in ax[-1]]


ax[0,0].set_ylabel("Temperature (C)")
ax[1,0].set_ylabel("Relative humidity (%)")
ax[2,0].set_ylabel("PBL height (m)")


plt.tight_layout()
fig.savefig('%s/meteo_4sites_333vs1p5'%path_figs, dpi=280)

#%%
sns.set_context('talk',font_scale=0.84)
fig,ax = plt.subplots(1,1,figsize=(6.5,3.8))
labels = '333-m','1.5-km'
for j,meteo in enumerate([meteo_333,meteo_1p5]):
    pbl = meteo[2][0].reshape(nday,24)
    mn,sd = pbl.mean(axis=0), pbl.std(axis=0)
    ax.plot(mn, label=labels[j])
    ax.fill_between(range(24), mn-sd, mn+sd, alpha=0.2)
ax.set_xlabel("Hour in day")
ax.set_ylabel("PBL height [m]")
ax.set_yticks([0,500,1000,1500])
ax.legend(loc='best')
fig.tight_layout()
plt.savefig('%s/PBL_height_TKA'%path_figs, dpi=280)
