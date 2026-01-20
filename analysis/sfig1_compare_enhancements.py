#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:19:43 2024

@author: nauss
"""

from inversion_base import InversionBase
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from base_paths import path_figs
path_figs = '%s/meteo/'%path_figs
from datetime import timedelta,datetime
sns.set_context('talk')

def read_enhancements_NAME(site, dates_day, domain, prior, simulation):
    invB = InversionBase(simulation)
    enh = np.zeros((len(dates_day), 24))
    for iday,day in enumerate(dates_day):
        enhi = invB.read_enhancements_basegrid(day, domain, prior, site)
        enh[iday] = enhi.sum(axis=(-1,-2,-3))
    return enh

sites = 'AUT','NWO','MKH','TKA'
priors = 'MahuikaAuckland','UrbanVPRM'
simulations = 'baseAKLNWP_base','baseNZCSM_base'
domains = 'Mah0p3_in','Mah0p3_out'

date_start = datetime(2022,1,1)
date_end = datetime(2022,12,31)

dates_day = []
date_curr = date_start
while date_curr<=date_end:
    dates_day.append(date_curr)
    date_curr += timedelta(days=1)

dates   = {}
enh_obs, enh_obs_sd = {},{}
enh_sim = {}

for sim in simulations:
    enh_sim[sim] = {}
    
    for isite,site in enumerate(sites):
        
        enh_sim[sim][site] = {prior:0 for prior in priors}
        for domain in domains:
            for prior in priors:
                enh_sim[sim][site][prior] += read_enhancements_NAME(site, dates_day, domain, prior, sim)
            
            
#%%

dates_hours = [datetime(d.year, d.month, d.day, hour) for d in dates_day for hour in range(24)]

nplot = len(sites)
fig,ax = plt.subplots(nplot,1, figsize=(10,5*nplot))

prior='MahuikaAuckland'

site_labels = {'TKA':'Takarunga', 'MKH':'Manukau Heads', 'AUT':'Auckland University of Technology', 'NWO':'Pourewa'}
sim_labels  = {'baseAKLNWP_base':'Auckland model', 'baseNZCSM_base':'NZCSM'}

colors = sns.color_palette('Set2')[1:]
for i,site in enumerate(sites):
    ax[i].set_title(site)
    
    for isim,sim in enumerate(simulations):
        ax[i].set_title(site_labels[site])
        ax[i].plot(dates_hours, enh_sim[sim][site][prior].flatten(), color=colors[isim], label=sim_labels[sim])
    
    ax[i].legend(loc='best')
    
fig.tight_layout() 
    
    
#%%

prior='UrbanVPRM'

sns.set_context('talk', font_scale=1.5)

nplot = len(sites)
fig,ax = plt.subplots(2,2, figsize=(20,10))
ax = ax.flatten()
sim_labels  = {'baseAKLNWP_base':'333-m', 'baseNZCSM_base':'1.5-km'}
colors = sns.color_palette('Set2')[1:]
for i,site in enumerate(sites):
    ax[i].set_title(site)
    
    for isim,sim in enumerate(simulations):
        ax[i].set_title(site_labels[site])
        ax[i].plot(range(24), enh_sim[sim][site][prior].mean(axis=0), color=colors[isim], label=sim_labels[sim])
    
[ax[i].set_ylabel('Enhancement\n[ppm]') for i in [0,2]]
[ax[i].legend(loc='best') for i in [0,3]]
    
[ax[i].set_xlabel("Hour in day") for i in [0,1]]
fig.tight_layout() 
fig.savefig("%s/enhancements_aklnwp_vs_nzcsm"%path_figs,dpi=280)
    
    
#%%
prior= 'MahuikaAuckland'
for site in sites:
    a = np.sum(np.abs(enh_sim['baseAKLNWP_base'][site][prior].mean(axis=0)))
    b = np.sum(np.abs(enh_sim['baseNZCSM_base'][site][prior].mean(axis=0)))
    err = 100*np.abs((a-b))/((a+b)/2)
    print('%s : %2.2f %% error'%(site,err))
    
    
    
    
    
    
    


            
            
            