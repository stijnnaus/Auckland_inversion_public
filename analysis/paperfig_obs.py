
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper figures to explore how well observations are represented in the inversion.

Created on Wed Feb 12 10:14:39 2025

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
from postprocessing_truth import Postprocess_truth
from postpostprocess_multi import Postpostprocess_multi
import time

from matplotlib.gridspec import GridSpec
    

#%%

t0 = time.time()

inversion_names = ['baseAKLNWP_base', 'baseAKLNWP_odiac', 'baseAKLNWP_100m', 
                   'baseNZCSM_base',  'baseAKLNWP_only_AUT','baseAKLNWP_afternoon']
startdate = datetime(2022,1,1)
enddate = datetime(2022,12,15)
dt_inversion = timedelta(days=28)
dt_spinup    = timedelta(days=7)
dt_spindown  = timedelta(days=7)

postpr_m = {}
for invname in inversion_names:
    print(invname)
    postpr_m[invname] = Postpostprocess_multi(invname, startdate, enddate, dt_inversion, dt_spinup, dt_spindown)
    postpr_m[invname].read_all_postprocessed_data()
    
#%%
    
winter = np.arange(151,243)
summer = np.append(np.arange(0,62), np.arange(334,364))
pp = postpr_m['baseAKLNWP_base']
doys = [d.timetuple().tm_yday for d in pp.obs_all['prior']['AUT-MKH']['dates']]

print(sum([d in winter for d in doys]), sum(d in summer for d in doys))

#%%

invname = 'baseAKLNWP_base'
colors = sns.color_palette('Set2')
colors = np.array(sns.color_palette('Set2'))[[0,1,2]]

postpr_m_i = postpr_m[invname]

sites = postpr_m_i.obs_all['prior'].keys()
nsite = len(sites)

sns.set_context("talk",font_scale=1.2)

fig,ax = plt.subplots(nsite,2,figsize=(18,4*nsite))
if ax.shape==(2,):
    ax = ax.reshape(1,2)

for isite,site in enumerate(sites):
    
    ax[isite,0].set_title("Obs %s"%site)
    # for ilab,label in enumerate(['prior','posterior','true']):
    for ilab,label in enumerate(['prior','posterior','true']):
        dates = postpr_m_i.obs_all[label][site]['dates']
        co2   = postpr_m_i.obs_all[label][site]['co2']
        
        ax[isite,0].scatter(dates, co2, facecolor='none', edgecolor=colors[ilab], 
                            linewidth=1.5, label=label)
        ax[isite,0].set_ylim(-15,15)
        
    ax[isite,0].legend(loc='best')
        
    ax[isite,1].set_title("Difference w truth %s"%site)
    co2_true = postpr_m_i.obs_all['true'][site]['co2']
    for ilab,label in enumerate(['prior','posterior']):
        dates, co2 = postpr_m_i.obs_all[label][site]['dates'], postpr_m_i.obs_all[label][site]['co2']
        
        diff = co2-co2_true
        
        bias = np.mean(diff)
        rms = np.sqrt(np.mean(diff**2))
        
        ax[isite,1].scatter(dates, diff, facecolor='none', edgecolor=colors[ilab], 
                            linewidth=1.5, label=label)
        box = dict(boxstyle='square', facecolor='white', alpha=0.6)
        ax[isite,1].text(0.05+0.65*ilab,0.1, 'Bias %2.2f ppm\nRMS %2.2f ppm'%(bias,rms), 
                         fontsize=18, color=colors[ilab], transform=ax[isite,1].transAxes, bbox=box)
        ax[isite,1].set_ylim(-15,15)
    
    ax[isite,1].legend(loc='upper right')
    
[a.set_ylabel("CO$_2$ [ppm]") for a in ax.flatten()]    
[st.adjust_xticks_dates_to_m(a,month_fmt='%b') for a in ax.flatten()]


fig.tight_layout()
fig.savefig("%s/obs_202201_onesite.png"%path_figs)

#%%

invname = 'baseAKLNWP_base'
postpr_m_i = postpr_m[invname]

fig,ax = plt.subplots(nsite,1,figsize=(9,4*nsite))

for isite,site in enumerate(sites):
    
    ax[isite].set_title("Difference %s"%site)
    co2_true = postpr_m_i.obs_all['true'][site]['co2']
    for ilab,label in enumerate(['prior','posterior']):
        dates, co2 = postpr_m_i.obs_all[label][site]['dates'], postpr_m_i.obs_all[label][site]['co2']
        
        diff = co2-co2_true
        
        bias = np.mean(diff)
        rms = np.sqrt(np.mean(diff**2))
        
        ax[isite].hist(diff, facecolor=colors[ilab], bins=np.linspace(-8,8), label=label, orientation='vertical', alpha=0.5)
        
        ax[isite].text(0.05,0.75-0.25*ilab, 'Bias %2.2f ppm\nRMS %2.2f ppm'%(bias,rms), 
                         fontsize=18, color=colors[ilab], transform=ax[isite].transAxes)
    
    ax[isite].legend(loc='upper right')
        
[a.set_xlabel("CO$_2$ [ppm]") for a in ax.flatten()]    
    
fig.tight_layout()
fig.savefig("%s/obs_hist_%s.png"%(path_figs,invname))

#%%

# Comparing the different inversions
fig,ax = plt.subplots(1,1,figsize=(10,5))

ax.set_ylabel("RMS CO$_2$ obs [ppm]")

site = 'AUT-MKH'
for i,invname in enumerate(inversion_names):
    postpr_m_i = postpr_m[invname]
    co2_true = postpr_m_i.obs_all['true'][site]['co2']
    
    for ilab,label in enumerate(['prior','posterior']):
        dates, co2 = postpr_m_i.obs_all[label][site]['dates'], postpr_m_i.obs_all[label][site]['co2']
        
        diff = co2-co2_true
        
        bias = np.mean(diff)
        rms = np.sqrt(np.mean(diff**2))
        
        bwidth = 0.2
        ax.bar(bwidth+i+0.5*ilab, rms, width=bwidth, color=colors[ilab])
    

ax.set_xlim(0,len(inversion_names))

#%%

pp = postpr_m['baseAKLNWP_odiac']
dates = pp.obs_all['prior']['AUT-MKH']['dates']
hours = np.array([d.hour for d in dates])

num = np.arange(24)
for h in range(24):
    num[h] = np.sum(hours==h)
    
nday = (dates.max()-dates.min()).days
plt.figure()
plt.plot(num/nday*100)
plt.ylim(0,100)

#%%


senstest_labels = {'baseAKLNWP_base':'Base',
                    'baseAKLNWP_mahk_bias':'Mahk+20%',
                    'baseAKLNWP_odiac':'ODIAC',
                    'baseAKLNWP_100m':'100m\nsample\nlayer', 
                    'baseNZCSM_base':'NZCSM\nmeteo',  
                    'baseAKLNWP_only_AUT':'Only\nAUT+MKH',
                    'baseAKLNWP_afternoon':'Afternoon-only',
                    'baseAKLNWP_biomebgc':'BiomeBGC',
                    'baseAKLNWP_biomebgc_afternoon':"BiomeBGC afternoon"
                   }


invnames = ['baseAKLNWP_base', 'baseAKLNWP_afternoon', 'baseAKLNWP_100m', 
            'baseNZCSM_base',  'baseAKLNWP_only_AUT', 'baseAKLNWP_odiac',]

colors = np.array(sns.color_palette('Paired'))
colors_rms = colors[[0,1]]
colors_bias = colors[[2,3]]

sites_all = np.array(list(postpr_m['baseAKLNWP_base'].obs_all['prior'].keys()))
nsite = len(sites_all)

sns.set_context("talk",font_scale=1.2)

fig,ax = plt.subplots(1,nsite,figsize=(5*nsite,10))



for iinv,invname in enumerate(invnames):
    pp = postpr_m[invname]
    
    sites_pp = pp.obs_all['prior'].keys()
    
    
    for site in sites_pp:
        isite = np.where(sites_all==site)[0][0]
        
        title = site.split('-')
        title = '%s $-$ %s'%(title[0],title[1])
        ax[isite].set_title(title)
        
        
        dates = pp.obs_all['prior'][site]['dates']
        co2_true = pp.obs_all['true'][site]['co2']
        for ilab,label in enumerate(['prior','posterior']):
            dates   = pp.obs_all[label][site]['dates']
            co2     = pp.obs_all[label][site]['co2']
            co2_err = pp.obs_all[label][site]['co2_err']
            print(co2_err)
            
            diff = (co2-co2_true)
            
            bias = np.mean(diff)
            rms = np.sqrt(np.mean(diff**2))
            
            
            if iinv==0:
                rms_label = 'RMS %s'%label
                bias_label = 'Bias %s'%label
            else: 
                rms_label = bias_label = None
            
            y = 5 - iinv*0.6 - ilab*0.22
            ax[isite].barh(y, bias, height=0.1, edgecolor='k', facecolor=colors_bias[ilab], label=bias_label)
            ax[isite].barh(y-0.1, rms, height=0.1, edgecolor='k', facecolor=colors_rms[ilab],label=rms_label)
    
        ax[isite].set_xlabel("CO$_2$ (SD)")
    

ax[0].set_yticks([5-0.6*i - 0.17 for i in range(len(invnames))]) 
ax[0].set_yticklabels([senstest_labels[inv] for inv in invnames])

[a.set_yticks([]) for a in ax[1:].flatten()]

ax[-1].legend(bbox_to_anchor=(1.1,0.98))
fig.tight_layout()
fig.savefig("%s/obs_stats_all_inversions_sd.png"%path_figs)










