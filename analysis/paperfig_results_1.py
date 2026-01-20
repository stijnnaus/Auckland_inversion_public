#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:00:59 2024

@author: nauss
"""

from datetime import timedelta, datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import functions_stijn as st
import seaborn as sns
from base_paths import path_figs
path_figs = '%s/osse_longterm/'%path_figs
from postpostprocess_multi import Postpostprocess_multi
from postprocessing_truth import Postprocess_truth

sns.set_context('talk')


#%%


inversion_names = 'baseAKLNWP_base','baseNZCSM_base','baseAKLNWP_odiac'
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

true_label = postpr_m[invname].rc.osse_truth_label

startdate = postpr_m[invname].dates_all['daily'][0] + timedelta(seconds=25*3600)
enddate   = postpr_m[invname].dates_all['daily'][-1]

pp_true = Postprocess_truth(true_label, startdate=startdate, enddate=enddate)
pp_true.run_standard_postprocessing()


#%%

def get_rms_errinv(pp, label, prior_inv):
    ee = pp.emis_all[label]['per_domain']['daily'][prior_inv][:,0]/1e6
    
    rms = np.sqrt(np.mean((ee-etru)**2))
    err_inv = np.sqrt(np.mean(postpr.unc_all_abs[label]['per_domain']['daily'][prior_inv][prior_inv][:,0]))/1e6
    return rms, err_inv

invname = 'baseAKLNWP_base'
postpr = postpr_m[invname]

sns.set_context("talk",font_scale=1.2)

fig = plt.figure(figsize=(20,10))

gs = fig.add_gridspec(2, 2,  width_ratios=(2, 1), height_ratios=(1, 1),
                  left=0.1, right=0.9, bottom=0.1, top=0.9,
                  wspace=0.2, hspace=0.12)


colors = sns.color_palette('Set2')

ax0a = fig.add_subplot(gs[0, 0])
ax0b = fig.add_subplot(gs[1, 0])
ax0 = [ax0a, ax0b]

plotlabels = ['Prior','Posterior','True']

priors_tru = ['MahuikaAuckland','UrbanVPRM']
if 'odiac' in invname:
    priors_inv = ['ODIAC','UrbanVPRM']
else:
    priors_inv = priors_tru
    
etrue_both = {}
for i,(prior_tru) in enumerate( priors_tru ):
    
    etru = pp_true.emis_agg['per_domain']['daily'][prior_tru][:,0]/1e6
    etru = etru.reshape(-1,7).sum(axis=1)
    
    etrue_both[prior_tru] = etru
print('r=%2.2f'%(np.corrcoef(etrue_both['MahuikaAuckland'],etrue_both['UrbanVPRM'])[0][1]))
    
    
lw = 3.5
for i,(prior_tru,prior_inv) in enumerate( zip(priors_tru, priors_inv) ):
    ax = ax0[i]
    
    dates = postpr.dates_all['daily']
    
    etru = pp_true.emis_agg['per_domain']['daily'][prior_tru][:,0]/1e6
    etru = etru.reshape(-1,7).sum(axis=1)
    
    for j,label in enumerate(['prior','posterior']):
        emi = postpr.emis_all[label]['per_domain']['daily'][prior_inv][:,0]/1e6
        emi = emi.reshape(-1,7).sum(axis=1)
        ax.plot(dates[3::7], emi, color=colors[j], lw=lw, label=plotlabels[j])
        
        rcoef = np.corrcoef(emi, etru)[0][1]
        
        y0_txt = 0.27 if prior_inv=='MahuikaAuckland' else 0.3
        x0_txt = 0.05 if prior_inv=='MahuikaAuckland' else 0.42
        ax.text(x0_txt, y0_txt-j*0.13,'r = %2.2f'%rcoef,color=colors[j],va='top',ha='left',transform=ax.transAxes,fontsize=28,fontweight='bold')
        
    ax.plot(dates[3::7], etru, color=colors[2], lw=lw, label=plotlabels[2])
    

    ax.set_ylabel("Weekly flux\n[kton/week]")

ax0a.legend(loc='lower right',ncol=3)
ax0a.set_xticks([])
# ax0a.set_ylim(0,30)
# ax0a.set_yticks(np.arange(0,31,5))

xticks = [datetime(2022,1,15) + timedelta(days=60)*i for i in range(6)]
st.adjust_xticks_dates_to_m(ax0b, xticks=xticks, month_fmt='%b')

ax1a = fig.add_subplot(gs[0, 1])
ax1b = fig.add_subplot(gs[1, 1])
ax1 = [ax1a, ax1b]

# Errors representing spread between inversions
yerrs = np.array([[0.0, 0.0], [0.0, 0]])

for i,(prior_tru,prior_inv) in enumerate( zip(priors_tru, priors_inv) ):
    ax = ax1[i]
    
    etru = pp_true.emis_agg['per_domain']['daily'][prior_tru][:,0]/1e6
    rms_both, err_both = {}, {}
    for j,label in enumerate(['prior','posterior']):
        rms, err_inv = get_rms_errinv(postpr, label, prior_inv)
        rms_both[label] = rms
        err_both[label] = err_inv
        
        spacing = 0.25
        ax.bar(spacing*j - spacing/2    , rms,     width=0.25, edgecolor='k', yerr=yerrs[i,j]*0.9, color=colors[j], label=plotlabels[j])
        ax.bar(1 + spacing*j - spacing/2 , err_inv,     width=0.25, edgecolor='k',yerr=yerrs[i,j], color=colors[j])
    
    print("RMS     prior %2.2f ; poste %2.2f ; red %3.3f"%(rms_both['prior'],rms_both['posterior'],(rms_both['prior']-rms_both['posterior'])/rms_both['prior']))
    print("Err_inv prior %2.2f ; poste %2.2f ; red %3.3f"%(err_both['prior'],err_both['posterior'],(err_both['prior']-err_both['posterior'])/err_both['prior']))
        
    if i == 0:
        ax.legend(loc='upper right', ncol=3)
    
    ax.set_ylabel("Error daily flux\n[kton/day]")
    # ax.yaxis.set_label_position("right")
    # ax.yaxis.tick_right()
    ax.set_xlim(-0.5,1.5)

props = dict(boxstyle='square,pad=0.4', facecolor='white', edgecolor='k', alpha=0.9)
for ax,name in zip([ax0a,ax0b],['Anthropogenic','Biosphere']):
    ax0a.text(0.983,0.874,'Anthropogenic',fontsize=28, horizontalalignment='right', transform=ax0a.transAxes,bbox=props)
    ax0b.text(0.983,0.874,'Biosphere',fontsize=28, horizontalalignment='right', transform=ax0b.transAxes,bbox=props)

ax1a.set_xticks([])
ax1b.set_xticks([0,1])
ax1b.set_xticklabels(["RMS with\ntruth","Inversion-reported\nerror"])

ax1a.set_ylim(0, 8)
ax1a.set_yticks(np.arange(0,9,2))

# Percentage axis
for i,prior in enumerate(priors_tru):
    ax = ax1[i]
    axt = ax.twinx()
    etru = pp_true.emis_agg['per_domain']['daily'][prior][:,0]/1e6
    emi_av = np.mean(etru)
    
    axt.set_ylim(0, 100*ax.get_ylim()[1]/np.abs(emi_av))
    axt.set_ylabel("Error daily flux\n[%]")

ax1b.set_yticks(np.arange(0,13,4))


# Panel labels
bbox=dict(boxstyle='square', fc="w", ec="k", alpha=0.9)
zorder=12
fontsize =24
ax0a.text(0.0102, 0.975, 'A', ha='left', va='top', fontsize=fontsize, zorder=zorder, transform=ax0a.transAxes, bbox=bbox)
ax0b.text(0.0102, 0.975, 'B', ha='left', va='top', fontsize=fontsize, zorder=zorder, transform=ax0b.transAxes, bbox=bbox)
ax1a.text(0.0187, 0.975, 'C', ha='left', va='top', fontsize=fontsize, zorder=zorder, transform=ax1a.transAxes, bbox=bbox)
ax1b.text(0.0187, 0.975, 'D', ha='left', va='top', fontsize=fontsize, zorder=zorder, transform=ax1b.transAxes, bbox=bbox)
    

plt.savefig('%s/timeseries_rms_%s_weekly.png'%(path_figs, invname), dpi=300)











