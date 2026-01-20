#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper figure in which I compare annual emission totals from all emission tests.

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
import inversion as invh
from postprocessing_truth import Postprocess_truth
from postpostprocess_multi import Postpostprocess_multi
import time

from matplotlib.gridspec import GridSpec
    
#%%

t0 = time.time()

inversion_names = ['baseAKLNWP_base', 'baseAKLNWP_odiac', 'baseAKLNWP_100m', 'baseNZCSM_base',  
                    'baseAKLNWP_only_AUT', 'baseAKLNWP_afternoon', 'baseAKLNWP_double_pri_err',
                    'baseAKLNWP_vprm_bias', 'baseAKLNWP_vprm_bias_aft','baseAKLNWP_biomebgc']
# inversion_names= [      'baseAKLNWP_base',              'baseAKLNWP_mahk_bias']

# inversion_names = ['baseAKLNWP_biomebgc']

# inversion_names = ['baseAKLNWP_base','baseAKLNWP_afternoon',]#'baseAKLNWP_100m']
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

print('%2.2f s'%(time.time()-t0))

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

# In order I want the bars to appear
tests = ['baseAKLNWP_base', 'baseAKLNWP_afternoon', 'baseAKLNWP_100m', 
         'baseNZCSM_base',  'baseAKLNWP_only_AUT', 'baseAKLNWP_odiac',]

axlim = 6500

axlim = 1700

all_days = np.arange(364)
winter = np.arange(151,243)
summer = np.append(np.arange(0,62), np.arange(334,364))

days = summer

sns.set_context('talk', font_scale=1.4)

colors = sns.color_palette('Paired')

fig = plt.figure(figsize=(15,20))

gs = GridSpec(100, 100)

axl_top = fig.add_subplot(gs[:30,:40])
axl_bot = fig.add_subplot(gs[35:,:40])
axr_top = fig.add_subplot(gs[:30,60:])
axr_bot = fig.add_subplot(gs[35:,60:])
axes = np.array([[axl_top, axl_bot], [axr_top, axr_bot]])

# Top axes
for ax in axes[:,0]:
    ax.spines[['bottom']].set_visible(False)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ntick_top = 3
    ax.set_ylim([0,ntick_top])
    ax.set_yticks(np.arange(0.5,0.5+ntick_top))
    ax.xaxis.set_label_position('top') 
axl_top.set_yticklabels(["True", "ODIAC", "Mahuika-\nAuckland"])
axr_top.set_yticklabels(["True", "", "UrbanVPRM"])
# axl_top.title.set_text("Anthropogenic flux\n")
# axr_top.title.set_text("Net Ecosystem Exchange\n")

# Bottom axes
for ax in axes[:,1]:
    ax.spines[['top']].set_visible(False)
    ntick_bot = len(tests)
    ax.set_ylim([0,ntick_bot])
    ax.set_yticks(np.arange(0.5,0.5+ntick_bot))
    ax.set_yticklabels([senstest_labels[k] for k in tests][::-1])

# Left axes
for ax in axes[0]:
    ax.set_xlabel("Anthropogenic emissions\n(kton/year)")
    ax.spines[['right']].set_visible(False)
    ax.set_xlim([0,axlim])

# Right axes
for ax in axes[1]:
    ax.set_xlabel("Net Ecosystem Exchange\n(kton/year)")
    ax.spines[['left']].set_visible(False)
    ax.tick_params(left=False, labelleft=False, right=True, labelright=True)
    ax.set_xlim([-axlim,0])
    

def plot_bar(ax, y, pp, inventory, color, days, height=0.6, plabel='true'):
    if plabel == 'true':
        emis = pp.emis_agg
    else:
        emis = pp.emis_all[plabel]
        unc  = pp.unc_all_abs[plabel]
    
    etot = emis['per_domain']['daily'][inventory][:,0][days].sum()/1e6
        
        
    ax.barh(y, etot, height=height, edgecolor='k', facecolor=color)
    
    if not plabel=='true':
        # Calculate aggregated error (errors are still per inversion time-slice)
        tsteps = pp.dates_all['daily']
        unc_per_step = np.sqrt(unc['per_domain']['daily'][inventory][inventory][:,0])
        corlen = pp.rc.prior_errors[inventory]['L_temp_long']
        
        
        unc_tot = invh.add_up_correlated_uncertainties(tsteps[days], unc_per_step[days], timedelta(days=corlen)) / 1e6
        ax.errorbar(etot, y, xerr=unc_tot, capsize=4, fmt="", color='black')
        
        etot_abs = np.abs(emis['per_domain']['daily'][inventory][:,0][days]).sum()/1e6
        print(inventory, plabel, '%2.2f'%(100*unc_tot/etot_abs), '%2.2f'%etot_abs)
    
    
# Priors / truth
plot_bar(axl_top, 0.5, pp_true, 'MahuikaAuckland', color='k', days=days, plabel='true')
plot_bar(axr_top, 0.5, pp_true, 'UrbanVPRM', color='k', days=days, plabel='true')
plot_bar(axl_top, 2.5, postpr_m['baseAKLNWP_base'], 'MahuikaAuckland', days=days, color=colors[0], plabel='prior')
plot_bar(axr_top, 2.5, postpr_m['baseAKLNWP_base'], 'UrbanVPRM', days=days, color=colors[2], plabel='prior')
# plot_bar(axr_top, 1.5, postpr_m['baseAKLNWP_biomebgc'], 'BiomeBGC', color=colors[2], plabel='prior')
plot_bar(axl_top, 1.5, postpr_m['baseAKLNWP_odiac'], 'ODIAC', days=days, color=colors[0], plabel='prior')

axl = [axl_bot,axl_top]
axr = [axr_bot,axr_top]
for ax,prior in zip([axl_bot,axr_bot], ['MahuikaAuckland','UrbanVPRM']):
    etru = pp_true.emis_agg['per_domain']['daily'][prior][:,0][days].sum()/1e6
    
    ax.plot([etru,etru], [ax.get_ylim()[0], 9.26], 'k--', clip_on=False, zorder=20)
    # ax.plot([etru,etru], [-10, ax.get_ylim()[1]], 'k--', clip_on=False, zorder=20)

# Posterior
nbar = len(tests)
for i,test in enumerate(tests):
    pos = nbar-0.5 - i
    if 'odiac' in test.lower():
        plot_bar(axl_bot, pos, postpr_m[test], 'ODIAC', color=colors[1], days=days, plabel='posterior')
    else:
        plot_bar(axl_bot, pos, postpr_m[test], 'MahuikaAuckland', color=colors[1], days=days, plabel='posterior')
        
    if 'biomebgc' in test.lower():
        plot_bar(axr_bot, pos, postpr_m[test], 'BiomeBGC', color=colors[3], days=days, plabel='posterior')
    else:
        plot_bar(axr_bot, pos, postpr_m[test], 'UrbanVPRM', color=colors[3], days=days, plabel='posterior')


plt.savefig("%s/fig2_emitot_bars_summer"%path_figs, bbox_inches='tight')

#%%

# Summer versus winter


axlim = 2000

all_days = np.arange(364)
winter = np.arange(151,243)
summer = np.append(np.arange(0,62), np.arange(334,364))

colors = sns.color_palette('Paired')

fig = plt.figure(figsize=(15,20))

gs = GridSpec(100, 100)

axl_top = fig.add_subplot(gs[:30,:40])
axl_bot = fig.add_subplot(gs[35:,:40])
axr_top = fig.add_subplot(gs[:30,60:])
axr_bot = fig.add_subplot(gs[35:,60:])
axes = np.array([[axl_top, axl_bot], [axr_top, axr_bot]])

# Top axes
for ax in axes[:,0]:
    ax.spines[['bottom']].set_visible(False)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ntick_top = 3
    ax.set_ylim([0,ntick_top])
    ax.set_yticks(np.arange(0.5,0.5+ntick_top))
    ax.xaxis.set_label_position('top') 
axl_top.set_yticklabels(["True", "ODIAC", "Mahuika-\nAuckland"])
axr_top.set_yticklabels(["True", "", "UrbanVPRM"])
# axl_top.title.set_text("Anthropogenic flux\n")
# axr_top.title.set_text("Net Ecosystem Exchange\n")

# Bottom axes
for ax in axes[:,1]:
    ax.spines[['top']].set_visible(False)
    ntick_bot = len(tests)
    ax.set_ylim([0,ntick_bot])
    ax.set_yticks(np.arange(0.5,0.5+ntick_bot))
    ax.set_yticklabels([senstest_labels[k] for k in tests][::-1])

# Left axes
for ax in axes[0]:
    ax.set_xlabel("Anthropogenic emissions\n(kton/year)")
    ax.spines[['right']].set_visible(False)
    ax.set_xlim([0,axlim])

# Right axes
for ax in axes[1]:
    ax.set_xlabel("Net Ecosystem Exchange\n(kton/year)")
    ax.spines[['left']].set_visible(False)
    ax.tick_params(left=False, labelleft=False, right=True, labelright=True)
    ax.set_xlim([-axlim,0])
    
for i,days in enumerate([winter,summer]):
        
    # Priors / truth
    plot_bar(axl_top, 0.65-i*0.3, pp_true, 'MahuikaAuckland', color='k', height=0.3, days=days, plabel='true')
    plot_bar(axr_top, 0.65-i*0.3, pp_true, 'UrbanVPRM', color='k', days=days, height=0.3, plabel='true')
    plot_bar(axl_top, 2.65-i*0.3, postpr_m['baseAKLNWP_base'], 'MahuikaAuckland', days=days, height=0.3, color=colors[0], plabel='prior')
    plot_bar(axr_top, 2.65-i*0.3, postpr_m['baseAKLNWP_base'], 'UrbanVPRM', height=0.3, days=days, color=colors[2], plabel='prior')
    # plot_bar(axr_top, 1.5, postpr_m['baseAKLNWP_biomebgc'], 'BiomeBGC', color=colors[2], plabel='prior')
    plot_bar(axl_top, 1.65-i*0.3, postpr_m['baseAKLNWP_odiac'], 'ODIAC', days=days, height=0.3, color=colors[0], plabel='prior')
    
    # axl = [axl_bot,axl_top]
    # axr = [axr_bot,axr_top]
    # for ax,prior in zip([axl_bot,axr_bot], ['MahuikaAuckland','UrbanVPRM']):
    #     etru = pp_true.emis_agg['per_domain']['daily'][prior][:,0][days].sum()/1e6
        
    #     ax.plot([etru,etru], [ax.get_ylim()[0], 9.26], 'k--', clip_on=False, zorder=20)
    #     # ax.plot([etru,etru], [-10, ax.get_ylim()[1]], 'k--', clip_on=False, zorder=20)
    
    # Posterior
    nbar = len(tests)
    for j,test in enumerate(tests):
        pos = nbar-0.35 - j - i*0.3
        if 'odiac' in test.lower():
            plot_bar(axl_bot, pos, postpr_m[test], 'ODIAC', color=colors[1], height=0.3, days=days, plabel='posterior')
        else:
            plot_bar(axl_bot, pos, postpr_m[test], 'MahuikaAuckland', color=colors[1], height=0.3, days=days, plabel='posterior')
            
        if 'biomebgc' in test.lower():
            plot_bar(axr_bot, pos, postpr_m[test], 'BiomeBGC', color=colors[3], height=0.3, days=days, plabel='posterior')
        else:
            plot_bar(axr_bot, pos, postpr_m[test], 'UrbanVPRM', color=colors[3], height=0.3, days=days, plabel='posterior')


plt.savefig("%s/sfig_emitot_bars_seasons"%path_figs, bbox_inches='tight')

#%%

# Error in bio/ant separate versus in net flux
inv = 'baseNZCSM_base'
emi_pos_bio = postpr_m[inv].emis_all['posterior']['per_domain']['one_timestep']['UrbanVPRM'][:,0].sum() / 1e6
emi_pos_ant = postpr_m[inv].emis_all['posterior']['per_domain']['one_timestep']['MahuikaAuckland'][:,0].sum() / 1e6
emi_pos_tot = emi_pos_bio + emi_pos_ant

emi_tru_bio = pp_true.emis_agg['per_domain']['one_timestep']['UrbanVPRM'][:,0].sum() / 1e6
emi_tru_ant = pp_true.emis_agg['per_domain']['one_timestep']['MahuikaAuckland'][:,0].sum() / 1e6
emi_tru_tot = np.abs(emi_tru_bio) + np.abs(emi_tru_ant)

err_abs_bio = np.abs(emi_pos_bio-emi_tru_bio)
err_abs_ant = np.abs(emi_pos_ant-emi_tru_ant)
err_abs_tot = np.abs(emi_pos_tot-emi_tru_tot)

print('Bio: %2.2f kt/year; Ant: %2.2f kt/year; Total: %2.2f kt/year'%(err_abs_bio, err_abs_ant, err_abs_tot))
print('Bio: %2.2f %%; Ant: %2.2f %%; Total: %2.2f %%'%(100*err_abs_bio/emi_tru_bio, 100*err_abs_ant/emi_tru_ant, 100*err_abs_tot/emi_tru_tot))


#%%

# Quick look at total uncertainty in emissions
pp = postpr_m['baseAKLNWP_base']
for inventory in ['MahuikaAuckland','UrbanVPRM']:
    
    unc = np.sqrt(pp.unc_all_abs['prior']['per_domain']['one_timestep'][inventory][inventory][:,0])
    tsteps = pp.dates_all['one_timestep']
    corlen = pp.rc.prior_errors[inventory]['L_temp_long']
    
    unc_tot = invh.add_up_correlated_uncertainties(tsteps, unc, corlen)
    emi_tot = np.abs(pp.emis_all['prior']['per_domain']['daily'][inventory][:,0].sum())
    
    print('%s %2.2f %%'%(inventory, 100*unc_tot/emi_tot))


#%%

# Compare total uncertainty of two inversions

inv2 = 'baseAKLNWP_only_AUT'
inv1 = 'baseAKLNWP_base'

for invi in [inv1,inv2]:
    pp = postpr_m[invi]
    for inventory in ['MahuikaAuckland','UrbanVPRM']:
        
        unc = np.sqrt(pp.unc_all_abs['posterior']['per_domain']['one_timestep'][inventory][inventory][:,0])
        tsteps = pp.dates_all['one_timestep']
        corlen = pp.rc.prior_errors[inventory]['L_temp_long']
        
        unc_tot = invh.add_up_correlated_uncertainties(tsteps, unc, corlen)
        emi_tot = np.abs(pp.emis_all['prior']['per_domain']['daily'][inventory][:,0].sum())
        
        print('%s %s %2.2f %%'%(invi, inventory, 100*unc_tot/emi_tot))












