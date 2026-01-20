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
import inversion as invh
from base_paths import path_figs
path_figs = '%s/osse_longterm/'%path_figs
from postpostprocess_multi import Postpostprocess_multi
from postprocessing_truth import Postprocess_truth

sns.set_context('talk')


#%%


inversion_names = 'baseAKLNWP_base','baseAKLNWP_afternoon','baseAKLNWP_odiac','baseAKLNWP_vprm_bias','baseAKLNWP_vprm_bias_aft'
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
import time

t0 = time.time()

true_label = postpr_m[invname].rc.osse_truth_label

startdate = postpr_m[invname].dates_all['daily'][0] + timedelta(seconds=25*3600)
enddate   = postpr_m[invname].dates_all['daily'][-1]

pp_true = Postprocess_truth(true_label, startdate=startdate, enddate=enddate)
pp_true.run_standard_postprocessing()

print('%2.2s'%(time.time()-t0))

#%%

def get_diurnal_cycle_emis(pp, label, prior, truth=False):
    
    if truth:
        efull = pp.emis_agg['per_domain']['per_timestep'][prior][:,0]
        
    else:
        efull = pp.emis_all[label]['per_domain']['per_timestep'][prior][:,0]
        
    ediurn = efull.reshape(-1,8).mean(axis=0)
        
    return ediurn/ 1e6

def calc_rms_with_truth(pp, pp_true, label, prior):
    
    emi_inv = pp.emis_all[label]['per_domain']['per_timestep'][prior][:,0]
    if prior=='ODIAC' or prior=='MahuikaAuckland':
        emi_tru = pp_true.emis_agg['per_domain']['per_timestep']['MahuikaAuckland'][:,0]
    else:
        emi_tru = pp_true.emis_agg['per_domain']['per_timestep']['UrbanVPRM'][:,0]
    
    print(label,prior,emi_inv.shape,emi_tru.shape)
    diff = (emi_inv - emi_tru).reshape(-1,8) / 1e6
    rms  = np.sqrt( np.mean( diff**2, axis=0 ) ) 
    diff = np.mean(diff,axis=0)
    
    return rms,diff

def get_Bunc_diurnal(pp, label, prior):
    '''
    Get diurnal cycle of prior or posterior uncertainties. Note that after postprocessing
    I still have a diurnal cycle per inversion window (= 4 weeks), so I need to add those
    together taking into account temporal correlations.
    '''
    
    unc = np.sqrt(pp.unc_all_abs[label]['per_domain']['diurnal'][prior][prior][:,0]) # kg/inversion window
    unc = unc.reshape(-1,8) # Separate inversion window from 3-hour window
    unc = unc / (1e6*28*8) # kton/three-hour
    
    print(prior,unc.mean())
    
    unc_agg = np.zeros(8)
    tsteps = pp.startdates_all
    corlen = timedelta(days=pp.inversion_example.rc.prior_errors[prior]['L_temp_long'])
    for i in range(8):
        unc_agg[i] = invh.add_up_correlated_uncertainties(tsteps, unc[:,i], corlen)
        
    return unc_agg


pfull = postpr_m['baseAKLNWP_vprm_bias']
paft  = postpr_m['baseAKLNWP_vprm_bias_aft']

colors = sns.color_palette('Set2')
cpri = colors[0]
cpos = colors[1]
ctru = 'k'
lw = 4.

sns.set_context('talk',font_scale=1.2)

hours = 1.5 + np.arange(0,24,3)

fig,ax = plt.subplots(2,3,figsize=(30,10))
priornames = {'MahuikaAuckland':'anthropogenic', 'UrbanVPRM':'biosphere'}

for i,prior in enumerate(['MahuikaAuckland','UrbanVPRM']):
    
    ax[i,0].set_title('Diurnal cycle %s'%priornames[prior], fontsize=28)
    
    emi_pri       = get_diurnal_cycle_emis(pfull,    'prior',     prior, truth=False)
    emi_pos_night = get_diurnal_cycle_emis(pfull,    'posterior', prior, truth=False)
    emi_pos_aft   = get_diurnal_cycle_emis(paft,     'posterior', prior, truth=False)
    
    if prior=='ODIAC' or prior=='MahuikaAuckland':
        emi_tru       = get_diurnal_cycle_emis(pp_true,  'posterior', 'MahuikaAuckland', truth=True)
    else:
        emi_tru       = get_diurnal_cycle_emis(pp_true,  'posterior', 'UrbanVPRM', truth=True)
    
    ax[i,0].plot(hours, emi_pri, color=cpri, linewidth=lw,       label='Prior')
    ax[i,0].plot(hours, emi_pos_night, color=cpos, linewidth=lw,   label='Posterior, full')
    ax[i,0].plot(hours, emi_pos_aft, '--', color=cpos, linewidth=lw, label='Posterior, afternoon-only')
    ax[i,0].plot(hours, emi_tru, color=ctru, linewidth=lw,       label='Truth')
    ax[i,0].set_ylabel('Three-hourly flux \n[kton/three-hours]')
    

    # ax[i,1].set_title('Difference with truth %s'%priornames[prior])
    ax[i,1].plot(hours, calc_rms_with_truth(pfull, pp_true, 'prior',     prior)[1], color=cpri, linewidth=lw, label='Prior')
    ax[i,1].plot(hours, calc_rms_with_truth(pfull, pp_true, 'posterior', prior)[1], color=cpos, linewidth=lw, label='Posterior, full')
    ax[i,1].plot(hours, calc_rms_with_truth(paft , pp_true, 'posterior', prior)[1], '--', color=cpos, linewidth=lw, label='Posterior, afternoon-only')
    ax[i,1].set_xlim(ax[i,1].get_xlim())
    ax[i,1].plot(ax[i,1].get_xlim(), [0,0], color='gray',alpha=0.6)
    if prior=='MahuikaAuckland':
        ax[i,0].set_ylim(0, ax[i,0].get_ylim()[1])
    ax[i,1].set_ylabel("Difference with truth\n[kton/three-hours]")
    
    # ax[i,2].set_title('Inversion-reported error')
    ax[i,2].plot(hours, get_Bunc_diurnal(pfull, 'prior',     prior), color=cpri, linewidth=lw, label='Prior')
    ax[i,2].plot(hours, get_Bunc_diurnal(pfull, 'posterior', prior), color=cpos, linewidth=lw, label='Posterior, full')
    ax[i,2].plot(hours, get_Bunc_diurnal(paft , 'posterior', prior), '--', color=cpos, linewidth=lw, label='Posterior, afternoon-only')
    ax[i,2].set_ylim(0, ax[i,2].get_ylim()[1])
    ax[i,2].set_ylabel("Inversion-reported error\n[kton/three-hours]")
    
    if i == 0 :
        ax[i,0].legend(loc='best')
        ax[i,1].legend(loc='best')

ax[0,1].set_ylim(-0.15,0.53)
[a.set_xlabel("Hour in day") for a in ax[1]]
fig.tight_layout()

# Panel labels
bbox=dict(boxstyle='square', fc="w", ec="k", alpha=0.9)
for a,panel_label in zip(ax.flatten(), ['A','B','C','D','E','F']):
    a.text(0.986, 0.97, panel_label, va='top', ha='right', fontsize=24, zorder=12, transform=a.transAxes, bbox=bbox)
    
             

fig.savefig('%s/diurnal_cycle.png'%path_figs, dpi=320)



#%%

prior = 'UrbanVPRM'

hours = [0,1,7]
# hours = [3,4,5]

diff_pri = calc_rms_with_truth(pfull, pp_true, 'prior',     prior)[1]
diff_full = calc_rms_with_truth(pfull, pp_true, 'posterior',     prior)[1]
diff_aft = calc_rms_with_truth(paft, pp_true, 'posterior',     prior)[1]

emi_tru       = get_diurnal_cycle_emis(postpr_m['baseAKLNWP_base'],  'prior', prior, truth=False)

print('Prior : %2.2f%%'%(100*np.sum(np.abs(diff_pri)[hours]) / np.sum(np.abs(emi_tru)[hours] ) ))
print('Poste full : %2.2f%%'%(100*np.sum(np.abs(diff_full)[hours]) / np.sum(np.abs(emi_tru)[hours] ) ))
print('Poste aft : %2.2f%%'%(100*np.sum(np.abs(diff_aft)[hours]) / np.sum(np.abs(emi_tru)[hours] ) ))





#%%

efull = pfull.emis_agg['per_domain']['per_timestep'][prior][:,0].reshape(-1,8)


















    