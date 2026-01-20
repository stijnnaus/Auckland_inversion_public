#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:22:28 2025

@author: nauss
"""

from datetime import timedelta, datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import functions_stijn as st
import seaborn as sns
from base_paths import path_figs
path_figs = '%s/vprm_new_xt/'%path_figs
from postpostprocess_multi import Postpostprocess_multi
from postprocessing_truth import Postprocess_truth

sns.set_context('talk')


#%%


inversion_names = 'baseAKLNWP_100m','baseAKLNWP_afternoon',
startdate = datetime(2022,1,1)
enddate = datetime(2022,5,2)
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

sns.set_context('talk',font_scale=1.3)
colors = sns.color_palette('Set2')

labels_full = {'baseAKLNWP_100m':'Base', 'baseAKLNWP_afternoon':'Afternoon-only'}

hours = np.arange(0,24,3)+1.5

fig,ax = plt.subplots(2,1,figsize=(15,10))

ls = '-','--'
for iinversion,inversion in enumerate(['baseAKLNWP_100m','baseAKLNWP_afternoon']):
    for iinventory,inventory in enumerate(['MahuikaAuckland','UrbanVPRM']):
        for ilab,label in enumerate(['prior','posterior']):
            
            unc = np.sqrt(postpr_m[inversion].unc_all_rel[label]['per_domain']['diurnal'][inventory][inventory][:,0])
            unc = unc.reshape(5,8).mean(axis=0)
            
            ax[iinventory].plot(np.arange(0,24,3), unc*100, color=colors[ilab], linestyle=ls[iinversion], label='%s %s'%(labels_full[inversion], label))
        ax[iinventory].set_ylabel("Uncertainty [%]")
        ax[iinventory].set_title(inventory)
ax[1].set_xlabel("Hour in day")
ax[1].legend(bbox_to_anchor=(1.05,1.3), ncol=1)
plt.tight_layout()
plt.savefig('%s/diurnal_cycle_uncertainty_rel.png'%path_figs, bbox_inches='tight')












