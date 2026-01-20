#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checking whether scaling the biosphere error to GPP and Respiration make sense
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime,timedelta
import seaborn as sns
from construct_B import ConstructorB
import functions_stijn as st
from construct_x import ConstructorX

sns.set_context('talk')

invname = "baseAKLNWP_truth2"
kwargs = {'date_start':['direct',datetime(2022,2,1)+timedelta(seconds=3600*25)],
          'date_end':['direct',datetime(2022,3,1)]}

cx = ConstructorX(invname, kwargs)
cx.construct_x()

cB = ConstructorB(invname, kwargs)
cB.get_emissions_on_inversion_grid()
scalars = cB.get_Bmatrix_error_scalings([])

prior = 'UrbanVPRM'
NEE = cB.get_emis_invgrid(prior, 'NEE')
GEE = cB.get_emis_invgrid(prior, 'GEE')
Re = cB.get_emis_invgrid(prior,  'Re')

#%%

ttot = cx.get_timesteps_opt_full()
nt = len(ttot)

fig,ax = plt.subplots(1,1,figsize=(12,5))

ax.set_ylabel("Total Auckland flux\n[kton CO$_2$ / 3 hour]")
for v,l in np.transpose([[NEE,GEE,Re],['NEE','GPP','Resp']]):
    vr = v.reshape(nt,-1).sum(axis=-1)
    print(np.sum(vr/1e6)/29*365)
    ax.plot(ttot, vr/1e6, label=l)
    
st.adjust_xticks_dates_to_md(ax)
ax.legend(loc='best')

#%%

fig,ax = plt.subplots(2,1,figsize=(12,8))

ax[0].set_ylabel("Error relative to NEE\n(fractional)")

ax[1].set_ylabel("Absolute error\n(kton/3-hour)")

vr = np.sqrt(scalars['UrbanVPRM']).reshape(nt,-1).mean(axis=-1)
ax[0].plot(ttot, vr)

vr = np.abs((np.sqrt(scalars['UrbanVPRM'])*NEE)).reshape(nt,-1).sum(axis=-1)
ax[1].plot(ttot, vr/1e6)
    
[st.adjust_xticks_dates_to_md(a) for a in ax]

fig.tight_layout()

fig,ax = plt.subplots(1,1,figsize=(12,5))

ax.set_ylabel("Error relative to NEE\n(fractional)")

vr = np.sqrt(scalars['UrbanVPRM']).reshape(nt,-1).mean(axis=-1)
ax.plot(np.arange(0,24,3), vr.reshape(int(nt/8),8).mean(axis=0), label='mean')
vr = np.median(np.sqrt(scalars['UrbanVPRM']).reshape(nt,-1),axis=-1)
ax.plot(np.arange(0,24,3), np.median(vr.reshape(int(nt/8),8), axis=0), label='median')
ax.legend(loc='best')
ax.set_xlim(0,ax.get_xlim()[1])
ax.set_xlabel("Hour in day")
    
ax.legend(loc='best')



fig,ax = plt.subplots(1,1,figsize=(12,5))

ax.set_xlabel("Error relative to NEE (fractional)")

vr = np.sqrt(scalars['UrbanVPRM'])
ax.hist(vr, bins=np.linspace(0,5,30))
    




