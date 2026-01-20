#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to analyze an aggregate of parallel inversions that together cover a longer period.
We use the post-processed inversion output.

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
from postpostprocess_multi import Postpostprocess_multi
    
#%%


inversion_name = 'baseAKLNWP_base'
startdate = datetime(2022,1,1)
enddate = datetime(2022,3,9)
dt_inversion = timedelta(days=28)
dt_spinup    = timedelta(days=7)
dt_spindown  = timedelta(days=7)

postpr_m = Postpostprocess_multi(inversion_name, startdate, enddate, dt_inversion, dt_spinup, dt_spindown)
postpr_m.read_all_postprocessed_data()


#%%

fig,ax = plt.subplots(3,2, figsize=(18,10.5))

colors = sns.color_palette('Set2')
sns.set_context('talk')


priors = 'UrbanVPRM','MahuikaAuckland','Total'
# priors = 'MahuikaAuckland','Total'
labels = 'prior','posterior',#'true'
for ipri,prior in enumerate(priors):
    
    dates = postpr_m.dates_all['daily']
    
    ax[ipri,0].set_title("Daily fluxes %s"%prior)
    
    for ilab,label in enumerate(labels):
        emi = postpr_m.emis_all[label]['per_domain']['daily'][prior][:,0]/1e6
        ax[ipri,0].plot(dates, emi, color=colors[ilab], label=label)
    
    ax[ipri,0].legend(loc='best')
    
    ax[ipri,1].set_title("Difference daily fluxes with truth %s"%prior)
    etrue = postpr_m.emill['true']['per_domain']['daily'][prior][:,0]
    for ilab,label in enumerate(['prior','posterior']):
        elab = postpr_m.emis_all[label]['per_domain']['daily'][prior][:,0]
        
        diff = (etrue-elab)
        bias = np.mean(diff)
        rms  = np.sqrt(np.mean(diff**2))
        
        ax[ipri,1].plot(dates, diff/1e6, color=colors[ilab], label=label)
        
        ax[ipri,1].text(0.25+0.28*ilab,0.8, 'Bias %1.1f kt/day\nRMS %1.1f kt/day'%(bias/1e6,rms/1e6), 
                        fontsize=16, color=colors[ilab], transform=ax[ipri,1].transAxes)
        
    ax[ipri,1].legend(loc='lower left')
    
ax[1,0].get_legend().remove()
    
[a.set_ylabel("Flux (kt CO$_2$/day)") for a in ax.flatten()]
[st.adjust_xticks_dates_to_m(a, month_fmt='%b') for a in ax.flatten()]
fig.tight_layout()
fig.savefig("%s/fluxes_inner.png"%path_figs)

#%%


colors = sns.color_palette('Set2')

priors = 'UrbanVPRM','MahuikaAuckland','BiomeBGC','EDGARv8'
fig,ax = plt.subplots(len(priors),1, figsize=(12,len(priors)*5))
# priors = 'MahuikaAuckland',
labels = 'prior','posterior'
for ipri,prior in enumerate(priors):
    
    ax[ipri].set_title("Daily flux uncertainty %s"%prior)
    
    for ilab,label in enumerate(labels):
        ax[ipri].plot(postpr_m.dates_all['daily'], 100*np.sqrt(postpr_m.unc_all_rel[label]['per_domain']['daily'][prior][prior][:,0]), color=colors[ilab], label=label)
    
    ax[ipri].legend(loc='best')
    ax[ipri].set_ylim([0,ax[ipri].get_ylim()[1]+3])
    
    st.adjust_xticks_dates_to_md(ax[ipri])
    
    
[a.set_ylabel("Uncertainty (%)") for a in ax.flatten()]

fig.tight_layout()
fig.savefig("%s/flux_uncertainty_rel.png"%path_figs, bbox_inches='tight', dpi=300)


#%%

# Absolute uncertainty

fig,ax = plt.subplots(2,1, figsize=(12,10))

colors = sns.color_palette('Set2')


priors = 'UrbanVPRM','MahuikaAuckland',
# priors = 'MahuikaAuckland',
labels = 'prior','posterior'
for ipri,prior in enumerate(priors):
    
    ax[ipri].set_title("Daily flux uncertainty %s"%prior)
    
    for ilab,label in enumerate(labels):
        unc = postpr_m.unc_all_abs[label]['all_domains']['daily'][prior][prior] / 1e6
        ax[ipri].plot(postpr_m.dates_all['daily'], np.abs(unc), color=colors[ilab], label=label)
    
    ax[ipri].legend(loc='best')
    
    
[a.set_ylabel("Uncertainty (kt/day)") for a in ax.flatten()]

fig.tight_layout()
fig.savefig("%s/flux_uncertainty_abs.png"%path_figs, bbox_inches='tight', dpi=300)

#%%

sites = postpr_m.obs_all['prior'].keys()
nsite = len(sites)

sns.set_context("talk")

fig,ax = plt.subplots(nsite,2,figsize=(18,4*nsite))

for isite,site in enumerate(sites):
    
    ax[isite,0].set_title("Obs %s"%site)
    for ilab,label in enumerate(['prior','posterior','true']):
        dates = postpr_m.obs_all[label][site]['dates'], 
        co2   = postpr_m.obs_all[label][site]['co2']
        
        ax[isite,0].scatter(dates, co2, facecolor='none', edgecolor=colors[ilab], linewidth=1.5, label=label)
        
    ax[isite,0].legend(loc='best')
        
    ax[isite,1].set_title("Difference %s"%site)
    co2_true = postpr_m.obs_all['true'][site]['co2']
    for ilab,label in enumerate(['prior','posterior']):
        dates, co2 = postpr_m.obs_all[label][site]['dates'], postpr_m.obs_all[label][site]['co2']
        
        diff = co2-co2_true
        
        bias = np.mean(diff)
        rms = np.sqrt(np.mean(diff**2))
        
        ax[isite,1].scatter(dates, diff, facecolor='none', edgecolor=colors[ilab], linewidth=1.5, label=label)
        ax[isite,1].text(0.05+0.25*ilab,0.1, 'Bias %2.2f ppm\nRMS %2.2f ppm'%(bias,rms), 
                         fontsize=13, color=colors[ilab], transform=ax[isite,1].transAxes)
    
    ax[isite,1].legend(loc='upper right')
        
[a.set_ylabel("CO$_2$ [ppm]") for a in ax.flatten()]    
    
fig.tight_layout()
fig.savefig("%s/obs.png"%path_figs)


#%%

sns.set_context('talk',font_scale=1.2)
fig,ax = plt.subplots(nsite,2,figsize=(18,5*nsite))

bins_obs = np.linspace(-10,12,30)
bins_diff = np.linspace(-5,5,30)

for isite,site in enumerate(sites):
    
    ax[isite,0].set_title("Obs %s"%site)
    for ilab,label in enumerate(['prior','posterior','true']):
        dates, co2 = postpr_m.obs_all[label][site]['dates'], postpr_m.obs_all[label][site]['co2']
        
        ax[isite,0].hist(co2, color=colors[ilab], bins=bins_obs, alpha=0.5, linewidth=1.5, label=label)
        
    ax[isite,0].legend(loc='best')
        
    ax[isite,1].set_title("Difference %s"%site)
    co2_true = postpr_m.obs_all['true'][site]['co2']
    for ilab,label in enumerate(['prior','posterior']):
        dates, co2 = postpr_m.obs_all[label][site]['dates'], postpr_m.obs_all[label][site]['co2']
        
        diff = co2-co2_true
        
        bias = np.mean(diff)
        rms = np.sqrt(np.mean(diff**2))
        
        ax[isite,1].hist(diff, color=colors[ilab], bins=bins_diff, alpha=0.5, linewidth=1.5, label=label)
        ax[isite,1].text(0.05,0.75-0.23*ilab, 'Bias %2.2f ppm\nRMS %2.2f ppm'%(bias,rms), 
                         fontsize=22, color=colors[ilab], transform=ax[isite,1].transAxes)
        ax[isite,1].set_xlim(-5,5)
    ax[isite,1].legend(loc='upper right')
        
[a.set_ylabel("# obs") for a in ax.flatten()]        
[a.set_xlabel("CO$_2$ [ppm]") for a in ax.flatten()]    

fig.tight_layout()
fig.savefig("%s/hist_obs.png"%path_figs)


#%%

# # Plotting the different stages of perturbation to y

# import pandas as pd

# from construct_y_synthetic import ConstructorY_synthetic as ConstructorY

# date0 = postpr_m.startdates_all[0]
# date1 = postpr_m.enddates_all[0]

# cy = ConstructorY(inversion_name,  {'date_start':['direct',date0], 'date_end':['direct', date1]})
# cy.read_observation_data_raw()

# ysites,ydates,_ = cy.read_yvector()

# H = cy.read_Hmatrix()
# y_from_xprior    = H @ cy.read_state()[0]
# y_from_xpert     = H @ cy.read_xtrue()
# ypert_from_xpert = cy.read_yvector()[-1]

# usites = np.unique(ysites)
# colors = sns.color_palette('Set2')

# fig,ax = plt.subplots(len(usites), 1, figsize=(10,5*len(usites)))

# sns.set_palette('Set2')

# for i,site in enumerate(usites):
    
#     mask = ysites==site
#     datess = ydates[mask]
#     keys = np.argsort(datess)
    
#     n = len(keys)
#     labels = ['Prior']*n + ['Only prior error']*n + ['Prior + obs error']*n
    
#     y = np.concatenate( [y_from_xprior[mask][keys], y_from_xpert[mask][keys], ypert_from_xpert[mask][keys]] )
    
#     data = np.array( [ np.concatenate([datess[keys]]*3), y, labels ])
    
#     df = pd.DataFrame(data=data.T, columns=['dates','XCO2','Stage'])
    
#     # grid = sns.jointplot(data=df, x='dates', y='XCO2', hue='Stage', 
#     #                      xlim=(datetime(2022,1,8), datetime(2022,2,8)), legend=True)
    
#     # grid.ax_joint.set_xticks([datetime(2022,1,8)+timedelta(days=10*i) for i in range(4)])
#     # grid.ax_joint.set_xlabel(None)
#     # grid.ax_joint.set_ylabel("XCO$_2$ (ppm)")
#     # grid.ax_joint.set_ylim(-8,8)
#     # st.adjust_xticks_dates_to_md(grid.ax_joint)
#     # grid.fig.set_figwidth(12)
#     # grid.fig.set_figheight(6)
    
#     # # st.adjust_xticks_dates_to_md(ax[i])

#     # grid.fig.savefig('%s/obs_4week_jointplot_%s.png'%(path_figs,site))
    
#     ax[i].set_title(site)
#     g = sns.kdeplot(df, x='XCO2',hue='Stage',ax=ax[i])
#     ax[i].set_xlim(-5,5)
    
#     ax[i].legend_.set_title(None)
    
# plt.tight_layout()
# fig.savefig("%s/obs_plot_kdensity.png"%path_figs)


#%%


#%%

# Plot true versus prior emissoins

epri, etru = postpr_m.emis_all['prior'], postpr_m.emis_all['true']
nday = 28

lw = 4.5

priors = 'UrbanVPRM','MahuikaAuckland',
# priors = 'MahuikaAuckland',
fig, ax = plt.subplots(2,1, figsize=(12,10))
for i,prior in enumerate(priors):
    # ax[i].set_title(prior)
    
    eprii = epri['all_domains']['daily'][prior]
    etrui = etru['all_domains']['daily'][prior]
    
    dates=  postpr_m.dates_all['daily']
    ax[i].plot(dates[:nday], eprii[:nday]/1e6, linewidth=lw, label='Prior')
    ax[i].plot(dates[:nday], etrui[:nday]/1e6, linewidth=lw, label='True')
    
    ax[i].set_xticks( [datetime(2022,1,8)+ timedelta(days=7*i) for i in range(5)] )
    # st.adjust_xticks_dates_to_md(ax[i])
    ax[i].set_ylabel("Flux [kt/day]")
    
    if prior == 'MahuikaAuckland':
        ax[i].set_ylim(0,ax[i].get_ylim()[1])
        
    ax[i].legend(loc='best')
    
plt.tight_layout()
plt.savefig("%s/fluxes_prior_v_true_daily"%path_figs)

#%%

# Gains matrix versus wind directoin
dates = postpr_m.dates_all['per_timestep']

from read_meteo_NAME import RetrievorWinddataNAME, bin_based_on_ws_wd
retr = RetrievorWinddataNAME(inversion_name)
ws,wd,pbl,rh,t = retr.retrieve_winddata_NAME(dates, ['Mangere', 'ManukauHeads'])

Gsum = {'MahuikaAuckland':np.array([]), 'UrbanVPRM':np.array([])}
# Gsum = {'MahuikaAuckland':np.array([])}
for startdate_i,enddate_i in zip(postpr_m.startdates_all, postpr_m.enddates_all):
    mod_rc =  {'date_start':['direct',startdate_i], 'date_end':['direct',enddate_i]}
    postpr = Postprocessing(postpr_m.inversion_name, mod_rc, postpr_m.dt_spinup, postpr_m.dt_spindown)
    Gagg_i = postpr.read_aggregated_Gmatrix()['all_domains']['per_timestep']
    
    for cat in Gsum.keys():
        Gsum[cat] = np.append(Gsum[cat], Gagg_i[cat].sum(axis=(1,2)))
        

Gbinned = {}
wd_bins = np.arange(0,360,45)
for cat in Gsum.keys():
    Gbinned[cat] = bin_based_on_ws_wd([0,100], wd_bins, ws[0], wd[0], Gsum[cat])
    
    
fig,ax = plt.subplots(1,1,figsize=(10,6))
ax.plot(st.mid(wd_bins), Gbinned['MahuikaAuckland'][0][0])
ax.set_ylim(0,ax.get_ylim()[1])
axt = ax.twinx()
axt.plot(st.mid(wd_bins), Gbinned['MahuikaAuckland'][2][0], color=colors[1])
axt.set_ylim(0,axt.get_ylim()[1])



#%%

# # Chi-squared
# osse = postpr_m.inversion_example
# osse.read_inversion_input()
# osse.read_inversion_output()
# ndeg = len(osse.yvector)
# ydiff = osse.yvector - (osse.Hmatrix @ osse.xopt)
# xdiff = osse.xprior - osse.xopt

# Rinv = np.linalg.inv(osse.Rmatrix)
# Binv = np.linalg.inv(osse.Bmatrix)
# X2 = ( (ydiff.T @ Rinv @ ydiff) + (xdiff.T @ Binv @ xdiff) ) / ndeg
# print(X2)
# print((ydiff.T @ Rinv @ ydiff)/ndeg, (xdiff.T @ Binv @ xdiff)/ndeg)


#%%

# # UrbanVPRM, MahuikaAuckland correlation

# u = postpr_m.unc_all_rel['posterior']['per_domain']['diurnal']
# cros = u['MahuikaAuckland']['UrbanVPRM'][:,0]
# inv1 = u['MahuikaAuckland']['MahuikaAuckland'][:,0]
# inv2 = u['UrbanVPRM']['UrbanVPRM'][:,0]

# fig,ax = plt.subplots(1,1,figsize=(10,5))
# ax.plot( np.arange(1.5,24,3), np.nanmean( (cros / np.sqrt(inv1*inv2)).reshape(-1,8),axis=0), 'k', linewidth=2.5)

# # colors = sns.color_palette('RdBu',12)
# # for i in range(12):
# #     ax.plot( np.arange(1.5,24,3), (cros / np.sqrt((inv1*inv2))).reshape(-1,8)[i], label=i, color=colors[i])
    
# ax.legend()
# ax.set_xlabel("Hour in day")
# ax.set_ylabel("R$^2$")
# ax.set_ylim(0,ax.get_ylim()[1])

#%%



# u = postpr_m.unc_all_abs['posterior']['per_domain']['diurnal']
# cros = u['MahuikaAuckland']['UrbanVPRM'][:,0]
# inv1 = u['MahuikaAuckland']['MahuikaAuckland'][:,0]
# inv2 = u['UrbanVPRM']['UrbanVPRM'][:,0]

# print( np.nanmean( (cros**2 / (inv1*inv2)).reshape(-1,8), axis=0 ) )


#%%

domains_plot = 'Mah0p3_in',#'Mah0p3_out'

x,priors,domains = osse.read_state()

ysites,ydates,yvector = osse.read_yvector()

upriors = np.unique(priors)
usites  = np.unique(ysites)

fig,ax = plt.subplots(len(usites),1,figsize=(10,5*len(usites)))
for prior in np.unique(priors)[::-1]:
    mask_x_1 = (priors==prior) 
    mask_x_2 = np.array([d in domains_plot for d in domains])
    mask_x = mask_x_1 & mask_x_2
    
    y = osse.Hmatrix[:,mask_x] @ x[mask_x]
    
    for i,site in enumerate(usites):
        mask_y = (ysites==site)
        ax[i].set_title(site)
        ax[i].plot(ydates[mask_y], y[mask_y], '.', label=prior)
        
[a.set_xticks([datetime(2022,1,1)+timedelta(days=7*i) for i in range(7) ]) for a in ax]
    
[st.adjust_xticks_dates_to_md(a) for a in ax]
[a.legend(loc='best') for a in ax]
[a.set_ylabel("CO$_2$ (ppm)") for a in ax]

fig.tight_layout()
fig.savefig("%s/prior_enh_per_inventory"%path_figs)

y = []

afternoon = np.arange(12,17)
for prior in np.unique(priors)[::-1]:
    mask_x_1 = (priors==prior) 
    mask_x_2 = np.array([d in domains_plot for d in domains])
    mask_x = mask_x_1 & mask_x_2
    
    mask_y = np.array([ d.hour in afternoon for d in ydates ])
    
    y.append(osse.Hmatrix[mask_y][:,mask_x] @ x[mask_x])

print(np.corrcoef(y[0],y[-1]))

#%%

postpr_aft = Postpostprocess_multi('baseAKLNWP_vprm_afternoon', startdate, enddate, dt_inversion, dt_spinup, dt_spindown)
postpr_aft.read_all_postprocessed_data()


priors = 'UrbanVPRM','MahuikaAuckland'
# Sum over 4-week periods
dt_diurn = postpr_m.inversion_example.rc.opt_freq_diurnal
nt_diurn = int(24/dt_diurn)
tsteps = np.arange(0,24,dt_diurn)

domains = 'per_domain'

# Diurnal cycle
fig, ax = plt.subplots(len(priors), 2, figsize=(20,5*len(priors)))

for i,prior in enumerate(priors):
    
    for label,e in postpr_m.emis_all.items():
        e = e[domains]['diurnal'][prior][:,0]
        ls = '--' if label=='posterior' else '-'
        ax[i,0].plot(tsteps, e.reshape(-1,8).sum(axis=0)/(12*28)/1e6, linestyle=ls, label=label)
        
    e2 = postpr_aft.emis_all['posterior'][domains]['diurnal'][prior][:,0]
    ax[i,0].plot(tsteps, e2.reshape(-1,8).sum(axis=0)/(12*28)/1e6, '--', label='posterior afternoon-only')
        
    for label,unc in postpr_m.unc_all_rel.items():
        unc = unc[domains]['diurnal'][prior][prior][:,0]
        ax[i,1].plot(tsteps, 100*unc.reshape(-1,8).mean(axis=0), label=label)
    unc = postpr_aft.unc_all_rel['posterior'][domains]['diurnal'][prior][prior][:,0]
    ax[i,1].plot(tsteps, 100*unc.reshape(-1,8).mean(axis=0), label='posterior afternoon-only')
    ax[i,1].set_ylim(0,ax[i,1].get_ylim()[1])
        
    
    ax[i,0].set_xlabel('Hour in day')
    ax[i,1].set_xlabel('Hour in day')
    ax[i,0].set_ylabel('Flux (kt/day)')
    ax[i,1].set_ylabel('Uncertainty (%)')
    ax[i,0].legend()
    ax[i,1].legend()

plt.tight_layout()

#%%

a = np.random.rand(5,5)
b = np.random.rand(5,5)
c = np.zeros((10,10))
c[:5,:5] = a
c[5:,5:] = b

fig,ax = plt.subplots(3,1,figsize=(10,27))

cp = ax[0].pcolormesh(np.linalg.inv(c), vmin=0, vmax=1)
plt.colorbar(cp,ax=ax[0])

cp = ax[1].pcolormesh(np.linalg.inv(a), vmin=0, vmax=1)
plt.colorbar(cp,ax=ax[1])

#%%

B = np.load('/nesi/nobackup/niwa03154/nauss/Data/inversions/baseAKLNWP_vprm_std/20220102_20220211/input/Bmatrix.npy')

#%%

def is_semipos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

B[B<0] = 0.0

print(is_semipos_def(B))

# x2 = np.random.multivariate_normal(np.ones(len(B)), B, 1)[0]

#%%

nt = 10
nx = 5
x = np.ones(nt*nx)
clen = 3

dt = np.arange(nt)
dt_2d = np.abs(dt[:,np.newaxis]-dt[np.newaxis,:])
bt = np.exp(-dt_2d/clen)



dx = np.arange(nx)
dx_2d = np.abs(dx[:,np.newaxis]-dx[np.newaxis,:])

x2 = np.random.multivariate_normal(x, bb, 1)[0]




#%%

# What fraction of NZ anthropogenic CO2 fluxes is in Auckland?

import read_priors as rp
import inversion as inv

dates = [datetime(2022,1,1) + timedelta(days=i) for i in range(365)]
lon, lat = inv.getLonLatNAME('Out7p0')
edgarv8 = rp.read_preprocessed_edgarv8(dates, 'Out7p0')

#%%

lonlim_akl = 174,175.5
latlim_akl = -37.5, -36

areas = st.calc_area_per_gridcell(lat, lon, bounds=False)

fig, ax = plt.subplots(1,1,figsize=(10,8))
cp = ax.pcolormesh(lon, lat, edgarv8.mean(axis=(0,1)), vmax=1e-7)
plt.colorbar(cp, ax=ax)
st.plotSquare(ax, lonlim_akl, latlim_akl, color='white')

etot_nz = (edgarv8*areas*3600).sum()

masklon = (lon>lonlim_akl[0]) & (lon<lonlim_akl[1])
masklat = (lat>latlim_akl[0]) & (lat<latlim_akl[1])
mask_akl = np.outer(masklat,masklon)
etot_akl = (edgarv8*areas*3600).sum(axis=(0,1))[mask_akl].sum()

print("%2.2f %%"%(100*etot_akl/etot_nz))













