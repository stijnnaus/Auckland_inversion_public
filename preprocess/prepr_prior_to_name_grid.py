#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here we regrid the emission priors to the NAME grid.
Currently set-up for MahuikaAuckland, UrbanVPRM, EDGARv7
For Mahuika we regrid per-category. For UrbanVPRM we only regrid NEE. For EDGARv7
we regrid as one.
We use xesmf conservative regridding.
The resulting emissions on NAME grid (always in kg/m2/s) can then be combined
with footprints to calculate CO2 enhancements at sites.
"""

import functions_stijn as st
import numpy as np
import read_priors as rp
import seaborn as sns
import calendar
import inversion as inv
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
from netCDF4 import Dataset
import os,glob,sys,time
import warnings
sns.set_context('talk')
colors = sns.color_palette('Set2')

        
def calc_monthly_averages(dates, emis, time_ax=0):
    umonths = find_unique_months(dates)
    emis_m = []
    for i,umonth in enumerate(umonths):
        y, m = umonth.year, umonth.month
        mask = [((y==d.year) & (m==d.month)) for d in dates]
        emis_m.append(np.mean(emis[mask], axis=0))
    return umonths, np.array(emis_m)

def find_unique_months(dates):
    ym = [[d.year,d.month] for d in dates]
    ym = np.unique(ym,axis=0)
    dates_m = np.array([datetime(y, m, 1) for y,m in ym])
    return dates_m

warnings.filterwarnings(action='ignore', message='Input array is not C_CONTIGUOUS')

from base_paths import path_inv_base
path_out_prepr = '%s/priors/preprocessed/'%path_inv_base
domains = ['Mah0p3', 'In1p5', 'Out7p0']


#%%

def preprocess_mahuika(domain):
    cats = rp.get_all_mahuika_cats()
    # for cat in cats:
    for cat in ['residential_CO2bio', 'residential_CO2ff']:
        print("Preprocessing Mah-%s"%cat)
        if cat[:5]=='resid':
            # Residential varies per year, the other categories don't
            for year in [2021,2022,2023]:
                print("Year %i"%year)
                preprocess_mahuika_onecat(domain, cat, year)
                
        else:
            preprocess_mahuika_onecat(domain, cat)
        
def preprocess_mahuika_onecat(domain, cat, year=2016):
    '''
    Parent function that reads original emission fields, regrids them to a 
    prescribed NAME domain and then writes to output for one Mahuika emission category.
    
    The only category for which year matters is residential (bio and ff)
    '''
    
    lons_mah, lats_mah, emis = rp.read_mahuika_onecat(cat, year=year, unit='kg/m2/s')
    emis_regr = regrid_mahuika_onecat(domain, lons_mah, lats_mah, emis)
    write_preprocessed_mahuika_onecat_to_out(emis_regr, domain, cat, year)

def regrid_mahuika_onecat(domain, lons_mah, lats_mah, emis):
    '''
    Regrid original Mahuika emissions to a new grid described by domain
    We use conservative regridding from the xesmf regridder.
    '''
    
    lons_name, lats_name = inv.getLonLatNAME(domain)
    nlon_regr, nlat_regr = len(lons_name), len(lats_name)
    ntime = emis.shape[2]
    
    regridder = st.make_xesmf_regridder(lons_mah, lats_mah, lons_name, lats_name, 'conservative')
    if len(emis.shape)==4:
        # Includes an extra dimension for weekday/weekend
        emis_regr = np.zeros((ntime, 2, nlat_regr, nlon_regr))
        for i in range(ntime):
            for j in [0,1]: # weekday/weekend
                emis_regr[i,j,:,:] = regridder(emis[:,:,i,j])
    elif len(emis.shape)==3:
        emis_regr = np.zeros((ntime, nlat_regr, nlon_regr))
        for i in range(ntime):
            emis_regr[i,:,:] = regridder(emis[:,:,i])
            
    return emis_regr

def write_preprocessed_mahuika_onecat_to_out(emis, domain, cat, year):
    '''
    Write the regridded emissions to an output file.
    Depending on the frequency  at which a category is available we either write one 
    file (fixed daily cycle), two (weekday/weekend) or 366 (daily). We write
    binary .npy files for fastest reading; the lat, lon coordinates can be later
    retrieved from the domain name.    
    '''
    
    freq = rp.get_freq_from_cat_mah(cat)
    path_out = "%s/MahuikaAuckland/%s/%s/"%(path_out_prepr,domain,cat)
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    
    if freq=='fixed':
        np.save(path_out + 'emis_fixed.npy', emis)
    elif freq=='weekday1':
        np.save(path_out + 'emis_weekday.npy', emis)
    elif freq=='weekday2':
        np.save(path_out + 'emis_weekday.npy', emis[:,1,:,:])
        np.save(path_out + 'emis_weekend.npy', emis[:,0,:,:])
    elif freq=='daily':
        date = datetime(year,1,1)
        nday = st.get_nday_in_year(year)
        for i in range(nday):
            np.save(path_out + date.strftime('emis_%Y%m%d.npy'), emis[24*i:24*(i+1)])
            date += timedelta(days=1)
    else:
        raise ValueError("Unknown frequency %s"%freq)
        
        
domains = ['Mah0p3','In1p5','Out7p0']
for domain in domains:
    preprocess_mahuika(domain)
    
#%%

def preprocess_UrbanVPRM(domain, varb='NEE'):
    # Read UrbanVPRM
    lons_name, lats_name = inv.getLonLatNAME(domain)
    dates = [datetime(2018,1,1) + timedelta(days=i) for i in range(1)]
    lons_in, lats_in, emis_in = rp.read_UrbanVPRM(dates, varb=varb)
    emis_out = regrid_UrbanVPRM(lons_in, lats_in, emis_in, lons_name, lats_name)
    write_preprocessed_UrbanVPRM(dates, emis_out, varb=varb)

def regrid_UrbanVPRM(lons_in, lats_in, emis_in, lons_name, lats_name):
    ndate = emis_in.shape[0]
    emis_out = np.zeros((ndate, 24, len(lats_name), len(lons_name)))
    regridder = st.make_xesmf_regridder(lons_in, lats_in, lons_name, lats_name, 'conservative')
    for i in range(ndate):
        for j in range(24):
            emis_out[i,j] = regridder(emis_in[i,j])

    return emis_out

def write_preprocessed_UrbanVPRM(dates, emis, varb):
    # We write one file per day
    path_out = path_out_prepr + '/UrbanVPRM/%s/%s/'%(varb,domain)
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    fname_fmt = path_out+'emis_%Y%m%d.npy'
    for i,date in enumerate(dates):
        np.save(date.strftime(fname_fmt), emis[i])

varb = 'NEE'
for domain in domains:
    print(domain)
    preprocess_UrbanVPRM(domain, varb)
    
    
#%%

# I want to assess separately how much information to distinguish UrbanVPRM from Mahuika-Auckland
# comes from spatial information, and how much from temporal information. For this I generate
# two new versions of Mahuika-Auckland: one with UrbanVPRM spatial distribution, and one with
# UrbanVPRM temporal distribution.

def preprocess_MahuikaAuckland_mixed_with_UrbanVPRM(domain):
    
    lonc, latc = inv.getLonLatNAME(domain, bounds=False)
    lonb, latb = inv.getLonLatNAME(domain, bounds=True)
    area = st.calc_area_per_gridcell(latb, lonb, bounds=True)
    
    dates = [datetime(2021,12,1) + timedelta(days=i) for i in range(31)]
    
    # I'm looping over days so I don't have to hold large arrays in memory for no reason
    for date in dates:
        print(date)
        
        dates_i = [date]
        
        emi_mahk = rp.read_preprocessed_1inventory('MahuikaAuckland', dates_i, domain, cats='all') # (day,hour,lat,lon) // [kg/m2/s] 
        emi_vprm = rp.read_preprocessed_1inventory('UrbanVPRM', dates_i, domain, varb='NEE')
        
        etot_mahk = np.sum(emi_mahk*area, axis=(2,3)) # [kg/s]
        etot_vprm = np.sum(emi_vprm*area, axis=(2,3))
        
        # 1. Replace spatial distribution, conserve temporal
        emahk_new_x = (etot_mahk/etot_vprm)[:,:,np.newaxis,np.newaxis] * emi_vprm
        write_preprocessed_Mahk_replaced_x(dates_i, emahk_new_x)
        
        # 2. Replace temporal distribution, conserve spatial
        emahk_new_t = (etot_vprm/etot_mahk)[:,:,np.newaxis,np.newaxis] * emi_mahk
        write_preprocessed_Mahk_replaced_t(dates_i, emahk_new_t)

def write_preprocessed_Mahk_replaced_x(dates, emis):
    path_out = path_out_prepr + '/MahuikaAuckland_new_x/%s/'%domain
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    fname_fmt = path_out+'emis_%Y%m%d.npy'
    for i,date in enumerate(dates):
        np.save(date.strftime(fname_fmt), emis[i])

def write_preprocessed_Mahk_replaced_t(dates, emis):
    path_out = path_out_prepr + '/MahuikaAuckland_new_t/%s/'%domain
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    fname_fmt = path_out+'emis_%Y%m%d.npy'
    for i,date in enumerate(dates):
        np.save(date.strftime(fname_fmt), emis[i])
        
domains = ['Mah0p3']
for domain in domains:
    print(domain)
    preprocess_MahuikaAuckland_mixed_with_UrbanVPRM(domain)
    
    
#%%

# Same but keeping Mhauika the same and changing UrbanVPRM
def preprocess_UrbanVPRM_mixed_with_MahuikaAuckland(domain):
    
    lonc, latc = inv.getLonLatNAME(domain, bounds=False)
    lonb, latb = inv.getLonLatNAME(domain, bounds=True)
    area = st.calc_area_per_gridcell(latb, lonb, bounds=True)
    
    dates = [datetime(2021,12,1) + timedelta(days=i) for i in range(31+365)]
    
    # I'm looping over days so I don't have to hold large arrays in memory for no reason
    for date in dates:
        print(date)
        
        dates_i = [date]
        
        emi_mahk = rp.read_preprocessed_1inventory('MahuikaAuckland', dates_i, domain, cats='all') # (day,hour,lat,lon) // [kg/m2/s] 
        emi_vprm = rp.read_preprocessed_1inventory('UrbanVPRM', dates_i, domain)
        
        etot_mahk = np.sum(emi_mahk*area, axis=(2,3)) # [kg/s]
        etot_vprm = np.sum(emi_vprm*area, axis=(2,3))
        
        # 1. Replace spatial distribution, conserve temporal
        evprm_new_x = (etot_vprm/etot_mahk)[:,:,np.newaxis,np.newaxis] * emi_mahk
        write_preprocessed_UrbanVPRM_replaced_x(dates_i, evprm_new_x)
        
        # 2. Replace temporal distribution, conserve spatial
        evprm_new_t = (etot_mahk/etot_vprm)[:,:,np.newaxis,np.newaxis] * emi_vprm
        write_preprocessed_UrbanVPRM_replaced_t(dates_i, evprm_new_t)

def write_preprocessed_UrbanVPRM_replaced_x(dates, emis):
    path_out = path_out_prepr + '/UrbanVPRM_new_x/%s/'%domain
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    fname_fmt = path_out+'emis_%Y%m%d.npy'
    for i,date in enumerate(dates):
        np.save(date.strftime(fname_fmt), emis[i])

def write_preprocessed_UrbanVPRM_replaced_t(dates, emis):
    path_out = path_out_prepr + '/UrbanVPRM_new_t/%s/'%domain
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    fname_fmt = path_out+'emis_%Y%m%d.npy'
    for i,date in enumerate(dates):
        np.save(date.strftime(fname_fmt), emis[i])
        
domains = ['Mah0p3']
for domain in domains:
    print(domain)
    preprocess_UrbanVPRM_mixed_with_MahuikaAuckland(domain)
        
#%%

def preprocess_UrbanVPRM_monthly(domain):
    # Read UrbanVPRM
    lons_name, lats_name = inv.getLonLatNAME(domain)
    dates = [datetime(2018,1,1) + timedelta(days=i) for i in range(365)]
    lons_in, lats_in, emis_in = rp.read_UrbanVPRM(dates)
    dates_m, emis_in_m = calc_monthly_averages(dates, emis_in)
    emis_out = regrid_UrbanVPRM(lons_in, lats_in, emis_in_m, lons_name, lats_name)
    write_preprocessed_UrbanVPRM_monthly(dates_m, emis_out)

def write_preprocessed_UrbanVPRM_monthly(dates, emis):
    # We write one file per day
    path_out = path_out_prepr + '/UrbanVPRM_monthly/NEE/%s/'%domain
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    fname_fmt = path_out+'emis_%Y%m.npy'
    for i,date in enumerate(dates):
        np.save(date.strftime(fname_fmt), emis[i])

for domain in domains:
    print(domain)
    preprocess_UrbanVPRM_monthly(domain)

#%%

def preprocess_EDGARv7(domain):
    year = 2021
    lons_name, lats_name = inv.getLonLatNAME(domain)
    lons_in, lats_in, emis_in = rp.read_EDGARv7(year=year)
    emis_out = regrid_EDGARv7(lons_in, lats_in, emis_in, lons_name, lats_name)
    write_preprocessed_EDGARv7(year, emis_out, domain)
    
def regrid_EDGARv7(lons_in, lats_in, emis_in, lons_name, lats_name):
    print(lats_in.min(), lats_in.max())
    regridder = st.make_xesmf_regridder(lons_in, lats_in, lons_name, lats_name, 'conservative')
    return regridder(emis_in)
    
def write_preprocessed_EDGARv7(year, emis, domain):
    path_out = path_out_prepr + '/EDGARv7/%s/'%domain
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    np.save(path_out + 'emis_%4.4i.npy'%year, emis)
    
for domain in domains:
    print(domain)
    preprocess_EDGARv7(domain)
    
#%%

def preprocess_EDGARv8(domain, years):
    for year in [2021,2022,2023]:
        lons_name, lats_name = inv.getLonLatNAME(domain)
        lons_in, lats_in, emis_in = rp.read_EDGARv8_allcats(year=year)
        emis_out = regrid_EDGARv7(lons_in, lats_in, emis_in, lons_name, lats_name)
        write_preprocessed_EDGARv8(year, emis_out, domain)
    
def write_preprocessed_EDGARv8(year, emis, domain):
    
    path_out = path_out_prepr + '/EDGARv8/%s/'%domain
    if not os.path.exists(path_out):
        os.makedirs(path_out)
        
    for imonth,month in enumerate(range(1,13)):
        np.save(path_out + 'emis_%4.4i%2.2i.npy'%(year,month), emis[imonth])
    
years = [2021,2022,2023]
domains = 'Mah0p3','In1p5','Out7p0'
for domain in domains:
    print(domain)
    preprocess_EDGARv8(domain, years=years)
    
#%%

def preprocess_ODIAC(domain):
    dates = [datetime(2021,i,1) for i in range(1,13)]
    lons_name, lats_name = inv.getLonLatNAME(domain)
    lonb = lons_name.min()-0.1, lons_name.max()+0.1
    latb = lats_name.min()-0.1, lats_name.max()+0.1
    lons_in, lats_in, emis_in = rp.read_ODIAC(dates, lonb, latb, timezone='NZST')
    emis_out = regrid_ODIAC(lons_in, lats_in, emis_in, lons_name, lats_name)
    write_preprocessed_ODIAC(dates, emis_out, domain)
    
def regrid_ODIAC(lons_in, lats_in, emis_in, lons_name, lats_name):
    regridder = st.make_xesmf_regridder(lons_in, lats_in, lons_name, lats_name, 'conservative')
    return regridder(emis_in)
    
def write_preprocessed_ODIAC(dates, emis, domain):
    path_out = path_out_prepr + '/ODIAC/%s/'%domain
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    for i,date in enumerate(dates):
        np.save(path_out + date.strftime('emis_%Y%m.npy'), emis[i])
    
for domain in domains:
    print(domain)
    preprocess_ODIAC(domain)




#%%

def preprocess_BiomeBGC(domain, varb):
    year  = 2021
    dates = [datetime(year,1,1) + timedelta(days=i) for i in range(365)]
    lons_name, lats_name = inv.getLonLatNAME(domain)
    lons_in, lats_in, emis_in = rp.read_BiomeBGC_year(year, varb, timezone='NZST')
    print(lons_in.shape,lats_in.shape,emis_in.shape)
    emis_out = regrid_UrbanVPRM(lons_in, lats_in, emis_in, lons_name, lats_name)
    write_preprocessed_BiomeBGC(dates, emis_out, domain, varb)
    
def write_preprocessed_BiomeBGC(dates, emis, domain, varb):
    path_out = path_out_prepr + '/BiomeBGC/%s/%s/'%(varb,domain)
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    fname_fmt = path_out+'emis_%Y%m%d.npy'
    for i,date in enumerate(dates):
        np.save(date.strftime(fname_fmt), emis[i])

for varb in ['Re','GEE']:
    for domain in domains:
        print(domain,varb)
        preprocess_BiomeBGC(domain, varb)
    
#%%

domain = 'Mah0p3'
lons_name, lats_name = inv.getLonLatNAME(domain)
lons_in, lats_in, emis_in = rp.read_BiomeBGC_NEE(2022, timezone='NZST')
emis_out = regrid_UrbanVPRM(lons_in, lats_in, emis_in, lons_name, lats_name)
    
#%%

def preprocess_BiomeBGC_monthly(domain):
    year  = 2022
    dates = [datetime(year,1,1) + timedelta(days=i) for i in range(365)]
    lons_name, lats_name = inv.getLonLatNAME(domain)
    lons_in, lats_in, emis_in = rp.read_BiomeBGC_NEE(year, timezone='NZST')
    dates_m, emis_in_m = calc_monthly_averages(dates, emis_in)
    emis_out = regrid_UrbanVPRM(lons_in, lats_in, emis_in_m, lons_name, lats_name)
    write_preprocessed_BiomeBGC_monthly(dates_m, emis_out, domain)

def write_preprocessed_BiomeBGC_monthly(dates, emis, domain):
    # We write one file per day
    path_out = path_out_prepr + '/BiomeBGC_monthly/NEE/%s/'%domain
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    fname_fmt = path_out+'emis_%Y%m.npy'
    for i,date in enumerate(dates):
        fname = date.strftime(fname_fmt)
        print(fname)
        np.save(fname, emis[i])

for domain in domains:
    print(domain)
    preprocess_BiomeBGC_monthly(domain)
    
    


#%%

from netCDF4 import Dataset
import matplotlib.pyplot as plt

d = Dataset("%s/priors/MahuikaAuckland/industrial_hourly_500m_CO2ff_2016_weekday.nc"%path_inv_base)

dates = [datetime(2021,1,1)]

domain = 'Mah0p3'
lons_name, lats_name = inv.getLonLatNAME(domain, bounds=False)

cat = 'onroad_CO2ff'
lons_mah, lats_mah, mah = rp.read_mahuika_onecat(cat, unit='kg/m2/s')
mah = regrid_mahuika_onecat(domain, lons_mah, lats_mah, mah)


lons_in, lats_in, urbanVPRM = rp.read_UrbanVPRM(dates)
urbanVPRM = regrid_UrbanVPRM(lons_in, lats_in, urbanVPRM, lons_name, lats_name)

lonb = lons_name.min()-0.1, lons_name.max()+0.1
latb = lats_name.min()-0.1, lats_name.max()+0.1
lons_in, lats_in, odiac = rp.read_ODIAC(dates, lonb, latb)
odiac = regrid_ODIAC(lons_in, lats_in, odiac, lons_name, lats_name)

lon_ed, lat_ed, edgar = rp.read_EDGARv7(year=2020)
edgar = regrid_ODIAC(lon_ed, lat_ed, edgar, lons_name, lats_name)





