#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 14:24:19 2023

@author: nauss
"""

from netCDF4 import Dataset
import numpy as np
from numpy import newaxis as nax
from calendar import monthrange as mor
import xarray as xr
import calendar
import os
import functions_stijn as st
from datetime import datetime,timedelta
import inversion as inv

def get_path_priors():
    from base_paths import path_priors_raw
    return path_priors_raw

def read_mahuika_onecat(cat, unit='kg/hr', year=2016):
    filename = get_filename_from_cat_mah(cat, year)
    with Dataset(filename, 'r') as d:
        lons = d['longitude'][:]
        lats = d['latitude'][:]
        if cat == 'onroad_CO2ff':
            # It's in tonnes/hour instead of kg/hour for some reason, and the variable name is accordingly different
            emi  = d['CO2ff_annual'][:]
            emi *= 1000 # ton/hr to kg/hour
            emi  = np.swapaxes(emi,0,1) # The only category with dimensions (lon,lat) instead of (lat,lon)....
        else:
            fueltype = get_fueltype_from_cat_mah(cat)
            emi  = d['%s_hourly'%fueltype][:] # kg/hour
        
        # Fill values to 0
        emi = np.array(emi)
        emi[emi>1e30] = 0.0 
        emi[np.isnan(emi)] = 0.0 
        
    if unit =='kg/m2/s':
        emi /= 3600.
        area_per_gridcell = st.calc_area_per_gridcell(lats,lons)
        if   len(emi.shape)==3:
            emi /= area_per_gridcell[:,:,np.newaxis]
        elif len(emi.shape)==4: # Weekday / weekend
            emi /= area_per_gridcell[:,:,np.newaxis,np.newaxis]
        
    return lons, lats, emi

def get_freq_from_cat_mah(cat):    
    if cat in ['air_transport_CO2ff','commercial_CO2ff','industrial_CO2bio','sea_transport_CO2ff']:
        return 'fixed'
    elif cat in ['industrial_glenbrook_CO2ff','industrial_CO2ff']:
        return 'weekday1'
    elif cat in ['onroad_CO2ff']:
        return 'weekday2'
    elif cat in ['residential_CO2bio','residential_CO2ff']:
        return 'daily'
    else:
        raise KeyError("Unknown category for MahuikaAuckland emissions: %s"%cat)
    
def get_fueltype_from_cat_mah(cat):
    return cat.split('_')[-1]

def get_all_mahuika_cats():
    return [ 'sea_transport_CO2ff', 'air_transport_CO2ff','commercial_CO2ff','industrial_CO2bio','industrial_glenbrook_CO2ff', \
            'industrial_CO2ff','onroad_CO2ff','residential_CO2bio','residential_CO2ff']
    
def get_filename_from_cat_mah(cat, year):
    fnames_per_cat = {}
    fnames_per_cat['air_transport_CO2ff']        = 'air_transport_hourly_500m_CO2ff_2016_24hrs.nc'
    fnames_per_cat['commercial_CO2ff']           = 'commercial_hourly_500m_CO2ff_2016_24hrs.nc'
    fnames_per_cat['industrial_CO2bio']          = 'industrial_hourly_500m_CO2bio_2016_24hrs.nc'
    fnames_per_cat['industrial_glenbrook_CO2ff'] = 'industrial_glenbrook_hourly_500m_CO2ff_2016_weekday.nc'
    fnames_per_cat['industrial_CO2ff']           = 'industrial_hourly_500m_CO2ff_2016_weekday.nc'
    fnames_per_cat['residential_CO2bio']         = 'residential_hourly_500m_CO2bio_%4.4i.nc'%year
    fnames_per_cat['residential_CO2ff']          = 'residential_hourly_500m_CO2ff_%4.4i.nc'%year
    fnames_per_cat['onroad_CO2ff']               = 'onroad_hourly_500m_CO2ff_2016.nc'
    fnames_per_cat['sea_transport_CO2ff']        = 'sea_transport_hourly_500m_CO2ff_2016_24hrs.nc'
    
    path_mah = get_path_priors() + '/MahuikaAuckland/'
    return path_mah+fnames_per_cat[cat]


def read_UrbanVPRM(dates, varb='NEE'):
    path = get_path_priors()
    fname_fmt = path + 'urbanVPRM_fluxes/%m/'+varb+'_DBF_2018_%m_%d.nc'
    
    lons, lats, _ = read_UrbanVPRM_onefile(dates[0].strftime(fname_fmt), varb)
    emis = np.zeros((len(dates), 24, len(lats), len(lons)))
    for i,date in enumerate(dates):
        fname = date.strftime(fname_fmt)
        
        if varb=='Re' and date==datetime(2018,1,1):
            # This file is corrupted somehow, and contains only 11 hours. So just
            # read the next day for now
            fname = datetime(2018,1,2).strftime(fname_fmt)
        
        _, _, emis[i] = read_UrbanVPRM_onefile(fname, varb)
    return lons, lats, emis

def read_UrbanVPRM_onefile(fname, varb='NEE'):

    with Dataset(fname) as d:
        lons = d['longitude'][:]
        lats = d['latitude'][:]
        emis = np.array(d[varb][:]) # micromoles m-2 s-1, shape (hour,lat,lon)

    # Latitude is from North to South, so flip it (also necessary for regridding)
    emis = emis[:,::-1,:]
    lats = lats[::-1]
    # Replace fill values
    emis[emis>1e5] = 0.0 
    # Output units should be kgCO2 m-2 s-1
    xmco2 = 44.01e-3 # kg/mol
    emis = emis*xmco2*1e-6 # micromoles to kg

    return lons,lats,emis

def read_EDGARv7(year=2021):
    path = get_path_priors() + '/EDGARv7/'
    fname = '%s/v7.0_FT2021_CO2_excl_short-cycle_org_C_%4.4i_TOTALS.0.1x0.1.nc'%(path,year)
    with Dataset(fname) as d:
        lon, lat = d['lon'][:], d['lat'][:]
        emi = d['emi_co2'][:] # kg m-2 s-1
    emi.fill_value = 0.0
    return lon, lat, emi

def read_EDGARv8_allcats(year):
    all_cats = get_all_EDGARv8_cats()
    
    emis = 0.
    for cat in all_cats:
        lon, lat, emii = read_EDGARv8_onecat(cat, year)
        emis += emii
        
    return lon, lat, emis

def get_all_EDGARv8_cats():
    return ['AGRICULTURE','BUILDINGS','FUEL_EXPLOITATION','IND_COMBUSTION','IND_PROCESSES','POWER_INDUSTRY','TRANSPORT','WASTE']

def read_EDGARv8_onecat(cat, year):
    path  = '%s/EDGARv8/'%get_path_priors()
    fname = '%s/v8.0_FT2022_GHG_CO2_%4.4i_%s_flx.nc'%(path, year, cat)
    with Dataset(fname) as d:
        lon = d['lon'][:]
        lat = d['lat'][:]
        emi = d['fluxes'][:] # kg m-2 s-1
        emi.fill_value = 0.0
        emi = np.array(emi)
        
    return lon, lat, emi

def get_path_ODIAC():
    return get_path_priors() + '/ODIAC/'

def read_ODIAC(dates_months, lonbounds=[-180,180], latbounds=[-90,90], timezone='NZST'):
    '''
    Read fossil fuel emission data from the ODIAC emission inventory. 
    We construct a timeseries of emission on a 1 by 1 km grid from:
        a) The monthly emission distribution 1x1km ; all except aviation/marine
        b) Monthly aviation/marine emission data (1x1 degree)
        c) Fixed scalars for diurnal variability (1x1 degree)
    Because a) is a lot of data, we only read those data within certain lon, lat bounds.
    '''    
    
    # Read spatial distribution; one per month; 1x1 km2
    emifields = []
    for i,date in enumerate(dates_months):
        lon1km, lat1km, emii = read_ODIAC_emission_distribution_1km(date, lonbounds, latbounds) # [kgCO2/m2/s]
        emifields.append(emii)
    emifields = np.array(emifields)
    
    area1km = st.calc_area_per_gridcell(lat1km, lon1km)
    spm = np.array([mor(d.year,d.month)[1]*24*3600 for d in dates_months])
    etot1 = np.sum(emifields*area1km*spm[:,nax,nax]/1e6) # kton CO2 over whole period
    
    # Read aviation and marine emissions; yearly files; monthly data; 1x1 degree
    emifields_other = np.zeros((len(dates_months), 180, 360))
    for i,date in enumerate(dates_months):
        lon1d, lat1d, emifields_other[i] = read_ODIAC_other_emissions(date)  # [kgCO2/m2/s]
    # Regrid to 1km grid and combine
    regridder = st.make_xesmf_regridder(lon1d, lat1d, lon1km, lat1km)
    emifields_other = regridder(emifields_other)
    emifields += emifields_other
    
    etot2 = np.sum(emifields*area1km*spm[:,nax,nax]/1e6) # kton CO2 over whole period
    
    # Read diurnal scaling factors
    lon_diur, lat_diur, scaling_diur = read_ODIAC_diurnal_scaling(timezone) # Unitless
    regridder = st.make_xesmf_regridder(lon_diur, lat_diur, lon1km, lat1km)
    scaling_diur = regridder(scaling_diur)
    emifields = emifields[:,nax,:,:] * scaling_diur[nax,:,:,:]
    
    etot3 = np.sum((emifields.mean(axis=1))*area1km*spm[:,nax,nax]/1e6) # kton CO2 over whole period
    
    print("Totals: %4.4f ; %4.4f ; %4.4f kt"%(etot1,etot2,etot3))
    
    return lon1km, lat1km, emifields

def read_ODIAC_emission_distribution_1km(date, lonbounds, latbounds):
    
    fname = get_path_ODIAC() + date.strftime('/%Y/odiac2022_1km_excl_intl_%y%m.tif')
    xds = xr.open_dataset(fname, engine='rasterio')
    
    xds = xds.isel(y=slice(None,None,-1)) # Latitude is inverted in file
    xds = xds.sel(x=slice(*lonbounds), y=slice(*latbounds))
    lon = np.array(xds['x'])
    lat = np.array(xds['y'])
    emi = np.array(xds['band_data'])[0] # [ton C month-1 cell-1]
    
    # Unit conversion to kgCO2/m2/s
    c_to_co2     = (44.01/12.01)
    ton_to_kg    = 1e3
    area         = st.calc_area_per_gridcell(lat,lon)
    month_to_sec = mor(date.year,date.month)[1]*24*3600
    emi = emi*c_to_co2*ton_to_kg/area/month_to_sec
    
    return lon,lat,emi
    
def read_ODIAC_other_emissions(date):
    '''
    From the annual, 1 by 1 degree .nc files, extract one month of aviation+
    marine emission data.
    '''
    
    fname = get_path_ODIAC() + date.strftime('/%Y/odiac2022_1x1d_%Y.nc')
    xds = xr.open_dataset(fname)
    lon = np.array(xds['lon'])
    lat = np.array(xds['lat'])
    emi = np.array(xds['intl_bunker']) # gC/m2/d
    emi = emi[date.month-1]
    emi *= (44.01/12.01)/(24*3600)*1e-3 # kgCO2/m2/s
    
    return lon, lat, emi

def read_ODIAC_diurnal_scaling(timezone='NZST'):
    '''
    Read the fixed diurnal cycle at 0.5 degree resolution
    '''
    
    fname = '%s/diurnal_scale_factors.nc'%(get_path_ODIAC())
    xds = xr.open_dataset(fname)
    # Longitude and latitude are not included in the file
    lonb = np.linspace(-180,180, xds.dims['longitude']+1)
    latb = np.linspace(-90, 90  , xds.dims['latitude']+1)
    lonc = 0.5*(lonb[1:]+lonb[:-1])
    latc = 0.5*(latb[1:]+latb[:-1])
    scaling = np.array(xds['diurnal_scale_factors']) # unitless
    scaling = np.rollaxis(scaling, 2, 0) # Time dimension to front
    
    if timezone.lower()=='utc':
        pass
    elif timezone.lower()=='nzst':
        scaling = np.roll(scaling, 12, axis=0)
    else:
        raise KeyError("Unknown timezone for reading ODIAC diurnal scaling: %s"%timezone)
    
    return lonc, latc, scaling
        
def read_BiomeBGC_year(year, varb, timezone='NZST'):
    '''
    Loop over monthly files.
    '''
    
    # Read example for nlat, nlon
    lons, lats, _ = read_BiomeBGC_onemonth(year, 1, varb, timezone=timezone)
    
    nday_in_year = st.get_nday_in_year(year)
    emis_year = np.zeros((nday_in_year, 24, len(lats), len(lons)))
    idx = 0
    for month in range(1,13):
        nday_in_month = calendar.monthrange(year, month)[1]
        _,_,emis_month = read_BiomeBGC_onemonth(year, month, varb, timezone)
        
        emis_year[idx:(idx+nday_in_month)] = emis_month
        idx += nday_in_month
        
    return lons, lats, emis_year
        
def read_BiomeBGC_onemonth(year, month, varb, timezone='NZST'):
    '''
    These are Daemon's processed files. Monthly files with hourly data in UTC.
    One file for all variables (NEE, GEE, Re, etc.)
    '''
    
    # Variable names different in BiomeBGC than in UrbanVPRM
    if varb == 'GEE':
        varb = 'GPP'
    elif varb == 'Re':
        varb = 'ER'
    
    path = '%s/BiomeBGC/'%get_path_priors()
    fname = '%s/flux_%4.4i%2.2i.nc'%(path, year, month)
    
    d = Dataset(fname)
    flux = np.array(d[varb]) # (time, nlat, nlon) / gCO2 m-2 s-1
    flux[flux==1e20] = 0.    # Fill values to zero
    flux /= 1e3              # kgCO2 m-2 s-1
    
    lon, lat = d['lon'][:], d['lat'][:]
    
    if timezone=='UTC':
        pass
    elif timezone=='NZST':
        flux = np.roll(flux, 12, axis=0)
    
    # (nhour) to (nday,24)
    nhour,nlat,nlon = flux.shape
    nday = int(nhour/24)
    flux = flux.reshape(nday, 24, nlat, nlon)
    
    return lon, lat, flux

def read_preprocessed_1inventory_totals(inventory, dates, domain, cat=None, overwrite=False):
    """
    I don't have emissions in my inverse system because I pre-calculate enhancements, 
    but sometimes I want to do something with emissions. Here, I retrieve emission 
    totals (i.e., spatial total) per date (hours). Because reading totals is much quicker 
    than emission maps, I set it up such that I only calculate totals once.
    
    I set this up initially because I wanted to perturb the diurnal cycle in the OSSE,
    and scaling the prior by emission totals seemed the easiest way to do it (compared
    to recalculating enhancements for each perturbation).
    
    cat is the category for Mahuika-Auckland to read
    
    Returns units kg/year
    """
    
    dates = np.array(dates)
    inventory = inventory.lower()
    
    # One file per year
    uyears = np.unique( [d.year for d in dates] )
    fname_fmt = get_filename_fmt_emis_totals(inventory, domain, cat)
    
    emis_tots = np.zeros(len(dates), dtype=float)
    for i,year in enumerate(uyears):
        fname        = fname_fmt%year
        nday_in_year = 365 + calendar.isleap(year)
        dates_year   = [datetime(year,1,1) + timedelta(days=i) for i in range(nday_in_year)]
        
        if not os.path.isfile(fname) or overwrite:
            # If annual file not yet available, we make it
            print('Hourly emission totals for %i not available, need to calculate first'%year)
            emis_tots_year = calculate_from_preprocessed_1inventory_totals(inventory, dates_year, domain, cat)
            np.save(fname, emis_tots_year)
        else:
            # Otherwise just load prepared totals
            emis_tots_year = np.load(fname)
            
        # Fill out the specific days requested
        for d in dates[[d.year==year for d in dates]]:
            idx1 = np.where(dates==d)[0][0]
            emis_tots[idx1] = emis_tots_year[d.day-1, d.hour]
        
    return emis_tots

def get_filename_fmt_emis_totals(inventory, domain, cat):
    """
    Get filename format for the annual emission total files. It's a format in the
    sense that the year still needs to be filled in.
    """
    
    path = '%s/hourly_totals/'%get_path_priors()
    
    if inventory=='Mahuika-Auckland':
        if cat is None or cat=='all':
            cat = 'tot'
        fname = '%s/%s_%s_%s'%(path, inventory, domain, cat)
        
    else:
        fname = '%s/%s_%s'%(path, inventory, domain)
        
    fname += '_%i.npy' # The year that can be filled in later
    return fname

def calculate_from_preprocessed_1inventory_totals(inventory, dates, domain, cats):
    '''
    Calculate hourly emission totals from preprocessed emission fields (kg/hour).
    Input dates are days and then we read all 24 hours
    '''
    
    lons, lats = inv.getLonLatNAME(domain, bounds=True)
    areas = st.calc_area_per_gridcell(lats, lons, bounds=True) # m2
    emis_fields = read_preprocessed_1inventory(inventory, dates, domain, cats) # kg/m2/s
    
    emis_tots = emis_fields * areas * 3600
    emis_tots = emis_tots.sum(axis=(-1,-2))
    return emis_tots
    

def read_preprocessed_1inventory(inventory, dates, domain, cats=None, varb='NEE'):
    '''
    Creates functionality to give inventory name as input argument rather than
    having to call the inventory-specific function; can be handy in e.g., loops.
    All in [kg/m2/s], output shape (days, hours, lats, lons)
    
    Note that cats is a variable that is  applicable to Mahuika (traffic, industrial etc)
    whereas varb is for biosphere (NEE, respiration/Re, etc).
    '''
    
    # Something in here is modifying the variable "dates", which is a problem for functions that
    # call read_preprocessed_1inventory. I can't figure out where it's getting modified, so I'm
    # just fixing it like this:
    dates = np.copy(dates)
    
    inventory = inventory.lower()
    
    if inventory[-10:] == '_nodiurnal':
        # I don't actually process emissions with diurnal cycle removed, I just read the original emissions
        # and then remove the diurnal cycle
        inventory = inventory[:-10]
        remove_diurnal = True
    else:
        remove_diurnal = False
    
    if   inventory=='mahuikaauckland' or inventory=='mahuika-auckland':
        emis = read_preprocessed_mahuika(cats, dates, domain)
    elif inventory=='mahuikaauckland_new_x':
        emis = read_preprocessed_new_x_or_t(dates, domain, prior='MahuikaAuckland',label='x')
    elif inventory=='mahuikaauckland_new_t':
        emis = read_preprocessed_new_x_or_t(dates, domain, prior='MahuikaAuckland',label='t')
    elif inventory=='urbanvprm':
        emis = read_preprocessed_urbanvprm(dates, domain, varb=varb)
    elif inventory=='urbanvprm_new_x':
        emis = read_preprocessed_new_x_or_t(dates, domain, prior='UrbanVPRM',label='x', varb=varb)
    elif inventory=='urbanvprm_new_t':
        emis = read_preprocessed_new_x_or_t(dates, domain, prior='UrbanVPRM',label='t', varb=varb)
    elif inventory=='urbanvprm_monthly':
        emis = read_preprocessed_urbanvprm_monthly(dates, domain)
    elif inventory=='biomebgc':
        emis = read_preprocessed_biomebgc(dates, domain, varb=varb)
    elif inventory=='biomebgc_monthly':
        emis = read_preprocessed_biomebgc_monthly(dates, domain, varb=varb)
    elif inventory=='edgarv7':
        emis = read_preprocessed_edgarv7(dates, domain)
    elif inventory=='edgarv8':
        emis = read_preprocessed_edgarv8(dates, domain)
    elif inventory=='odiac':
        emis = read_preprocessed_odiac(dates, domain)
    else:
        raise KeyError("Unknown inventory name for reading prepr emis: %s"%(inventory))
        
    if remove_diurnal:
        emis[:,:,:,:] = emis.mean(axis=1)[:,np.newaxis,:,:]
        
    return emis
        
def read_preprocessed_mahuika(cats, dates, domain):
    path = get_path_priors() + '/preprocessed/MahuikaAuckland/%s/'%domain
    lons, lats = inv.getLonLatNAME(domain)
    emis = np.zeros((len(dates), 24, len(lats), len(lons)))
    if cats is None:
        raise ValueError("If reading Mahuika, cats need to be prescribed!!!")
    
    if cats=='all' or cats=='tot':
        cats = get_all_mahuika_cats()
        
    for cat in cats:
        emis += read_preprocessed_mahuika_onecat(path, cat, dates, lats, lons)
    return emis

def read_preprocessed_mahuika_onecat(path, cat, dates, lats, lons):
    freq = get_freq_from_cat_mah(cat)
    path = '%s/%s/'%(path,cat)
    emis = np.zeros((len(dates), 24, len(lats), len(lons)))
    if freq=='fixed':
        emis[:] = np.load('%s/emis_fixed.npy'%(path))
    elif freq=='weekday1' or freq=='weekday2':
        idx_weekday, idx_weekend = st.get_idx_weekday_weekend(dates)
        emis[idx_weekday] = np.load('%s/emis_weekday.npy'%path)
        emis[idx_weekend] = np.load('%s/emis_weekend.npy'%path)
    elif freq=='daily':
        for i,date in enumerate(dates):
            date = datetime(date.year,date.month,date.day)
            fname = date.strftime('emis_%Y%m%d.npy')
            emis[i] = np.load('%s/%s'%(path,fname))
    else:
        raise ValueError("Unknow freq %s"%freq)
            
    return emis

def read_preprocessed_new_x_or_t(dates, domain, prior='MahuikaAuckland', label='x', varb=None):
    if prior=='MahuikaAuckland':
        path = get_path_priors() + '/preprocessed/%s_new_%s/%s/'%(prior, label, domain)
    elif prior=='UrbanVPRM':
        if varb is None:
            varb = 'NEE'
        path = get_path_priors() + '/preprocessed/%s_new_%s/%s/%s/'%(prior, label, domain, varb)
        
    lons, lats = inv.getLonLatNAME(domain)
    emis = np.zeros((len(dates),24,len(lats),len(lons)))
    for i,date in enumerate(dates):
        fname = date.strftime('emis_%Y%m%d.npy')
        emis[i] = np.load('%s/%s'%(path,fname))
    return emis

def read_preprocessed_urbanvprm(dates, domain, varb='NEE'):
    if varb is None:
        varb = 'NEE'
    path = get_path_priors() + '/preprocessed/UrbanVPRM/%s/%s/'%(varb,domain)
    lons, lats = inv.getLonLatNAME(domain)
    emis = np.zeros((len(dates),24,len(lats),len(lons)))
    for i,date in enumerate(dates):
        # I only have 2018 UrbanVPRM data!
        date = datetime(2018,date.month,date.day)
        fname = date.strftime('emis_%Y%m%d.npy')
        emis[i] = np.load('%s/%s'%(path,fname))
    return emis

def read_preprocessed_urbanvprm_monthly(dates, domain, varb='NEE'):
    if varb is None:
        varb = 'NEE'
    path = get_path_priors() + '/preprocessed/UrbanVPRM_monthly/%s/%s/'%(varb,domain)
    lons, lats = inv.getLonLatNAME(domain)
    emis = np.zeros((len(dates),24,len(lats),len(lons)))
    unique_months = np.unique([[d.year,d.month] for d in dates], axis=0)
    for umo in unique_months:
        # I only have 2018 UrbanVPRM data!
        date = datetime(2018,umo[1],1)
        mask = [((d.year==umo[0]) & (d.month==umo[1])) for d in dates]
        fname = date.strftime('emis_%Y%m.npy')
        emis[mask] = np.load('%s/%s'%(path,fname))
    return emis

def read_preprocessed_edgarv7(dates, domain):
    path = get_path_priors() + '/preprocessed/EDGARv7/%s/'%domain
    lons, lats = inv.getLonLatNAME(domain)
    # Only 2021
    path = '%s/emis_2021.npy'%(path)
    emis = np.zeros((len(dates), 24, len(lats), len(lons)))
    emis[:,:] = np.load(path)
    return emis

def read_preprocessed_edgarv8(dates, domain):
    path = get_path_priors() + '/preprocessed/EDGARv8/%s/'%domain
    lons, lats = inv.getLonLatNAME(domain)
    # Unique months
    umonths = np.unique([datetime(d.year,d.month,1) for d in dates])
    emis = np.zeros((len(dates),24, len(lats), len(lons)))
    for month in umonths:
        mask = [datetime(d.year,d.month,1)==month for d in dates]
        fname = path + month.strftime('emis_%Y%m.npy')
        emis[mask] = np.load(fname)
    return emis

def read_preprocessed_odiac(dates, domain):
    path = get_path_priors() + '/preprocessed/ODIAC/%s/'%domain
    lons, lats = inv.getLonLatNAME(domain)
    # Monthly, for now only 2021
    umonths = np.unique([d.month for d in dates])
    emis = np.zeros((len(dates),24, len(lats), len(lons)))
    for month in umonths:
        mask = [d.month==month for d in dates]
        fname = path + datetime(2021,month,1).strftime('emis_%Y%m.npy')
        emis[mask] = np.load(fname)
    return emis

def read_preprocessed_biomebgc(dates, domain, varb='NEE'):
    if varb is None:
        varb = 'NEE'
    path = get_path_priors() + '/preprocessed/BiomeBGC/%s/%s/'%(varb, domain)
    lons, lats = inv.getLonLatNAME(domain)
    
    # I only have 2021,2022
    for i,date in enumerate(dates):
        if date.year<=2021:
            dates[i] = datetime(2021,date.month,date.day)
        else:
    	    dates[i] = datetime(2022,date.month,date.day)

    emis = np.zeros((len(dates),24,len(lats),len(lons)))
    for i,date in enumerate(dates):
        date = datetime(date.year, date.month, date.day)
        fname = date.strftime('emis_%Y%m%d.npy')
        emis[i] = np.load('%s/%s'%(path,fname))
    return emis

def read_preprocessed_biomebgc_monthly(dates, domain, varb='NEE'):
    if varb is None:
        varb = 'NEE'
    path = get_path_priors() + '/preprocessed/BiomeBGC_monthly/%s/%s/'%(varb,domain)
    lons, lats = inv.getLonLatNAME(domain)
    
    emis = np.zeros((len(dates), 24, len(lats), len(lons)))
    unique_months = st.find_unique_months(dates)
    for umo in unique_months:
        mask = [((d.year==umo.year) & (d.month==umo.month)) for d in dates]
        date = datetime(umo.year, umo.month, 1)
        fname = date.strftime('emis_%Y%m.npy')
        emis[mask] = np.load('%s/%s'%(path, fname))
    return emis

def get_specific_hours_from_daily_cycles(hours_needed, days_emis, emis):
    '''
    When I read preprocessed emissions, I read full daily cycles. But the hourly footprints
    generally do not start/end at start/end of days. So I select here from the hourly emissions
    for full daily cycles the specific hours I need to combine with the footprints.
     - Both "hours_needed" and "days" should be datetime objects.
     - Shape of emis is (len(days), 24, nlat, nlon).
    '''
    
    nday,_,nlat,nlon = emis.shape
    emis_sel = np.zeros((len(hours_needed), nlat, nlon))
    
    if nday!=len(days_emis):
        raise ValueError("When selecting timesteps, prescribed days have length %i, while the emission array has length %i"%(len(days_emis),nday))
    
    days_needed = [datetime(d.year,d.month,d.day) for d in hours_needed]
    for i,date_hour in enumerate(hours_needed):
        ihour = date_hour.hour
        iday  = days_emis.index(days_needed[i])
        emis_sel[i] = emis[iday,ihour]
    return emis_sel






















