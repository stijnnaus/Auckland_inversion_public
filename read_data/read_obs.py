#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for reading CO2 and wind data
"""

import numpy as np
from datetime import datetime, timedelta


def get_path_obs():
    from base_paths import path_obs_co2
    return path_obs_co2

def get_filename_site(site, median=False):
    path = get_path_obs()
    if not median:
        fnames = {'win':'WIN_co2_all_hours_nzst.txt', 
                  'mkh':'MKH_co2_all_hours_nzst.txt',
                  'bhd':'bhco2_merged_all_hours.txt', 
                  'lau':'lauder_co2_hourly_insitu_2014-2020.txt',
                  'pur':'nwo_hour_avg_calibrated_data_flagged.cszv',
                  'nwo':'nwo_hour_avg_calibrated_data_flagged.csv',
                  'aut':'aut_hour_avg_calibrated_data_flagged.csv', 
                  'rbm':'RBM_140201_230223_all_x2007.lhr',
                  'tka':'none',
                  'lau_licor':'LDR_080611_220413_x2007.lhr'}
    else:
        fnames = {'mkh':'MKH_co2_all_hours_nzst_wmedian.txt',
                  'pur':'nwo_hour_median_calibrated_data_flagged.cszv',
                  'nwo':'nwo_hour_median_calibrated_data_flagged.csv',
                  'aut':'aut_hour_median_calibrated_data_flagged.csv', }
        
    fnames = {site:path+fname for site,fname in fnames.items()}
    print(site,fnames[site])
    return fnames[site]


def readDataSites(sites, median=False):
    dates, co2, co2_sd = {}, {}, {}
    for site in sites:
        out = readSpecificSite(site, median)
        if out!='na':
            dates[site],co2[site],co2_sd[site] = out
    return dates, co2, co2_sd

def readSpecificSite(site, median=False):
    # Different sites have different file formats, so we need different functions to read them
    site = site.lower()
    if   site in ['bhd','mkh','win']:
        return readDataFMT_bhd(site, median)
    elif site in ['lau']:
        return readDataFMT_lau(site, median)
    elif site in ['nwo','pur','aut']:
        return readDataFMT_nwo(site, median)
    elif site in ['tka']:
        return readDataFMT_tka(site, median)
    else:
        print('Site not found: %s'%site)
        return 'na'
        
def readDataFMT_bhd(site, median=False):
    dates,co2s,co2_sd = [],[],[]
    with open(get_filename_site(site, median)) as f:
        lines = f.readlines()
        for line in lines[2:-1]:
            line = line.split()
            date = datetime.strptime(line[0]+' '+line[1],'%Y-%m-%d %H:%M:%S')
            
            if median:
                co2 = float(line[3])
                
            else:
                co2 = float(line[2])
            
            co2_sd_i = line[5]
            co2_sd_i = np.nan if co2_sd_i=='NA' else float(co2_sd_i)
            
            dates.append(date)
            co2s.append(co2)
            co2_sd.append(co2_sd_i)
            
    return np.array(dates),np.array(co2s),np.array(co2_sd)
    
def readDataFMT_lau(site, median=False):
    dates, co2s, co2_sd = [], [], []
    with open(get_filename_site(site, median)) as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.split()
            y,m,d,h = [int(l) for l in line[1:5]]
            dates.append(datetime(y,m,d,h))
            co2s.append(float(line[13]))
            co2_sd.append(float(line[14]))
    return np.array(dates),np.array(co2s),np.array(co2_sd)

def readDataFMT_nwo(site, median=False):
    dates, co2s, co2_sd = [], [], []
    with open(get_filename_site(site, median)) as f:
        for line in f.readlines()[1:]:
            line = line.split(',')
            date = datetime.strptime(line[0],'"%Y-%m-%d %H:%M:%S"') 
            date = date + timedelta(hours=12) # UTC to NZST
            dates.append(date)
            
            co2i = line[2]
            co2i_sd = line[1]
            if co2i == 'NA':
                co2s.append(np.nan)
                co2_sd.append(np.nan)
            else:
                co2s.append(float(co2i))
                co2_sd.append(float(co2i_sd))
                
    return np.array(dates),np.array(co2s),np.array(co2_sd)

def readDataFMT_tka(site, median=False):
    # No data yet available, so placeholder
    dates, co2s, co2_sd = [], [], []
    return np.array(dates), np.array(co2s), np.array(co2_sd)

def read_winddata(dates, site, timezone='nzst', daily_or_hourly='daily'):
    cliflo_sites = list(get_cliflo_site_ids().keys())
    site = site.lower()
    if site=='mkh':
        windfunc = read_mkh_winddata_all
    elif site=='aklairp' or site=='aucklandairport':
        windfunc = read_aucklairp_winddata_all
    elif site in cliflo_sites:
        windfunc = read_winddate_cliflo_all
    else:
        raise KeyError("Unknown winddata location: %s"%site)
        
    dates_all, ws_all, wd_all = windfunc(site)
    if timezone=='utc':
        # I've set the windfunc to read winddata in NZST, so if we want UTC we can convert 
        dates_all = dates_all - timedelta(12*3600) # NZST to UTC
        
    # Either for each date in dates read the whole day, or assume each date in dates
    # describes a specific hour and so only read those hours
    if daily_or_hourly=='daily':
        ws,wd = select_winddata_dates_daily(dates, dates_all, ws_all, wd_all)
    elif daily_or_hourly=='hourly':
        ws,wd = select_winddata_dates_hourly(dates, dates_all, ws_all, wd_all)
    
    return ws,wd

def get_cliflo_site_ids():
    # Sites that I retrieved from cliflo, with the agent ID
    site_ids = {}
    site_ids['pukekohe']    = 2006 
    site_ids['skytower_nw'] = 39915
    site_ids['skytower_se'] = 44068
    site_ids['motat']       = 41351
    site_ids['mangere']     = 43711
    site_ids['albany']      = 37852
    #site_ids['mkh2']         45217 # This one is also in the file that Sylvia sent me
    return site_ids

def read_mkh_winddata_all(site):
    fname = '%s/mkh_hourly_wind_20201022-20240324.csv'%get_path_to_meteo_obs()
    dates, ws, wd = [], [], []
    with open(fname) as f:
        for line in f.readlines()[1:]:
            line = line.split(',')
            date_str = '%sT%s'%(line[1], line[2])
            date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
            dates.append(date)
            ws.append(float(line[3]))
            wd.append(float(line[4]))
            
    return np.array(dates), np.array(ws), np.array(wd)

def read_aucklairp_winddata_all(site):
    knots_to_ms = 0.514
    fname = '%s/winddata_AucklAirp.csv'%get_path_to_meteo_obs()
    dates, ws, wd = [], [], []
    with open(fname) as f:
        for line in f.readlines()[1:]:
            line = line.split(',')
            dates.append(datetime.strptime(line[1],'%Y-%m-%d %H:%M'))
            wsi = line[6]
            wdi = line[5]
            if wsi=='null' or wdi=='null':
                ws.append(np.nan)
                wd.append(np.nan)
            else:
                ws.append(float(wsi)*knots_to_ms) # in knots!
                wd.append(float(wdi))
                
    # UTC to NZST
    dates = [d+timedelta(seconds=12*3600) for d in dates]
    return np.array(dates), np.array(ws), np.array(wd)

def get_path_to_meteo_obs():
    from base_paths import path_obs_meteo
    return path_obs_meteo

def read_winddate_cliflo_all(site):
    # Data from the cliflo sites. All data are in one file and each site has an agent ID
    # by which it is sorted. So to get data for a specific site, I need to get site ID
    site_id = get_cliflo_site_ids()[site]
    fname = '%s/winddata_cliflo.txt'%get_path_to_meteo_obs()
    
    data_all = np.genfromtxt(fname,skip_header=16,skip_footer=8,dtype='object')
    site_id_all = np.array(data_all[:,0], dtype=int)
    site_mask = (site_id_all==site_id)
    
    dates = np.array([datetime.strptime(d.decode('utf-8'), "%Y%m%d:%H%M") for d in data_all[:,1]])[site_mask]

    wd = np.array(data_all[:,2])[site_mask]
    ws = np.array(data_all[:,3])[site_mask]
    return dates, ws, wd
    
def select_winddata_dates_daily(dates_sel, dates_all, ws_all, wd_all):
    '''
    From the full observational dataset, select only the requested dates. 
    In this function, dates_sel are assumed to be daily dates, and all hours
    for each date are kept.
    '''
    ws_sel = np.zeros((len(dates_sel),24), dtype=float)
    wd_sel = np.zeros((len(dates_sel),24), dtype=float)
    for i,date in enumerate(dates_sel):
        for ihour,hour in enumerate(range(0,24)):
            date_i = date + timedelta(seconds=3600*hour)
            ws_sel[i,ihour], wd_sel[i,ihour] = find_specific_ws_wd_from_date(date_i, dates_all, ws_all, wd_all)
    
    return ws_sel, wd_sel

def select_winddata_dates_hourly(dates_sel, dates_all, ws_all, wd_all):
    '''
    Same as daily, but dates_sel here are assumed to be specific hours that 
    are requested, so not all 24 hours for each dates_sel are read.
    '''
    
    ws_sel = np.zeros(len(dates_sel), dtype=float)
    wd_sel = np.zeros(len(dates_sel), dtype=float)
    for i,date in enumerate(dates_sel):
        ws_sel[i], wd_sel[i] = find_specific_ws_wd_from_date(date, dates_all, ws_all, wd_all)

    return ws_sel, wd_sel

def find_specific_ws_wd_from_date(date, dates_all, ws_all, wd_all):
    if date in dates_all:
        idate = list(dates_all).index(date)
        ws = ws_all[idate]
        wd = wd_all[idate]
    else:
        print(date, 'not found in obs data')
        ws = np.nan
        wd = np.nan
        
    return ws,wd

def calculate_obs_gradient(dates_1, co2_1, dates_2, co2_2):
    '''
    Subtract co2 data from two sites from each other. Note that dates_1 and dates_2
    might not be the same and we subtract only on dates that are available for both.
    '''
    
    mask_1 = [d in dates_2 for d in dates_1]
    mask_2 = [d in dates_1 for d in dates_2]
    
    dates_mixed = dates_1[mask_1]
    dco2 = co2_1[mask_1] - co2_2[mask_2]
    
    return dates_mixed, dco2

def get_site_coords(site):
    
    coords = {}
    
    # CO2
    coords['MKH'] = [174.5615, -37.0507]
    coords['TKA'] = [174.7993, -36.8265]
    coords['AUT'] = [174.7657, -36.8543]
    coords['NWO'] = [174.8239, -36.8621]
    
    # Met sites
    coords['MOTAT']   = [174.71185, -36.86297]
    coords['AklAirp'] = [174.78873, -37.00813]
    coords['Mangere'] = [174.78052, -36.96480]
    coords['Sky']     = [174.76242, -36.85004]   
    
    return coords[site]
    
    
    
    
    
