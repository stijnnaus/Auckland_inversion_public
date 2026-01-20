#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 10:42:46 2024

@author: nauss
"""

import numpy as np
from inversion_base import InversionBase
import functions_stijn as st
import os
import calendar
from datetime import datetime,timedelta
import inversion as inv
import tarfile

def bin_based_on_ws_wd(bins_ws, bins_wd, ws, wd, enh):
    nbin_ws, nbin_wd = len(bins_ws)-1, len(bins_wd)-1
    
    enh_binned = np.zeros((nbin_ws,nbin_wd))
    enh_binned_sd = np.zeros((nbin_ws,nbin_wd))
    bincount = np.zeros((nbin_ws,nbin_wd))
    for ibin_ws in range(nbin_ws):
        lo, hi = bins_ws[ibin_ws], bins_ws[ibin_ws+1]
        mask_ws = (ws>lo) & (ws<=hi)
        for ibin_wd in range(nbin_wd):
            lo, hi = bins_wd[ibin_wd], bins_wd[ibin_wd+1]
            mask_wd = (wd>lo) & (wd<=hi)
            mask = mask_ws & mask_wd
            bincount[ibin_ws,ibin_wd] = mask.sum()
            if mask.sum()>5:
                enh_binned[ibin_ws,ibin_wd]    = enh[mask].mean()
                enh_binned_sd[ibin_ws,ibin_wd] = enh[mask].std() 
            else:
                enh_binned[ibin_ws,ibin_wd]    = np.nan
                enh_binned_sd[ibin_ws,ibin_wd] = np.nan 
    return enh_binned, enh_binned_sd, bincount



class RetrievorWinddataNAME(InversionBase):
    
    def __init__(self, inversion_name, input_timezone='nzst'):
        super().__init__(inversion_name)
        self.input_timezone = input_timezone
        
    def retrieve_winddata_NAME(self, dates, sites, sitename_version='model', overwrite=False):
        sites = self.adjust_sitenames_based_on_version(sites, sitename_version)
        self.process_missing_months(dates, sites, overwrite)
        ws, wd, pbl, temp, rh = self.read_processed_winddata_NAME(dates, sites)
        return ws, wd, pbl, temp, rh
    
    def adjust_sitenames_based_on_version(self, sites, sitename_version):
        '''
        NAME doesn't use the same site names as the meteo dataset. Probably more of 
        an inconsistency in my code than anything else, but it's just easier to map
        the names here.
        '''
        
        if sitename_version.lower() in ['model', 'name']:
            return sites
        elif sitename_version.lower()[:3]=='obs':
            sites_new = []
            for site in sites:
                site_new = self.get_sitename_model_from_sitename_obs(site)
                sites_new.append(site_new)
            return sites_new
                
    def get_sitename_model_from_sitename_obs(self, sitename_obs):
        sitename_obs = sitename_obs.lower()
        if sitename_obs=='mkh':
            return 'ManukauHeads'
        elif sitename_obs=='aklairp':
            return 'AucklandAirport'
        elif sitename_obs=='motat':
            return 'MOTAT'
        elif sitename_obs[:8]=='skytower': # Both NE and SW
            return 'SkyTower'
        elif sitename_obs=='mangere':
            return 'Mangere'
        elif sitename_obs=='aut':
            return 'AucklandUni'
        elif sitename_obs=='tka':
            return 'Takarunga'
        elif sitename_obs=='nwo':
            return 'Pourewa'
        else:
            raise ValueError("Unknown sitename for meteo obs site: %s!"%sitename_obs)
    
    def process_missing_months(self, dates, sites, overwrite):
        unique_months = st.find_unique_months(dates)
        for month in unique_months:
            if self.check_if_month_is_missing(month, sites) or overwrite:
                print(month.strftime("For wind data we need to process %Y-%m"))
                ws, wd, pbl, temp, rh = self.read_raw_winddata_month(month, sites)
                self.write_processed_winddata_multisite(month, sites, ws, wd, pbl, temp, rh)
            
    def check_if_month_is_missing(self, date, sites):
        exists = True
        for site in sites:
            fname = self.get_filename_processed_winddata(date, site)
            exists = exists & os.path.isfile(fname)
        missing = (not exists)
        return missing
    
    def get_filename_processed_winddata(self, date, site):
        path = self.get_path_processed_winddata()
        date_str = date.strftime('%Y%m')
        return os.path.join(path, '%s_%s.npy'%(site, date_str))
    
    def get_path_processed_winddata(self):
        path = self.get_path_inversion() + '/processed_winddata/'
        self.ensure_path_exists(path)
        return path
    
    def write_processed_winddata_multisite(self, date, sites, ws, wd, pbl, temp, rh):
        for i,site in enumerate(sites):
            self.write_processed_winddata_onefile(date, site, ws[i], wd[i], pbl[i], temp[i], rh[i])
    
    def write_processed_winddata_onefile(self, date, site, ws, wd, pbl, temp, rh):
        fname = self.get_filename_processed_winddata(date, site)
        np.save(fname, [ws,wd,pbl,temp,rh])
        
    def read_processed_winddata_NAME(self, dates, sites):
        unique_months = st.find_unique_months(dates)
        
        dates_all = []
        for month in unique_months:
            nday = calendar.monthrange(month.year,month.month)[1]
            dates_all += [datetime(month.year,month.month,1) + timedelta(seconds=3600*i) for i in range(nday*24)]
        dates_all = np.array(dates_all)
        
        # First read the full months, as we have preprocessed monthly files
        ws_all   = np.zeros((len(sites), len(dates_all)))
        wd_all   = np.zeros((len(sites), len(dates_all)))
        pbl_all  = np.zeros((len(sites), len(dates_all)))
        temp_all = np.zeros((len(sites), len(dates_all)))
        rh_all   = np.zeros((len(sites), len(dates_all)))
        i0 = 0
        for month in unique_months:
            ws_i, wd_i, pbl_i, temp_i, rh_i = self.read_processed_winddata_multisite(month, sites)
            i1 = i0 + len(ws_i[0])
            ws_all[:,i0:i1]   = ws_i
            wd_all[:,i0:i1]   = wd_i
            pbl_all[:,i0:i1]  = pbl_i
            temp_all[:,i0:i1] = temp_i
            rh_all[:,i0:i1]   = rh_i
            i0 = i1
        
        # Now select only the dates we need
        mask = [d in dates for d in dates_all]
        ws_sel   = np.array(ws_all)[:,mask]
        wd_sel   = np.array(wd_all)[:,mask]
        pbl_sel  = np.array(pbl_all)[:,mask]
        temp_sel = np.array(temp_all)[:,mask]
        rh_sel   = np.array(rh_all)[:,mask]
        
        return ws_sel, wd_sel, pbl_sel, temp_sel, rh_sel
        
    def read_processed_winddata_multisite(self, date, sites):
        nday = calendar.monthrange(date.year,date.month)[1]
        ws, wd, pbl = np.zeros((len(sites), nday*24)), np.zeros((len(sites), nday*24)), np.zeros((len(sites), nday*24))
        temp, rh    = np.zeros((len(sites), nday*24)), np.zeros((len(sites), nday*24))
        for i,site in enumerate(sites):
            ws[i], wd[i], pbl[i], temp[i], rh[i] = self.read_processed_winddata_onefile(date, site)
        return ws, wd, pbl, temp, rh
            
    def read_processed_winddata_onefile(self, date, site):
        fname = self.get_filename_processed_winddata(date, site)
        ws,wd,pbl,temp,rh = np.load(fname)
        return ws, wd, pbl, temp, rh
        
    def read_raw_winddata_month(self, date, sites):
        nday = calendar.monthrange(date.year,date.month)[1]
        dates = [datetime(date.year,date.month,1) + timedelta(seconds=3600*i) for i in range(nday*24)]
        ws, wd, pbl, temp, rh = self.read_raw_winddata_NAME_multi(dates, sites)
        return ws, wd, pbl, temp, rh
        
    def read_raw_winddata_NAME_multi(self, dates, sites):
        ws, wd, pbl = np.zeros((len(sites), len(dates))), np.zeros((len(sites), len(dates))), np.zeros((len(sites), len(dates)))
        temp, rh    = np.zeros((len(sites), len(dates))), np.zeros((len(sites), len(dates)))
        for i,date in enumerate(dates):
            ws[:,i], wd[:,i], pbl[:,i], temp[:,i], rh[:,i] = self.read_raw_winddata_NAME_onefile(date, sites)
        return ws, wd, pbl, temp, rh
        
    def read_raw_winddata_NAME_onefile(self, date, sites):
        subfolder = self.rc.name_run
        fname = inv.get_filename_winddata_NAME(subfolder, date, self.input_timezone)
        with tarfile.open(fname) as tarf:
            wd, ws, pbl, temp, rh = np.zeros(len(sites)), np.zeros(len(sites)), np.zeros(len(sites)), np.zeros(len(sites)), np.zeros(len(sites))
            for i,site in enumerate(sites):
                fname_in_tar = inv.get_filename_in_tar_meteo(subfolder, date, site, self.input_timezone)
                f = tarf.extractfile(fname_in_tar)
                # Byte to string
                line = str(f.readlines()[39])[2:]
                line = line.split(',')
                if site=='SkyTower':
                    # Different sampling height
                    ws[i] = float(line[3])
                    wd[i] = float(line[4])
                else:
                    ws[i] = float(line[1])
                    wd[i] = float(line[2])
                   
                pbl_idx  = self.get_pbl_idx(date)
                pbl[i]   = float(line[pbl_idx])
                
                temp_idx = pbl_idx-2
                temp[i]  = float(line[temp_idx])
                
                rh_idx   = pbl_idx+2
                rh[i]    = float(line[rh_idx])
        return ws, wd, pbl, temp, rh
    
    def get_pbl_idx(self, date):
        
        # I only later added in the wind speed sample heights, which shifted the PBL idx (in AKLNWP it's included from the start)
        # date_low is because I ran 2021-12-... at a later point, so that one includes all wind levels
        
        if self.input_timezone=='nzst':
            date = date-timedelta(hours=12)
            
        
        if 'baseNZCSM' in self.inversion_name:
            date0 = datetime(2022,1,1,0)
            date1 = datetime(2022,4,10,0)
            date2 = datetime(2022,5,20,21)
            if   (date>=date0) and (date<date1):
                # One sample height
                return 7
            elif (date>=date1) and (date<date2):
                # Two sample height
                return 9
            else:
                return 11
        else:
            return 11
            
    