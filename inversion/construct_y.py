#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construct the observational y-vector for the inversion. 
This goes in three steps:

Step 1: Filter observations for those we want to include, e.g., based on time of 
    day or wind data.

Step 2: We either optimize site-to-site gradients (e.g., with MKH as background)
    or we optimize site observations minus some background (e.g., CarbonTracker/CAMS).
    For now, only the first one is implemented.
    
Step 3: Subtract priors we don't want to optimize.
"""

import read_obs as ro
import inversion as inv
from inversion_base import InversionBase
import functions_stijn as st
import enhancement_calculator as enhc
import numpy as np
import os
from datetime import datetime,timedelta
from read_meteo_NAME import RetrievorWinddataNAME
import pandas as pd

inversion_name = 'baseAKLNWP_base'

#%%

class ConstructorY(InversionBase):
    
    def __init__(self, inversion_name, modification_kwargs={}):
        super().__init__(inversion_name, modification_kwargs)
        
    def construct_y(self):
        self.read_observation_data_raw()
        
        if self.check_if_filter_needed():
            self.select_simulation_timewindow()
            self.filter_observations()
            self.subtract_priors_from_obs()
            self.parse_obs_to_yvector()
        
        self.write_yvector_to_file()
        
    def read_observation_data_raw(self):
        self.dates, self.co2, self.co2_sd = ro.readDataSites(self.rc.sites)
        
    def filter_observations(self):
        """
        Apply observation filters. Multiple observation filters can be applied
        in one inversion.
        """
        
        if self.rc.obs_filters is not None:
            if 'time_of_day' in self.rc.obs_filters:
                self.filter_obs_timeofday()
            if 'windspeed' in self.rc.obs_filters:
                self.filter_obs_windspeed()
            if 'winddirection' in self.rc.obs_filters:
                self.filter_obs_winddirection()
            
        
    def select_simulation_timewindow(self):
        for site in self.co2.keys():
            mask = (self.dates[site]>=self.rc.date_start) & (self.dates[site]<=self.rc.date_end)
            self.dates[site]   = self.dates[site][mask]
            self.co2[site]     = self.co2[site][mask]
        
    def filter_obs_timeofday(self):
        '''
        Filter observations based on time of day (e.g., only use afternoon data)
        '''
        
        for site in self.rc.sites:
            mask = [d.hour in self.rc.obs_filter_hours for d in self.dates[site]]
            self.dates[site] = self.dates[site][mask]
            self.co2[site]   = self.co2[site][mask]
            
    def filter_obs_windspeed(self):
        '''
        Filter observations based on a minimum wind speed. This can be a minimum
        wind speed at one site, or at multiple sites.
        '''
        
        windspeed_lims = self.rc.windspeed_limits # Lower wind speed bound, specific to site
        sites_ws = list(windspeed_lims.keys())
        
        # Read windspeed data; construct windspeed mask
        dates_all = st.get_unique_values_from_dict(self.dates)
        windspeed_mask = np.ones(len(dates_all), dtype=bool)
        for i,site in enumerate(sites_ws):
            ws_i, wd_i = self.read_observed_winds_site(site, dates_all)
            ws_i = self.fill_missing_winds_site(dates_all, ws_i, wd_i, site)[0] # Fill in gaps in observational data
            # Windspeed has to be above windspeed limit at all sites
            windspeed_mask = windspeed_mask & (ws_i>=windspeed_lims[site])
            
        # Only apply windspeed mask during "nighttime"
        daytime_mask = np.array([d.hour in self.rc.afternoon_hours for d in dates_all])
        windspeed_mask[daytime_mask] = True
        
        # Apply windspeed filter
        for site in self.co2.keys():
            date_mask = [d in self.dates[site] for d in dates_all]
            mask_i = windspeed_mask[date_mask]
            self.dates[site] = self.dates[site][mask_i]
            self.co2[site]   = self.co2[site][mask_i]            
    
    def filter_obs_winddirection(self):
        '''
        Filtering based on wind direction is a little different from the other filters,
        because I include the option to optimize different gradients for different
        wind directions. So rather then creating one mask and then applying it to
        all data, I create multiple masks that I later apply when I construct the
        gradient. The idea being that NE winds need TKA as background site, and SW
        winds need MKH.
        self.rc.winddir_windows is a nested dictionary; the first key is each group
        (e.g., SW) and the second key is which sites I use to filter in that group.
        Note that the same keys/groups included in self.rc.winddir_windows are included
        in self.rc.obs_gradients. 
        '''
        
        # Dictionary with winddirection windows mapped to labeled groups (e.g., North-easterly, South-westerly)
        winddir_windows = self.rc.obs_filter_winddir_windows
        
        # Read wind direction data
        dates_all = st.get_unique_values_from_dict(self.dates)
        self.winddir_masks = {}
        for group, winddir_windows_i in winddir_windows.items():
            self.winddir_masks[group] = np.ones(len(dates_all), dtype=bool)
            for site_wd,(lo,hi) in winddir_windows_i.items():
                # Read and fill wind direction data
                ws_i,wd_i = self.read_observed_winds_site(site_wd, dates_all)
                wd_i = self.fill_missing_winds_site(dates_all, ws_i, wd_i, site_wd)[1]
                
                # Calculate mask for each wind direction window
                mask = (wd_i>=lo) & (wd_i<=hi)
                self.winddir_masks[group] = self.winddir_masks[group] & mask
    
    def read_observed_winds_site(self, site, dates):
        '''
        Read observed wind speeds & directions at a specific site, for a list of dates.
        This is just a parser to access the observed data.
        '''
        
        ws, wd = ro.read_winddata(dates, site, timezone='nzst', daily_or_hourly='hourly')
        return ws, wd
    
    def fill_missing_winds_site(self, dates_all, ws, wd, site):
        '''
        Fill the wind timeseries for the dates where we have no observational
        data. Multiple options: can interpolate, or use simulated winds.
        
        I later added an option in case I only want to use modeled winds, useful for OSSE
        '''
        
        if self.rc.winddata_fill_method=='interp':
            ws, wd = self.fill_missing_winds_with_interpolation(dates_all, ws, wd)
            
        elif self.rc.winddata_fill_method=='model':
            ws, wd = self.fill_missing_winds_with_model(dates_all, ws, wd, site)
            
        elif self.rc.winddata_fill_method=='no_obs_only_model':
            ws[:], wd[:] = np.nan, np.nan
            ws, wd = self.fill_missing_winds_with_model(dates_all, ws, wd, site)
            
        return ws, wd
        
    def fill_missing_winds_with_interpolation(self, dates_all, ws, wd):
        # Read interpolation method; if not available, interpolate linearly
        try:
            method = self.rc.wind_interpolation_method
        except AttributeError:
            print("Wind interpolation method not prescribed (wind_interpolation_method): "+\
                  "Using linear interpolation)")
            method = 'linear'
        
        df = pd.DataFrame(index=dates_all, data=dict(ws=ws, wd=wd))
        df = df.interpolate(method=method)
        
        return df['ws'].to_numpy(), df['wd'].to_numpy()     
    
    def fill_missing_winds_with_model(self, dates_all, ws, wd, site):
        
        # Read missing dates
        mask_missing = (np.isnan(ws)) | (np.isnan(wd))
        
        if mask_missing.sum()>0:
            dates_missing = np.array(dates_all)[mask_missing]
            ws_model, wd_model = self.read_windspeed_model(dates_missing, site)
            
            # Fill in the observation timeseries
            ws[mask_missing] = ws_model
            wd[mask_missing] = wd_model
        
        return ws, wd
    
    def read_windspeed_model(self, dates, site):
        retr = RetrievorWinddataNAME(inversion_name)
        winddata = retr.retrieve_winddata_NAME(dates, [site], sitename_version='obs')
        ws_model, wd_model = winddata[0][0], winddata[1][0] # Get out the site dimension
        return ws_model, wd_model
    
    def subtract_priors_from_obs(self):
        '''
        Sometimes we only want to optimize e.g., fossil fuel and not biogenic.
        To do this, we subtract the biogenic signal from the observations. Obviously
        that works for any prior. 
        '''
        
        for site in self.co2.keys():
            enh_i = self.retrieve_prior_enhancements(site)
            self.co2[site] -= enh_i
            
    def retrieve_prior_enhancements(self, site):
        dates_i = self.dates[site]
        prior_enh = np.zeros(len(dates_i))
        for domain,inventories in self.rc.priors_fixed.items():
            for inventory in inventories:
                calc = enhc.CalculatorIntegratedEnhancements(self.inversion_name, domain, inventory)
                unique_days = np.unique([datetime(d.year,d.month,d.day) for d in dates_i])
                enh_all_hours = calc.retrieve_integrated_enhancements_multiday(site, unique_days)
                enh_i = st.select_hours_from_fulldays(dates_i, unique_days, enh_all_hours)
                prior_enh += enh_i
            
        return prior_enh
        
    def parse_obs_to_yvector(self):
        
        if self.rc.obs_method=='gradient':
            self.parse_obs_to_yvector_gradient()
        elif self.rc.obs_method=='per_site':
            self.parse_obs_to_yvector_per_site()
        else:
            raise KeyError("Observation parser not implemented; obs_method=%s"%self.rc.obs_method)
        
    def parse_obs_to_yvector_gradient(self):
        '''
        In this set-up we optimize gradients between sites. 
        I put in an option to optimize different sets of gradients for different
        wind conditions, which might help capture that north-easterlies and south-
        westerlies have a different optimal background site (TKA / MKH respectively).
        But the code is generic so you can put in whatever gradients you want under
        different conditions.
        '''
        
        self.ydates  = np.zeros(0)
        self.yvector = np.zeros(0)
        self.ysites  = np.zeros(0,dtype=object)
        
        for k,gradients in self.rc.obs_gradients.items():
            for site1, site2 in gradients:
                # We can only take the gradient on overlapping days
                
                dates_i, dco2_i = ro.calculate_obs_gradient(self.dates[site1], self.co2[site1], self.dates[site2], self.co2[site2])
                dates_i, dco2_i = self.apply_groupspecific_filters(k, dates_i, dco2_i)
                dsite_i = ['%s-%s'%(site1,site2) for i in range(len(dco2_i))]
                
                self.ydates  = np.concatenate([self.ydates , dates_i])
                self.yvector = np.concatenate([self.yvector, dco2_i])
                self.ysites  = np.concatenate([self.ysites , dsite_i])
        
    def parse_obs_to_yvector_per_site(self):
        '''
        In this set-up we optimize site enhancements. This I only use for generating
        the OSSE truth, because to implement this in a way that is useable, we also
        need to include a background (e.g., carbontracker), which I haven't.
        '''
        
        self.ydates  = np.zeros(0)
        self.yvector = np.zeros(0)
        self.ysites  = np.zeros(0,dtype=object)
        
        for site in self.rc.sites:
            
            self.ydates  = np.concatenate([self.ydates , self.dates[site]])
            self.yvector = np.concatenate([self.yvector, self.co2[site]])
            self.ysites  = np.concatenate([self.ysites , [site]*len(self.co2[site])])
                
    def apply_groupspecific_filters(self, k, dates, dco2):
        '''
        We can optimize different groups of CO2 gradients. Here we apply the conditions
        that sets each group of site-to-site gradient apart from one another. 
        For now this is just wind direction.
        '''
        
        if 'winddirection' in self.rc.obs_filters:
            dates, dco2 = self.apply_mask_winddirection(k, dates, dco2)
        return dates, dco2
            
    def apply_mask_winddirection(self, k, dates, dco2):
        
        dates_all = st.get_unique_values_from_dict(self.dates)
        wd_mask = self.winddir_masks[k]
        
        wd_mask = wd_mask[[d in dates for d in dates_all]]
        return dates[wd_mask], dco2[wd_mask]
        
    def write_yvector_to_file(self):
        path = self.get_path_inversion_input()
        np.save('%s/yvector.npy'%path, self.yvector)
        np.save('%s/ydates.npy'%path, self.ydates)
        np.save('%s/ysites.npy'%path, self.ysites)
        
    def check_if_filter_needed(self):
        '''
        In creating real obs, filter is always needed. When creating synthetic
        obs, it's sometimes not, and then coding it like this makes it easy
        to build that in.
        '''
        
        return True
        

#%%
if __name__ == "__main__":
    
    
    proc = ConstructorY(inversion_name)
    proc.construct_y()
    
    import matplotlib.pyplot as plt
    
    ysites, ydates, yvector = proc.read_yvector()
    dates = []
    for grad in np.unique(ysites):
        mask = ysites==grad
        plt.figure()
        plt.title(grad)
        dates.append(ydates[mask])
        plt.plot(ydates[mask],yvector[mask], 'o')
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#%%

from datetime import datetime, timedelta

n = 10
x = [datetime(2020,1,1) + timedelta(seconds=5*3600*i) for i in range(n)]
y = np.random.rand(n)
z = np.random.rand(n)
y[1] = np.nan
y[5] = np.nan

z[1] = np.nan
z[5] = np.nan

import pandas as pd

df = pd.DataFrame(index=x, data=dict(y=y, z=z))        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


