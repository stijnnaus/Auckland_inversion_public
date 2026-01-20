#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:01:24 2024

@author: nauss

This space is to set up a way to construct synthetic observations that can be
used in an OSSE. It can heavily borrow from the base construct_y (e.g., wind filtering),
and only needs to replace the "read_obs" function. Since I'm expecting this part
of my code might get a bit bloated with OSSE options, it's easier to keep it split
from the core code that "only" has to read real observations.
"""

from construct_y import ConstructorY
import enhancement_calculator as enhc
import numpy as np
from datetime import datetime,timedelta
from netCDF4 import Dataset

class ConstructorY_synthetic(ConstructorY):
    
    def __init__(self, inversion_name, modification_kwargs={}, barebones=False, seed=None):
        super().__init__(inversion_name, modification_kwargs)
        
        # An option to create a placeholder version of y, that doesn't use R or H
        # Necessary, because I need y to create R (e.g., dt-matrix), but I need R to create y (perturbation based on R)
        self.barebones = barebones
        
        # Fixed random seed for reproducible results
        np.random.seed(1)
        
    def construct_y(self):
        self.read_observation_data_raw()
        
        if self.check_if_filter_needed():
            self.select_simulation_timewindow()
            self.filter_observations()
            self.subtract_priors_from_obs()
            self.introduce_data_gaps()
            self.parse_obs_to_yvector()
        
        if not self.barebones and not self.rc.osse_create_obs_method=='from_file':
            # We don't apply synthetic noise if y comes from file, assume that it already has synthetic noise
            self.apply_perturbations_to_yvector()
            
        self.write_yvector_to_file()
        
    def read_observation_data_raw(self):
        
        method = self.rc.osse_create_obs_method
        if self.barebones:
            self.calc_synthetic_enhancements_placeholder()
        elif method=='from_state':
            self.calc_synthetic_enhancements_from_pert_state()
        elif method=='from_inv_dict':
            self.calc_synthetic_enhancements_from_inv_dict()
        elif method=='from_file':
            self.get_synthetic_enhancements_from_file()
        
    def calc_synthetic_enhancements_placeholder(self):
        """
        Fill vector with ones for all sites, all dates. Of course, this is then
        still filtered based on wind data after read_observation_data_raw. The
        ysites, ydates can then be used to construct_R, construct_H.
        The H-matrix is needed to construct the "real" synthetic observations.
        """
        
        dates = self.get_full_datelist_obs()
        enh_tot = np.ones((len(self.rc.sites), len(dates)))
        self.dates, self.co2 = self.array_to_dict(dates, enh_tot)
        
    def calc_synthetic_enhancements_from_pert_state(self):
        """
        Calculate synthetic enhancements from a perturbation of the prior state.
        The perturbation we do based on prior error matrix B. 
        Still has option of introducing bias on top of that
        """
        
        H = self.read_Hmatrix()
        x = self.read_state()[0]
        B = self.read_Bmatrix()
        
        B = B*self.rc.osse_xpert_B_scale
        
        xtrue = np.random.multivariate_normal(x, B, 1)[0]
        
        self.include_biases_in_xtrue(xtrue)
        self.perturb_diurnal_cycle_xtrue(xtrue)
        
        self.yvector = H @ xtrue
        
        # We also need to define ysites and ydates for writing later
        self.ysites, self.ydates, _ = self.read_yvector()
        
        # We save xtrue since we need to know the truth in post-processing
        self.write_xtrue(xtrue)
        
    def get_synthetic_enhancements_from_file(self):
        """
        Prepared enhancements are in a netCDF per site, unfiltered.
        """
        
        self.dates, self.co2 = {}, {}
        with Dataset(self.get_filename_true_enhancements(self.rc.osse_truth_label)) as d:
            for site in self.rc.sites:
                if site in d.groups:
                    self.dates[site] = np.array([datetime(y,m,d,h) for [y,m,d,h] in d[site]['dates'][:]])
                    self.co2[site]   = np.array(d[site]['co2'][:])
                    
                else:
                    raise KeyError("All sites need to be included in true enhancements, but %s is not in true enhancements!"%site)
        
    def include_biases_in_xtrue(self, xtrue):
        """
        On top of the perturbation based on B, we can also scale a full domain/prior,
        i.e., introduce a bias.
        """
        
        _, priors_x, domains_x = self.read_state()
        for domain,priors_dom in self.rc.priors_to_opt.items():
            for prior in priors_dom:
                try:
                    mask  = (domains_x==domain) & (priors_x==prior)
                    scale = float(getattr(self.rc, 'osse_%s_%s_scale'%(domain, prior)))
                    xtrue[mask] *= scale
                except AttributeError:
                    print("No true scaling defined for %s / %s, so we're not including a bias."%(domain, prior))
        
        return xtrue
        
    def perturb_diurnal_cycle_xtrue(self, xtrue):
        """
        For now I hard-code a perturbation to the diurnal cycle to test the value of
        night-time data
        """
        
        pass
        
    def calc_synthetic_enhancements_from_inv_dict(self):
        """
        Calculate synthetic enhancements from a list of inventories. This way,
        I am not tied to the inventory set-up I use in the inversion.
        
        NOTE: This is kind of outdated, and a lot of the postprocessing assumes
            that there is an "xtrue". So not everything will work if this method is used,
            but this is the method you would need if you want to exchange priors.
        """
        
        dates = self.get_full_datelist_obs()
        enh_tot = np.zeros((len(self.rc.sites), len(dates)))
        for domain,inventories in self.rc.inventories_for_synth_obs.items():
            for inventory in inventories:
                enh_i = self.read_enhancements_inventory_domain(dates, domain, inventory)
                enh_i = self.scale_enhancements_inventory_domain(dates, domain, inventory, enh_i)
                enh_i = enh_i.sum(axis=(-1))
                enh_tot += enh_i
                
        self.dates, self.co2 = self.array_to_dict(dates, enh_tot)
                
    def read_enhancements_inventory_domain(self, dates, domain, inventory):
        
        udays = np.unique(([datetime(d.year,d.month,d.day) for d in dates]))
        enhancements = np.zeros((len(self.rc.sites), len(udays), 24, self.rc.nstepNAME))
        
        for isite,site in enumerate(self.rc.sites):
            for iday,day in enumerate(udays):
                enhi = self.read_enhancements_basegrid(day, domain, inventory, site)
                enhancements[isite,iday] = enhi.sum(axis=(-1,-2)) # For now we don't consider spatial dimensions
        
        # The integrated enhancements are now 24 hours for each day, but we only need specific hours
        enh_hours = self.select_specific_hours_from_full_days(dates, udays, enhancements)
        
        return enh_hours
    
    def scale_enhancements_inventory_domain(self, dates, domain, inventory, enhancements):
        '''
        Enhancements are now shape (len(sites), len(dates), nstepNAME). We generally
        want to scale based on emissions (e.g., diurnal cycle), which corresponds to nstepNAME.
        '''
        
        varbname = 'osse_%s_%s_scale'%(domain, inventory)
        if hasattr(self.rc, varbname):
            # Just a simple uniform scaling
            enhancements *= getattr(self.rc, varbname)
            
        varbname = 'osse_%s_%s_diurnalscaling'%(domain, inventory)
        if hasattr(self.rc, varbname):
            # Scaling of the diurnal cycle of emissions
            scaling = np.array(getattr(self.rc, varbname))
            dt      = np.array([timedelta(seconds=3600*i) for i in range(self.rc.nstepNAME)])
            for idate,date in enumerate(dates):
                hours = [d.hour for d in date-dt]
                enhancements[:, idate] *= scaling[hours]
        
        return enhancements
    
    def select_specific_hours_from_full_days(self, dates_hour, dates_day, arr_day):
        arr_hour = np.zeros((len(self.rc.sites), len(dates_hour), self.rc.nstepNAME))
        for i,date_hour in enumerate(dates_hour):
            date_day = datetime(date_hour.year, date_hour.month, date_hour.day)
            iday = np.where(dates_day==date_day)[0][0]
            # 0-th axis is site
            arr_hour[:,i] = arr_day[:,iday,date_hour.hour]
        return arr_hour
        
    def array_to_dict(self, dates, co2):
        # First dimension is sites, but construct_y has this dimension as dictionary keys
        # so make it a dictionary here
        dates_dict, co2_dict = {}, {}
        for i,site in enumerate(self.rc.sites):
            dates_dict[site] = dates
            co2_dict[site] = co2[i]
        return dates_dict, co2_dict
        
    def apply_perturbations_to_yvector(self):
        '''
        In case I want to apply perturbations to the yvector, such as perturbations
        based on R, I need to do it after wind-filtering etc.
        '''
        
        self.apply_random_fixed_error_to_yvector()
        self.perturb_yvector_based_on_R()
        
    def apply_random_fixed_error_to_yvector(self):
        try:
            err = self.rc.osse_fixed_obs_error
            self.yvector = np.random.normal(self.yvector, scale=err)
        except AttributeError:
            print("No fixed obs error prescribed")
        
    def perturb_yvector_based_on_R(self):
        '''
        Perturb the synthetic yvector based on the R matrix.
        '''
        
        R = self.read_Rmatrix()
        R = R*self.rc.osse_ypert_R_scale
        self.yvector = np.random.multivariate_normal(self.yvector, R, 1)[0]
    
    def introduce_data_gaps(self):
        pass
                
    def get_scaling_factor_synth_obs(self, domain, inventory):
        # It's an optional argument in the input file configuration
        varbname = 'osse_%s_%s_scale'%(domain,inventory)
        try:
            scale = getattr(self.rc, varbname)
        except AttributeError:
            print('Didnt find: %s in self.rc; so assuming it is one'%(varbname))
            scale = 1
            
        return scale
    
    def get_full_datelist_obs(self):
        dates = []
        date_curr = self.rc.date_start
        while date_curr<=self.rc.date_end:
            dates.append(date_curr)
            date_curr += timedelta(seconds=3600)
        return np.array(dates)
    
    # def construct_Bmatrix(self):
    #     # Read components and construct B
        
    #     self.dx_2d       = self.read_dx()
    #     self.dt_diurn_2d = self.read_dt('diurnal')
    #     self.dt_long_2d  = self.read_dt('longterm')
        
    #     from construct_B import ConstructorB
    #     constrB = ConstructorB(self.inversion_name, self.modification_kwargs)
        
    #     return constrB.create_B_from_dt_dx(self.dt_diurn_2d, self.dt_long_2d, self.dx_2d)
    
    def write_xtrue(self, xtrue):
        fname = self.get_filename_xtrue()
        np.save(fname, xtrue)
        
    def check_if_filter_needed(self):
        '''
        Used in construct_y.
        We only need to filter if read_obs_raw returns unfiltered, per-site data.
        But if I create observations from (H @ x) it's already gradients and 
        wind-filtered (and it would not be easy to go back to unfiltered per-site data).
        So we filter in all cases where y is not created from H@x
        '''
        
        method = self.rc.osse_create_obs_method
        return (self.barebones or method=='from_inv_dict' or method=='from_file')
        
        
        
        
#%%
if __name__ == "__main__":
    inversion_name = 'baseAKLNWP_base'
    
    proc = ConstructorY_synthetic(inversion_name)
    
    proc.construct_y()
    
    import matplotlib.pyplot as plt
    
    ysites, ydates, yvector = proc.read_yvector()
    usites = np.unique(ysites)
    fig, ax = plt.subplots(len(usites), 1, figsize=(10,6*len(usites)))
    for i,grad in enumerate(usites):
        mask = ysites==grad
        ax[i].set_title(grad)
        ax[i].plot(ydates[mask],yvector[mask], 'o') 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        