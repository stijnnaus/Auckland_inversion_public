# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:19:59 2023

@author: nauss
"""

import inversion as inv
import read_priors as rp
import functions_stijn as st
import numpy as np
from datetime import datetime,timedelta
import calendar
import os,time
from inversion_base import InversionBase

class CalculatorEnhancementsForInversion(InversionBase):
    
    def __init__(self, inversion_name):
        super().__init__(inversion_name) 
        self.nsite  = len(self.rc.sites)        
        
    def init_domain(self, domain):
        # NAME grid variables
        nxi, nyi = self.rc.nx_base[domain],self.rc.ny_base[domain]
        
        self.loncenters_name_full, self.latcenters_name_full   = inv.getLonLatNAME(self.rc.domains_NAME[domain], bounds=False)
        if self.rc.outer_bounds[domain] is not None:
            self.masklon_outer = st.make_mask_bounds_in(self.loncenters_name_full, self.rc.outer_bounds[domain]['lon'])
            self.masklat_outer = st.make_mask_bounds_in(self.latcenters_name_full, self.rc.outer_bounds[domain]['lat'])
        else:
            self.masklon_outer = np.ones_like(self.loncenters_name_full, dtype=bool)
            self.masklat_outer = np.ones_like(self.latcenters_name_full, dtype=bool)
        self.loncenters_name = self.loncenters_name_full[self.masklon_outer]
        self.latcenters_name = self.latcenters_name_full[self.masklat_outer]  
        self.lonbounds_name  = st.get_gridcell_bounds_from_centers(self.loncenters_name)
        self.latbounds_name  = st.get_gridcell_bounds_from_centers(self.latcenters_name)
        self.areas_name = st.calc_area_per_gridcell(self.latbounds_name,self.lonbounds_name,bounds=True)    
        
        if self.rc.inner_bounds[domain] is not None:
            self.masklon_inner = st.make_mask_bounds_in(self.loncenters_name, self.rc.inner_bounds[domain]['lon'])
            self.masklat_inner = st.make_mask_bounds_in(self.latcenters_name, self.rc.inner_bounds[domain]['lat'])
            
        
        # Regridded grid variables
        lonlo, lonhi = self.lonbounds_name.min(), self.lonbounds_name.max()
        latlo, lathi = self.latbounds_name.min(), self.latbounds_name.max()
        self.lonbounds_regr = np.linspace(lonlo,lonhi,nxi+1)
        self.latbounds_regr = np.linspace(latlo,lathi,nyi+1)
        self.loncenters_regr = 0.5*(self.lonbounds_regr[1:]+self.lonbounds_regr[:-1])
        self.latcenters_regr = 0.5*(self.latbounds_regr[1:]+self.latbounds_regr[:-1])    
        self.areas_regr = st.calc_area_per_gridcell(self.latbounds_regr,self.lonbounds_regr,bounds=True)    
        
    def read_all_emissions(self, dates, domain, inventories, remove_diurnal=False):
        '''
        Read emissions for all dates, all categories.
        Go back 2 extra days because we need those two days due to our runs being backwards.
        We read all emissions here so that we only need to read them once.
        '''
        
        nback = int(np.ceil(self.rc.nstepNAME/24.)) # How many days back we read emissions
        self.dates_emis = [dates[0]+timedelta(days=int(i)) for i in np.arange(-nback, len(dates))]
        self.emis_all = {}
        for emi_inv in inventories:
            self.emis_all[emi_inv] = rp.read_preprocessed_1inventory(emi_inv, self.dates_emis, domain, cats='all')
            self.emis_all[emi_inv] *= 1000 # From [kg/m2/s] to [g/m2/s]
            if remove_diurnal:
                # Remove diurnal cycle from emissions
                self.emis_all[emi_inv][:,:,:,:] = self.emis_all[emi_inv].mean(axis=1)[:,np.newaxis,:,:]
            
    def setup_regridder(self):
        '''
        Function to set-up the regridder. With how xesmf works, setting it up here
        once and then applying it multiple times (per inventory/date) is much faster
        than setting it up each time.
        '''
        
        self.regridder = st.make_xesmf_regridder(self.loncenters_name, self.latcenters_name, self.loncenters_regr, self.latcenters_regr)
            
    def process_enhancements(self, dates, domain, inventories, remove_diurnal=False):
        '''
        Calculate enhancements from emissions and footprints for all dates,
        and write one file per inventory, per hour (note: each hour is one
                                                    backwards simulation of 26 hours)
        remove_diurnal is an argument that means I do not include the diurnal
        cycle in emissions in the enhancements. Handy for sensitivity tests.
        If set to true, the output path is modified
        '''
        
        print("Reading emissions...")
            
        self.init_domain(domain)
        self.read_all_emissions(dates, self.rc.domains_NAME[domain], inventories, remove_diurnal=remove_diurnal)
        self.setup_regridder()
        
        nxi, nyi = self.rc.nx_base[domain],self.rc.ny_base[domain]
        for date in dates:
            print(date.strftime('%Y-%m-%d'))
            enh = {inv:np.zeros((self.nsite, 24, self.rc.nstepNAME, nyi, nxi)) for inv in inventories}
            for hour in range(24):
                date_i   = datetime(date.year, date.month, date.day, hour)
                timesteps, footpr = self.read_footprint(date_i, self.rc.domains_NAME[domain])
                for emis_inv in inventories:
                    emis_i = self.select_emis_steps(timesteps, emis_inv)
                    enh_i  = self.calc_enhancement(footpr, emis_i) # [ppm/m2] for regridding
                    enh_i  = self.crop_enhancements_inout(domain, enh_i)
                    enh_i  = self.regridder(enh_i)
                    enh_i *= self.areas_regr # To [ppm]
                    enh[emis_inv][:,hour] = enh_i
                    
            for emis_inv,enh_i in enh.items():
                for i,site in enumerate(self.rc.sites):
                    emis_inv_name = emis_inv+'_nodiurnal' if remove_diurnal else emis_inv
                    self.save_enhancements_basegrid(date, domain, emis_inv_name, site, enh_i[i])
                
    def save_enhancements_basegrid(self, date, domain, emis_inv, site, enh):
        fname = self.get_filename_enhancements_basegrid(date, domain, emis_inv, site)
        print("Writing to %s"%fname)
        np.save(fname, enh)
            
    def crop_enhancements_inout(self, domain, enh):
        '''
        Apply inner and outer boundaries of domain, which can be different from the boundaries
        of the NAME domain.
        '''
        
        if self.rc.outer_bounds[domain] is not None:
            enh = self.crop_outer_bounds(enh)
            
        if self.rc.inner_bounds[domain] is not None:
            enh = self.set_inner_nested_to_zero(enh)
            
        return enh

    def crop_outer_bounds(self,enh):
        return enh[:,:,self.masklat_outer][:,:,:,self.masklon_outer]
            
    def set_inner_nested_to_zero(self, enh):            
        mask_inner = np.outer(self.masklat_inner, self.masklon_inner)
        enh[:,:,mask_inner] = 0.0
        return enh
        
    def select_emis_steps(self, timesteps, emis_inv):
        return rp.get_specific_hours_from_daily_cycles(timesteps, self.dates_emis, self.emis_all[emis_inv])
                    
    def read_footprint(self, date, domain):
        gridlabel  = self.get_gridlabel_NAME(domain)
        domainNAME = self.get_domainname_NAME(domain)
        fieldlabel = 'grid_%s'%gridlabel
        tarlabel = '_%s'%gridlabel
        timesteps,footprints = inv.read_footprint_hourly(date, nsite=self.nsite, nstepNAME=self.rc.nstepNAME, dt=3600, 
                                                         domain=domainNAME, field_label=fieldlabel, subfolder=self.rc.name_run, 
                                                         tar_label=tarlabel, tarred=True, timezone='NZST') # ppm
        return timesteps,footprints
        
    def calc_enhancement(self, footpr, emis):
        return emis[np.newaxis,:,:,:]*footpr[:,:,:,:]
            
    def get_gridlabel_NAME(self, domain):
        return '%s_%s'%(self.get_domainname_NAME(domain),self.rc.samplelayer)
    
        
    
    
class CalculatorIntegratedEnhancements(InversionBase):
    '''
    A calculator for integrated enhancements per site (i.e., one value per NAME
    simulation per site). The calculator has three options for doing the calculation:
        1) Just read integrated enhancements, if these have been calculated before
        2) Read enhancements on inversion grid, if calculated before, and sum those
        3) Calculate enhancements on inversion grid, read those, and sum them
    In case 2) and 3) integrated enhancements are saved to daily files.
    '''
    
    def __init__(self, inversion_name, domain, inventory):
        super().__init__(inversion_name)
        self.domain    = domain
        self.inventory = inventory
        
    def retrieve_integrated_enhancements_multiday_multisite(self, sites, dates, overwrite_integrated=False):
        
        enh_all = np.zeros((len(sites), len(dates), 24))
        for i,site in enumerate(sites):
            enh_all[i] = self.retrieve_integrated_enhancements_multiday(site, dates, overwrite_integrated)
        return enh_all
        
    def retrieve_integrated_enhancements_multiday(self, site, dates, overwrite_integrated=False):
        
        # Integrated enhancement files are monthly, read monthly data
        months_unq = np.unique([datetime(d.year,d.month,1) for d in dates])
        # If certain months don't have an integrated file yet, we generate those
        self.generate_missing_months(site, months_unq, overwrite_integrated)
        
        # Once those files are there, we can read all data from all months
        dates_all, enh_all = self.read_enhancements_all_months(site, months_unq)
        
        # And then we just need to select the requested dates
        mask = [d in dates for d in dates_all]
        enh_sel = enh_all[mask]
        
        return enh_sel
    
    def generate_missing_months(self, site, months, overwrite_integrated):
        # Check which months do not have an integrated enhancements file; for those months generate one
        for month in months:
            fname = self.get_filename_enhancements_integrated(self.domain, self.inventory, site, month)
            if not os.path.isfile(fname) or overwrite_integrated:
                print(month.strftime("Generating missing month %Y-%m ") + "(site:%s ; inv:%s ; dom:%s)"%(site, self.inventory, self.domain))
                self.generate_integrated_file_month(month, site)
                
    def generate_integrated_file_month(self, month, site):
        fname = self.get_filename_enhancements_integrated(self.domain, self.inventory, site, month)
        nday = calendar.monthrange(month.year,month.month)[1]
        dates_all = np.array([datetime(month.year,month.month,i) for i in range(1,nday+1)])
        enh_all = np.zeros(((len(dates_all)), 24))
        for i,date in enumerate(dates_all):
        
            if not self.check_if_basegrid_enhancements_exist(site, date):
                print("Have to calculate enhancements for %s; %s, %s."%(self.rc.name_run, date.strftime('%Y-%m-%d'), self.inventory))
                self.calculate_basegrid_enhancements(date)
                
            enh_all[i] = self.load_basegrid_enhancements(self.domain, self.inventory, site, date).sum(axis=(1,2,3))
            
        np.save(fname, enh_all)
        
    def read_enhancements_all_months(self, site, months):
        dates_all = np.zeros((0), dtype=object)
        enh_all   = np.zeros((0,24), dtype=float)
        for month in months:    
            dates_i, enh_i = self.read_enhancements_one_month(site, month)
            dates_all = np.append(dates_all, dates_i, axis=0)
            enh_all   = np.append(enh_all  , enh_i  , axis=0)
            
        return dates_all, enh_all
    
    def read_enhancements_one_month(self, site, month):
        nday = calendar.monthrange(month.year,month.month)[1]
        dates_all = [datetime(month.year, month.month, i) for i in range(1, nday+1)]
        fname = self.get_filename_enhancements_integrated(self.domain, self.inventory, site, month)
        enh_all = np.load(fname)
        return dates_all, enh_all
    
    def calculate_basegrid_enhancements(self, date):
        '''
        Since enhancements on inversion grid don't exist, we first have to calculate
        those. Note that doing it per day loses some efficiency relative to doing
        it for a list of dates in a separate script (but not too much I think?).
        Also note that this is always done for all sites and inventories included 
        in the inversion set-up, regardless of which are actually required here.
        '''
        
        calc_inv = CalculatorEnhancementsForInversion(self.inversion_name)
        calc_inv.process_enhancements([date], self.domain)
        
    def check_if_basegrid_enhancements_exist(self, site, date):
        fname = self.get_filename_enhancements_basegrid(date, self.domain, self.inventory, site)
        
        return os.path.isfile(fname)
        
    
    
    
    
    
    
