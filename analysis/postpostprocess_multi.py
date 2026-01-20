#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:55:55 2024

@author: nauss
"""


from datetime import timedelta, datetime
import numpy as np
from base_paths import path_figs
path_figs = '%s/osse_longterm/'%path_figs
from postprocessing import Postprocessing
from inversion_main import Inversion


class Postpostprocess_multi(object):
    """
    This class is used to link together the output of one inversion set-up that was run
    in multiple timewindows. It's post-postprocess because we have already aggregated
    the raw inversion output with the Postprocess class, so this class only works
    with aggregated emissions, uncertainties, and with postprocessed observations.
    
    (startdate_full, enddate_full, dt_inv, dt_spinup, dt_spindown) is the "meta-data" 
    that individual inversions don't know about.
    """
    
    def __init__(self, inversion_name, startdate_full, enddate_full, dt_inv, dt_spinup, dt_spindown):
        self.inversion_name = inversion_name
        self.startdate_full = startdate_full
        self.enddate_full   = enddate_full
        self.dt_inv         = dt_inv         # Length of one inversion
        self.dt_spinup      = dt_spinup
        self.dt_spindown    = dt_spindown
        
        self.get_timewindows_inversions()
        self.ninv = len(self.startdates_all)
        
        # Can be helpful if I need to access the inversion configuration:
        mod_rc = {'date_start':['direct',self.startdates_all[0]], 'date_end':['direct', self.enddates_all[0]]}
        
        self.inversion_example = Inversion(self.inversion_name,      mod_rc)
        self.rc                = self.inversion_example.rc
        self.inventories       = np.unique(self.inversion_example.read_state()[1])
        
        self.postpr_example    = Postprocessing(self.inversion_name, mod_rc, self.dt_spinup, self.dt_spindown)
    
            
    def read_all_postprocessed_data(self):
        '''
        String together the inversions that cover consecutive timeperiods to one long
        timeseries for emissions and its uncertainties. 
        Note that postprocessing has already accounted for spinup/down, so we don't
        have to do anything for that here.
        '''
        
        
        self.emis_all    = self.postpr_example.setup_container_dict_emis()
        self.unc_all_rel = self.postpr_example.setup_container_dict_Bmatrix()
        self.unc_all_abs = self.postpr_example.setup_container_dict_Bmatrix()
        self.obs_all     = self.postpr_example.setup_container_dict_obs()
        self.dates_all   = self.postpr_example.setup_container_dict_dates()
        for startdate_i,enddate_i in zip(self.startdates_all, self.enddates_all):
            mod_rc =  {'date_start':['direct',startdate_i], 'date_end':['direct',enddate_i]}
            postpr = Postprocessing(self.inversion_name, mod_rc, self.dt_spinup, self.dt_spindown)
            
            self.parse_emissions_one(postpr)
            self.parse_uncertainties_one(postpr, 'rel')
            self.parse_uncertainties_one(postpr, 'abs')
            self.parse_observations_one(postpr)
            self.parse_dates_one(postpr)
            
        # self.add_total_category_to_emis_dict()
        
            
    def parse_emissions_one(self, postpr):
        emis_short_dict = postpr.read_aggregated_emissions()
        for v in postpr.get_emission_labels():
            for xres,tres in postpr.aggregate_resolutions:
                for inventory in self.inventories:
                    emis_short_i = emis_short_dict[v][xres][tres][inventory]
                    emis_long_i  = self.emis_all[v][xres][tres][inventory]
                    
                    self.emis_all[v][xres][tres][inventory] = self.vstack_conditional(emis_long_i, emis_short_i)
                        
    # def add_total_category_to_emis_dict(self):
    #     '''
    #     Add an additional entry to the emission dictionary that adds up all
    #     inventories to a total flux.
    #     '''
        
    #     for label in self.emis_all.keys():
    #         for xres in self.emis_all[label].keys():
    #             if xres!='per_gridcell':
    #                 for tres in self.emis_all[label][xres].keys():
    #                     etot = 0.
    #                     for prior in self.emis_all[label][xres][tres].keys():
    #                         etot += self.emis_all[label][xres][tres][prior]
    #                     ei2['Total'] = etot
                        
    def parse_uncertainties_one(self, postpr, rel_or_abs='rel'):
        # Read B matrix of inversion covering one time window and append to the full uncertainty vector
        Bmatrix_short_dict = postpr.read_aggregated_Bmatrix(rel_or_abs)
        for v in postpr.get_Bmatrix_labels():
            for xres,tres in postpr.aggregate_resolutions:
                for inventory1 in self.inventories:
                    for inventory2 in self.inventories:
                        
                        if rel_or_abs == 'rel':
                            unc_all = self.unc_all_rel
                        else:
                            unc_all = self.unc_all_abs
                        
                        unc_all_i = unc_all[v][xres][tres][inventory1][inventory2]
                        
                        B_short_i = Bmatrix_short_dict[v][xres][tres][inventory1][inventory2]
                        
                        # I don't want to deal with cross-correlations between priors that cover
                        # different domains, since that's too complicated.
                        if inventory1==inventory2:
                            # We need the diagonal, but it's a (nt,nx,nt,nx) dimensional matrix,
                            # so reshape to a (nt*nx,nt*nx) matrix
                            nt,nx,_,_   = B_short_i.shape
                            B_short_i   = B_short_i.reshape(nt*nx,nt*nx)
                            unc_short_i = np.diag(B_short_i)
                            unc_short_i = unc_short_i.reshape(nt,nx)
                            
                            unc_all[v][xres][tres][inventory1][inventory2] = self.vstack_conditional(unc_all_i, unc_short_i)

    def parse_observations_one(self, postpr):
        
        obs_one = postpr.read_postprocessed_observations()
        for v in ['prior','posterior','true']:           
            
            for site in postpr.get_obs_sites():
                obs_all_i = self.obs_all[v][site]                
                obs_one_i = obs_one[v][site]
                self.obs_all[v][site]['dates'] = self.append_conditional(obs_all_i['dates'], obs_one_i['dates'])
                self.obs_all[v][site]['co2']   = self.append_conditional(obs_all_i['co2'],   obs_one_i['co2'])
        
    def parse_dates_one(self, postpr):
        tress = np.unique(np.array(postpr.aggregate_resolutions)[:,1])
        start = postpr.start_after_spinup
        end   = postpr.end_before_spindown
        mid   = start + (end-start)/2.
        for tres in tress:
            if tres=='daily':
                tsteps = postpr.inversion.get_timesteps_opt_long()[postpr.mask_spinupdown_long]
                
            elif tres=='per_timestep':
                tsteps = postpr.inversion.get_timesteps_opt_full()[postpr.mask_spinupdown_tot]
                
            elif tres=='diurnal':
                hours  = np.arange(0,24, postpr.rc.opt_freq_diurnal)
                tsteps = np.array(datetime(mid.year, mid.month, mid.day, i) for i in hours)
                
            elif tres=='one_timestep':
                tsteps = np.array([datetime(mid.year,mid.month,mid.day)])
                
            self.dates_all[tres] = self.append_conditional(self.dates_all[tres], tsteps)
                
    def append_conditional(self, array_all, array_one):
        '''
        A conditional append that appends array_one to array_all, except when array_all
        is None, in which case array_one is the first entry and so array_all is set to
        array_one. This helps for filling in the containers that are by default set to
        None.
        '''
        
        if array_all is None:
            return array_one
        else:
            return np.append(array_all, array_one)
        
    def vstack_conditional(self, array_all, array_one):
        
        if array_all is None:
            return array_one
        else:
            return np.vstack([array_all, array_one])

    def get_timewindows_inversions(self):
        '''
        Get timewindows of the individual inversions that together form the longterm
        inversion.
        '''
        
        startdate = self.startdate_full
        self.startdates_all = []
        self.enddates_all   = []
        self.startdates_nospin = []
        self.enddates_nospin = []
        while startdate<self.enddate_full:
            
            start_full = startdate               - self.dt_spinup
            end_full   = startdate + self.dt_inv + self.dt_spindown - timedelta(days=1)
            
            # Account for 25-hours backward timesteps, analogous to run_inversion.py
            start_full += timedelta(seconds=25*3600)
            
            self.startdates_all.append(start_full)
            self.enddates_all.append(end_full)
            
            startdate += self.dt_inv