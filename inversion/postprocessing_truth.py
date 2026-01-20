#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 11:21:05 2025

@author: nauss

A script to postprocess the truth used in the OSSE. This is written for the OSSE where
I generate a truth outside the inversion based on the prior perturbed by a random vector
generated from the B (& R) uncertainty matrix. In principle it's analogous to normal post-processing,
except I don't postprocess uncertainties, since I never generate Btruth explicitly,
and I don't postprocess obs, because those are obviously included in the original 
inversions. So basically it's only for emissions. (observations also not needed, since
they are included in the inversions)

I do it separately, because it's not necessarily configured the same way as the inversion
in which it is used, e.g., it might have a different grid set-up. But the aggregation 
etc works similarly, so mostly I'm inheriting from the original postprocessing.

Note that the truth is generated for the whole inversion period (e.g., one year)
in one go, compared to the inversion which is done per month, and even then might
not cover the exact same period as the truth. Therefore, I need to select from the
truth the time period that is covered by the ensemble of inversions.
"""

from postprocessing import Postprocessing
import numpy as np
import inversion as inv
import functions_stijn as st
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
from construct_x import ConstructorX
from inversion_main import Inversion

class Postprocess_truth(Postprocessing):
    
    
    def __init__(self, inversion_name, startdate, enddate):
        
        super().__init__(inversion_name, rc_kwargs={}, dt_spinup=timedelta(days=0), dt_spindown=timedelta(days=0))
        self.startdate = startdate
        self.enddate   = enddate
        
        # I need to construct a state vector for this specific time window mainly so I have domain and prior per state
        # element available.
        kwargs = {'date_start':['direct', startdate], 'date_end':['direct',enddate]}
        self.cx = ConstructorX(inversion_name, kwargs)
        self.cx.construct_x()
        
        self.tsteps_sel = self.cx.get_timesteps_opt_full()
        
    def run_standard_postprocessing(self):
        
        self.read_xtrue_specific_timeperiod()
        self.inversion.setup_inversion_grid()
        self.inversion.remove_file_emis_on_invgrid()
        self.inversion.get_emissions_on_inversion_grid()
        self.postprocess_emissions()
        
    def postprocess_emissions(self):
        # Just a simpler version of the postprocessing one
        
        self.emis_agg   = self.setup_container_dict_emis()['prior']
        self.emis_prior = self.setup_container_dict_emis()['prior']
        
        # I also save prior fluxes because the grid is different from the inversion
        emis_pri  = self.inversion.emis_inv_vec
        emis_true = self.xtrue*emis_pri
        
        for xres,tres in self.aggregate_resolutions:
            
            
            for prior in self.get_unique_priornames():
                self.emis_agg[xres][tres][prior] = self.aggregate_array_onedim(emis_true, xres, tres, prior)
                self.emis_prior[xres][tres][prior] = self.aggregate_array_onedim(emis_pri, xres, tres, prior)
            
            # Adding up like this doesn't really work for per_gridcell with priors
            # covering different domains
            if xres!='per_gridcell':
                self.emis_agg[xres][tres]['Total'] = 0.
                for prior in self.get_unique_priornames():
                    self.emis_agg[xres][tres]['Total'] += self.emis_agg[xres][tres][prior]
        
        self.save_aggregated_emissions()
        

    def read_xtrue_specific_timeperiod(self):
        '''
        xtrue on file covers the full year + spinup+down. At least I want to exclude
        spin-up and spin-down, but sometimes I might also not do the inversion for
        the full year.
        '''
        
        xtrue_full  = self.inversion.read_xtrue()
        tsteps_full = self.inversion.get_timesteps_opt_full()
        
        tstep_per_state = []
        for prior in self.inversion.priors_all:
            domains_i = self.inversion.get_domains_for_prior(prior)
            nstep = self.inversion.get_ntimestep_opt()
            for itime in range(nstep):
                for domain in domains_i:
                    ngrid = self.inversion.rc.nx_inv[domain]*self.inversion.rc.ny_inv[domain]
                    tstep_per_state += [tsteps_full[itime]]*ngrid
                    
        tstep_per_state = np.array(tstep_per_state)
        mask = [t in self.tsteps_sel for t in tstep_per_state]
        self.xtrue = xtrue_full[mask]
        
        # Now I can replace startdate and enddate, because I've selected xtrue
        # I.e., in all aggregation I can just assume that this is the relevant timeperiod
        kwargs = {'date_start':['direct', self.startdate], 'date_end':['direct',self.enddate]}
        self.inversion = Inversion(self.inversion_name, kwargs)
        self.rc.date_start = self.startdate
        self.rc.date_end   = self.enddate
        self.make_masks_spinupdown_state()
        

if __name__ == "__main__":
    inversion_name = 'baseAKLNWP_truth'
    startdate = datetime(2022,1,1) + timedelta(seconds=3600*25)
    enddate   = datetime(2022,2,25)
            
    pp_true = Postprocess_truth(inversion_name, startdate, enddate)
    pp_true.run_standard_postprocessing()
        
        
        
        
        
        
        
        
        
        



