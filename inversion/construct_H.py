#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 12:53:07 2023

@author: nauss
"""


import inversion as inv
import functions_stijn as st
from inversion_base import InversionBase
import enhancement_calculator as enhc
import numpy as np
from construct_y import ConstructorY
import os,time
from datetime import datetime,timedelta

inversion_name = 'baseAKLNWP_base'

class ConstructorH(InversionBase):
    '''
    A class to construct the Jacobian. The Jacobian in our case consists of the
    (pre-)calculated enhancements on the inversion grid for each observation(/NAME run). 
    The constructor reads the pre-calculated enhancements and regrids those to the
    specific inversion grid we want to optimize on. Then, depending on which sites
    are included in the inversion and how we optimize them, it constructs the Jacobian.
    E.g., if we optimize site-to-site gradients, we also need to put gradients in
    the Jacobian.
    '''
    
    def __init__(self, inversion_name, modification_kwargs={}):
        super().__init__(inversion_name, modification_kwargs)
        
    def construct_H(self):
        self.read_enhancements_basegrid_all()
        self.regrid_enhancements_to_invgrid()
        self.aggregate_time_to_opt_freq()
        self.select_obs_timesteps_from_jacob()
        self.convert_jacobian_to_2dmatrix()
        self.write_Hmatrix()
        
    def read_enhancements_basegrid_all(self):
        self.determine_dates_to_read()
        self.enh_basegrid = {}
        for domain in self.rc.domains:
            priors_to_read = self.rc.priors_to_opt[domain]
            self.enh_basegrid[domain] = np.zeros((len(self.rc.sites), self.ndate, len(priors_to_read), self.rc.nstepNAME, self.rc.nx_base[domain], self.rc.ny_base[domain]))
            for ipri,prior in enumerate(priors_to_read):
                self.enh_basegrid[domain][:,:,ipri] = self.read_enhancements_basegrid_onedomain_oneprior(domain, prior)
                self.enh_basegrid[domain][:,:,ipri] *= self.get_scaling_factor_prior(prior)
        
    def determine_dates_to_read(self):
        '''
        For now we just read all possible dates, even though not each site has
        observations on the same dates. This makes regridding easier and we can
        subselect the correct dates later.
        '''
        
        constrY = ConstructorY(self.inversion_name, self.modification_kwargs)
        constrY.rc.inversion_label = self.rc.inversion_label
        self.ysites,self.ydates,_ = constrY.read_yvector()
        
        # usites is the observational quantity that is optimized, so can be gradients too
        self.usites = self.find_unq_sites_in_right_order(self.ysites)
        self.dates_to_read = np.sort(np.unique(self.ydates))
        self.ndate = len(self.dates_to_read)
        self.nobs = len(self.ydates)
        
    def find_unq_sites_in_right_order(self, ysites):
        '''
        I want the unique sites in the list ysites that contains sites per observation
        and I want them in the order they appear in ysites, so that my jacobian and
        observational vector match. Numpy.unique automatically sorts the result,
        so I need to "unsort".
        '''
        
        unq_sites, idx = np.unique(ysites,return_index=True)
        return unq_sites[np.argsort(idx)]
        
    def read_enhancements_basegrid_onedomain_oneprior(self, domain, prior):
        enh = np.zeros((len(self.rc.sites), self.ndate, self.rc.nstepNAME, self.rc.nx_base[domain], self.rc.ny_base[domain]))
        for idate,date in enumerate(self.dates_to_read):
            for isite,site in enumerate(self.rc.sites):
                enhi = self.read_enhancements_basegrid(date, domain, prior, site)[date.hour] # (NAMEstep, ny, nx)
                enhi = np.swapaxes(enhi, 1, 2) # (NAMEstep, nx, ny)
                enh[isite,idate] = enhi
        return enh
        
    def regrid_enhancements_to_invgrid(self):
        '''
        An easy regridding (coarsening) routine that assumes that the output dimensions fit
        an equal number of times in the original dimensions. 
        This is a good (/necessary) assumption, because the "original" grid is already 
        the NAME output regridded to a coarser grid, and regridding twice can introduce 
        unnecessary errors. This way, this second regridding introduces no errors.
        '''
        
        self.enh_invgrid = {}
        for domain in self.rc.domains:
            nx_in , ny_in  = self.rc.nx_base[domain], self.rc.ny_base[domain]
            nx_out, ny_out = self.rc.nx_inv[domain], self.rc.ny_inv[domain]
            dx, dy = int(nx_in/nx_out), int(ny_in/ny_out)
            enhi = self.enh_basegrid[domain]
            enhi = enhi.reshape(len(self.rc.sites), self.ndate, len(self.rc.priors_to_opt[domain]), self.rc.nstepNAME, nx_out, dx, ny_out, dy)
            self.enh_invgrid[domain] = enhi.sum(axis=(-1,-3))
        del(self.enh_basegrid)
            
        
    def get_domain_bounds(self, domain):
        if self.outerb[domain] is None:
            lonb,latb = inv.getLonLatNAME(domain, bounds=True)
            lonb,latb = [lonb.min(),lonb.max()], [latb.min(),latb.max()]
        else:
            lonb = np.array(self.outerb[domain]['lon'])
            latb = np.array(self.outerb[domain]['lat'])
        return lonb, latb
        
    def aggregate_time_to_opt_freq(self):
        '''
        Now we convert the hourly site enhancements to state vectors,
        meaning we aggregate in time to our optimization frequency.
        '''
        
        self.jacobian_full = {}
        timesteps_long = self.get_timesteps_opt_long(pos='start')
        nt_opt = self.get_ntimestep_opt()
        dt_diurnal = self.rc.opt_freq_diurnal # In hours
        for domain in self.rc.domains:
            priors_to_read = self.rc.priors_to_opt[domain]
            self.jacobian_full[domain] = np.zeros((len(self.rc.sites), self.ndate, len(priors_to_read), nt_opt, self.rc.nx_inv[domain], self.rc.ny_inv[domain]))
            
            # Loop over the observations / NAME simulations
            for idate,date in enumerate(self.dates_to_read):
                enhi = self.enh_invgrid[domain][:,idate,:,:,:]
                timestepsNAME = np.array([date-timedelta(seconds=3600*i) for i in range(self.rc.nstepNAME)])
                hoursNAME = np.array([d.hour for d in timestepsNAME])
                
                # Attribute NAME enhancements from one simulation to optimization timesteps
                istep = 0
                for tstep in timesteps_long: # Between days
                    dt_long = inv.get_timestep_from_freq(tstep, self.rc.opt_freq_longterm)
                    mask_long = (timestepsNAME>=tstep) & (timestepsNAME<(tstep+dt_long))
                    
                    for hour in range(0,24, dt_diurnal): # Within days
                        mask_diurnal = (hoursNAME>=hour) & (hoursNAME<(hour+dt_diurnal))
                        mask = (mask_long & mask_diurnal)
                        
                        self.jacobian_full[domain][:,idate,:,istep,:,:] += enhi[:,:,mask].sum(axis=2)
                        istep += 1
    
    def select_obs_timesteps_from_jacob(self):
        '''
        Up to now we have done all operations for all possible timesteps. Here,
        we select for each site only those timesteps that are actually found
        in the observation vector.
        '''
        
        self.jacobian_dict = {}
        
            
        for domain in self.rc.domains:
            self.jacobian_dict[domain] = {}
            for isite,site in enumerate(self.usites):
                
                if self.rc.obs_method=='gradient':
                    site1, site2 = site.split('-')
                    isite1 = self.rc.sites.index(site1)
                    isite2 = self.rc.sites.index(site2)
                    jac_i = self.jacobian_full[domain][isite1] - self.jacobian_full[domain][isite2]
                    
                elif self.rc.obs_method=='per_site':
                    
                    isite = self.rc.sites.index(site)
                    jac_i = self.jacobian_full[domain][isite]
                    
                dates_for_site = self.get_dates_for_site(site)
                idxs = [np.where(self.dates_to_read==d)[0][0] for d in dates_for_site]
                self.jacobian_dict[domain][site] = jac_i[idxs]
                
    def get_dates_for_site(self,site):
        mask = (self.ysites==site)
        return self.ydates[mask]
                
    def convert_jacobian_to_2dmatrix(self):
        '''
        Currently each variable still has their own dimension:
            (domain, site, ndate_obs, nt_opt, nprior, ny, nx)
        Here we flatten the quite complicated jacobian dictionary to the 2-D array that is used
         in the inversion calculations, with dimensions (nobs,nstate). The order of dimensions
         has to be the same as in the state, which is: (nprior, nt_opt, ndomain, nx, ny)
        '''
        
        nstate = self.get_nstate_opt()
        self.Hmatrix = np.zeros((self.nobs,nstate))
        
        for isite,site in enumerate(self.usites):
            nobs_i = len(self.get_dates_for_site(site))
            if nobs_i>0:
                idx_sites = np.where(self.ysites==site)[0]
                for prior in self.priors_all:
                    domains_i = self.get_domains_for_prior(prior)
                    for domain in domains_i:
                        ipri  = self.rc.priors_to_opt[domain].index(prior)
                        jac_i = self.jacobian_dict[domain][site][:,ipri]
                        
                        x, priors, domains = self.read_state()
                        idx_state = np.where( ((priors==prior) & (domains==domain)) )[0]
                        
                        jac_i_2d = jac_i.reshape(nobs_i,-1)
                        for i,idxi in enumerate(idx_sites):
                            self.Hmatrix[idxi,idx_state] = jac_i_2d[i]
                                               
    def write_Hmatrix(self):
        fname = self.get_filename_Hmatrix()
        np.save(fname, self.Hmatrix)
    
    def get_mask_site(self, site):
        return (self.ysites==site)
        
        
if __name__ == '__main__':
    constrH = ConstructorH('baseAKLNWP_base')
    constrH.construct_H()
    
    import matplotlib.pyplot as plt
    
    jac = constrH.Hmatrix
    djac = constrH.jacobian_dict
    
    gradients = np.unique( constrH.read_yvector()[0] )
    fig,ax = plt.subplots(1,1,figsize=(10,4))
    for igrad, grad in enumerate(gradients):
        y = constrH.Hmatrix.sum(axis=1)
        dates_i = constrH.get_dates_for_site(grad)
        mask = (constrH.ysites==grad)
        
        ax.plot(dates_i, y[mask], 'o-', label=grad)
    ax.legend()
    
























