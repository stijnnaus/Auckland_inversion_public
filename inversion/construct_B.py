#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 12:39:20 2023

@author: nauss

Construct the prior error covariance matrix. We save this in two parts: the 
temporal correlation matrix (Btemp) and the spatial correlation matrix (Bspat).
"""


import inversion as inv
import functions_stijn as st
import matplotlib.pyplot as plt
from inversion_base import InversionBase
import enhancement_calculator as enhc
import numpy as np
import os,time
from datetime import datetime,timedelta

class ConstructorB(InversionBase):
    
    def __init__(self, inversion_name, modification_kwargs={}):
        super().__init__(inversion_name, modification_kwargs)
    
    def construct_B(self):
        self.construct_dx_dt()
        self.Bmatrix = self.create_B_from_dt_dx( self.dt_diurnal_2d, self.dt_longterm_2d, self.dx_2d)
        self.write_Bmatrix()
        
        t0 = time.time()
        self.Bmatrix_inv = np.linalg.inv(self.Bmatrix)
        print("Inverting Bmatrix took: %2.2fs"%(time.time()-t0))
        self.write_Bmatrix_inv()
        
        # del(self.Bmatrix, self.Bmatrix_inv)
    
    def construct_dx_dt(self):        
        self.construct_dx()
        self.write_dx()
        self.construct_dt()
        self.write_dt()
            
    def construct_dx(self):
        '''
        The spatial correlations in the prior(s). We calculate this from the center of
        the coarse grid cells. Difficulty enters with the nested domains; e.g., sometimes
        inner and outer domain grid cell centers are the same, resulting in too-high
        correlations. For now we assume that the maximum correlation is capped by
        the grid cell size.
        We first calculate the spatial correlations of the three different domains; then,
        we combine this with the fact that we do not optimize all priors in all domains.
        '''
    
        self.get_gridcell_centers()
        self.calculate_gridcell_distances()
            
    def construct_dt(self):
        '''
        Populate the matrix for within-day (diurnal) and between-day (longterm)
        timesteps with the difference in timesteps. This can be combined with
        correlation length for populating the covariance matrix. We don't combine
        it with correlation length here, because different priors can have different
        correlation lengths, so what Bt results from the dt's depends on which
        priors we are combining them for.
        '''
        
        self.construct_dt_diurnal()
        self.construct_dt_longterm()
            
    def construct_dt_diurnal(self):
        dt   = self.rc.opt_freq_diurnal # timestep in hours
        
        tsteps_diurnal = np.arange(0, 24, dt)
        self.dt_diurnal_2d = np.abs(tsteps_diurnal[None,:] - tsteps_diurnal[:,None])
        # The maximum difference is 12 hours, since diurnal cycle is cyclical 
        # I.e., hour 0 is closer to hour 23 than to hour 12
        self.dt_diurnal_2d[self.dt_diurnal_2d>12] = 24 - self.dt_diurnal_2d[self.dt_diurnal_2d>12]
    
    def construct_dt_longterm(self):
        opt_freq = self.rc.opt_freq_longterm
        
        tsteps_long = inv.get_opt_timesteps(self.rc.date_start, self.rc.date_end, opt_freq, self.rc.nstepNAME)
        nstep = len(tsteps_long)
        dt_2d_datetime = np.abs(tsteps_long[None,:] - tsteps_long[:,None])
        self.dt_longterm_2d = np.zeros_like(dt_2d_datetime, dtype=float)
        for i in range(nstep):
            for j in range(nstep):
                self.dt_longterm_2d[i,j] = dt_2d_datetime[i,j].days + dt_2d_datetime[i,j].seconds/(24*3600)
        
    def create_B_from_dt_dx(self, dt_diurnal, dt_longterm, dx):
        '''
        Create the error covariance matrix from three 2-D matrices:
            - dt_diurnal: the distance between within-day timesteps in hours
            - dt_longterm: the distance between between-day(/week) timesteps in days
            - dx: the distance between grid cells in m, containing all combinations
                    of domains and priors per domain
        '''
        
        # For some scaling procedures I need emissions too, despite B in principle being fractional
        self.get_emissions_on_inversion_grid()
        
        # 1) Create the unscaled prior error matrix
        nstate = self.get_nstate_opt()
        Bmatrix = np.zeros((nstate, nstate))
        for pri1 in self.priors_all:
            
            label1 = self.rc.prior_error_labels[pri1]
            
            # Read error settings prior 1
            error_config1  = self.rc.prior_errors[pri1]
            Lt_long1       = error_config1['L_temp_long']
            Lt_diurn1      = error_config1['L_temp_diurnal']
            Lx1            = error_config1['L_spatial']
            
            # Indices in full covariance matrix
            idx1a = self.get_prior_idx_in_state(pri1)    # idx1a has length (nt*nx)
            
            # Indices in spatial covariance matrix
            idx1b = self.get_prior_idx_in_Bspatial(pri1) # idx1b has length (nx)
            
            for pri2 in self.priors_all:
                
                label2 = self.rc.prior_error_labels[pri2]
                idx2a = self.get_prior_idx_in_state(pri2)
                idx2b = self.get_prior_idx_in_Bspatial(pri2)
                
                if label1==label2:
                    # Priors are correlated
                    
                    # Read error settings prior 2
                    error_config2 = self.rc.prior_errors[pri2]
                    Lt_long2      = error_config2['L_temp_long']
                    Lt_diurn2     = error_config2['L_temp_diurnal']
                    Lx2           = error_config2['L_spatial']
                    
                    dx_i     = dx[idx1b,:][:, idx2b]
                    Bs       = self.calculate_cov_matrix(dx_i       , Lx1      , Lx2)
                    Bt_diurn = self.calculate_cov_matrix(dt_diurnal , Lt_diurn1, Lt_diurn2)
                    # Bt_long has separate treatment because of option weekend/weekday split
                    Bt_long  = self.calculate_cov_matrix_Bt_long(dt_longterm, Lt_long1 , Lt_long2, label1)                        
                    
                    Bt = np.kron(Bt_long, Bt_diurn)
                    Bmatrix[idx1a[:,np.newaxis],idx2a] = np.kron(Bt, Bs)
                    
                else:
                    # Priors are not correlated
                    Bmatrix[idx1a[:,np.newaxis],idx2a] = 0.
        
        # 2) Scale to the prescribed error per prior. For scaling the prior error
        #    to an aggregate total I need Bmatrix filled, so that's why I do it in two steps
        #    If this option is not used, B is just scaled to the prescribed (fixed) error
        scalars = self.get_Bmatrix_error_scalings(Bmatrix)
        
        for pri1 in self.priors_all:
            err1 = self.rc.prior_errors[pri1]['rel_error']
            
            err1 = err1*np.sqrt(scalars[pri1])
                
            idx_pri1 = self.get_prior_idx_in_state(pri1)
            
            for pri2 in self.priors_all:
                
                idx_pri2 = self.get_prior_idx_in_state(pri2)
                err2 = self.rc.prior_errors[pri2]['rel_error']
                
                err2 = err2*np.sqrt(scalars[pri2])
                
                print(pri1,pri2, Bmatrix[idx_pri1[:,np.newaxis],idx_pri2].shape, err1.shape, err2.shape)
                
                Bmatrix[idx_pri1[:,np.newaxis],idx_pri2] *= np.outer(err1, err2)
                
        # del(self.emis_inv_dict, self.emis_inv_vec)
        return Bmatrix
    
    def calculate_cov_matrix_Bt_long(self, dt, L1, L2, label):
        '''
        Calculate the error covariance matrix corresponding to the "long" time dimension.
        So this is the "between-day" correlation. It's done separately, because it has 
        the additional option to adjust correlations between weekdays and weekends, relevant
        for the anthropogenic fluxes.
        '''
        
        Bt_long  = self.calculate_cov_matrix(dt, L1 , L2)
        
        if label in self.rc.separate_weekend_error.keys():
            flag = self.rc.separate_weekend_error[label]['flag']
            corr = self.rc.separate_weekend_error[label]['corr'] # Correlation between weekends and weekdays
        else:
            flag = False
        
        if flag:
            days = self.get_timesteps_opt_long()
            idx_weekday, idx_weekend = st.get_idx_weekday_weekend(days)
            
            vec = np.ones_like(days, dtype=int)
            vec[idx_weekday] = -1
            m = np.outer(vec, vec)
            
            # m==1 for weekday-weekday and weekend-weekend correlations and -1 for weekend-weekday
            Bt_long[m==-1] *= corr  
            
        return Bt_long
            
    
    def calculate_cov_matrix(self, dx, L1, L2):
        """
        Calculate an unscaled covariance matrix based on distance between grid cells
        (or timesteps) and a correlation length of prior 1 and of prior 2.
        Calculation based on:
            r12 = sqrt(r1*r2) = sqrt(exp(-dx/L1)*exp(-dx/L2)) = exp(-0.5*dx* ((1/L1) + (1/L2)))
        If L1=L2, simplifies to exp(-dx/L)
        
            dx : Distance between the different elements (spatial or temporal)
            L1 : Correlation length 1
            L2 : Correlation length 2
        
            dx, L1, L2 should all be in same units
        """
        
        Linv = 0.5*(1/L1 + 1/L2)
        B = np.exp( -dx*Linv )
        
        return B
        
    def get_Bmatrix_error_scalings(self, Bmatrix):
        """
        Retrieve and combine all the different types of error scaling we can apply
        in the inversion. Whether or not a certain type of scaling is included
        is checked inside each get_ function.
        
        Scalars is a dictionary per prior
        """
        
        scalars = self.get_scalars_for_minimum_uncertainty_per_timestep()
        scalars = self.get_scalars_for_unc_total_per_prior(Bmatrix, scalars)
        scalars = self.get_scalars_for_GPP_Resp(scalars)
        
        return scalars
        
    def get_scalars_for_minimum_uncertainty_per_timestep(self):
        '''
        Here, we impose a minimum absolute uncertainty per timestep. This helps
        give additional freedom to NEE in the prior during the day-night transition,
        and possibly can give the anthropogenic emissions some potential to 
        adjust in night-time. Because we use relative scaling, it only works if prior
        emissions are not zero over a timestep. I also clip to a max value, because 
        you don't want e.g., 1e999 scaling (don't think it ever happens but still).
        '''
        
        scalars = {}
                    
        for prior in self.priors_all:
            
            if 'min_abs_error' in self.rc.prior_errors[prior].keys():
                
                # Get emissions / timestep, summed over domains
                emis_i, nspace = 0., 0
                for domain in self.get_domains_for_prior(prior):
                    # Sum over lat,lon; leave time dimension; unit is kg/timestep
                    emii = np.abs(self.emis_inv_dict[domain][prior])
                    emis_i += emii.sum(axis=(1,2))
                    nt,nx,ny = emii.shape
                    nspace += nx*ny
                
                err_min_abs = self.rc.prior_errors[prior]['min_abs_error'] # in kg/timestep
                err_rel     = self.rc.prior_errors[prior]['rel_error']     # fractional
                err_abs     = err_rel*emis_i
                
                scalars_i = err_min_abs / err_abs
                scalars_i[scalars_i<1]   = 1.0    # Minimum error so we only increase error
                scalars_i[err_abs==0]    = 1.0    # Not much to do if prior (error) is zero...
                scalars_i[scalars_i>1e9] = 1e9    # Don't want the scalars to get too high
                
                # Move from shape nt to shape nstate (= nt, ndomain, nx, ny)
                scalars_i = np.repeat(scalars_i, nspace)
                
                scalars[prior] = scalars_i**2
            
            else:
                scalars[prior] = 1.0
            
        return scalars
    
    def get_scalars_for_GPP_Resp(self, scalars_other={}):
        '''
        For biosphere fluxes, it is recommend to have uncertainties relative to
        GPP and Respiration rather than relative to NEE. Here:
            
            1) We read NEE, GPP and Respiration fluxes on the inversion grid.
            2) Calculate absolute uncertainties per gridcell/timestep from prescribed
                percentage errors on GPP and Resp.
            3) Convert the absolute uncertainties to relative uncertainties on NEE.
        
        I.e., : final_relative_error = (sigma_GPP*GPP + sigma_Re*Re) / NEE
        Note that in this implementation, I should put the relative NEE error on 100%
        so that applying the scalar calculated here is just x1 so it is conserved.
        '''        
        
        scalars = {}
        for prior in self.priors_all:
            if self.check_if_scale_prior_error_to_GPP_Resp(prior):
                rel_err_NEE = self.get_scalars_for_GPP_Resp_one(prior)
                scalars[prior] = scalars_other[prior] * rel_err_NEE**2
                
            else:
                scalars[prior] = scalars_other[prior]
            
        return scalars
    
    def check_if_scale_prior_error_to_GPP_Resp(self, prior):
        
        if hasattr(self.rc, 'scale_err_to_GPP_Resp'): # So as not to break backwards compatibility
            if prior in self.rc.scale_err_to_GPP_Resp:
                return True
            else:
                return False
        else:
            return False
            
    
    def get_scalars_for_GPP_Resp_one(self, prior):
        '''
        More general function, since I need to do the same operation for 
        UrbanVPRM and BiomeBGC both
        
        Output format is 1-D vector shape (nt, nspace_prior)
        '''
        
        # Read NEE + GPP + Resp
        NEE  = self.get_emis_invgrid(prior, 'NEE')
        GPP  = self.get_emis_invgrid(prior, 'GEE')
        Resp = self.get_emis_invgrid(prior, 'Re')
        
        # Calculate absolute error per state element
        rel_err_GPP  = self.rc.scale_err_to_GPP_Resp[prior]['GPP']
        abs_err_GPP  = np.abs(rel_err_GPP*GPP)
        
        rel_err_Resp = self.rc.scale_err_to_GPP_Resp[prior]['Re']
        abs_err_Resp = np.abs(rel_err_Resp*Resp)
        
        # Convert to relative error per state element
        rel_err_NEE = (abs_err_GPP + abs_err_Resp) / np.abs(NEE)
        
        # Take care of very high scalars, or zero-NEE = inf rel errors
        rel_err_NEE[np.abs(NEE)==0]  = 1.0 # Case NEE = 0
        rel_err_NEE[rel_err_NEE>1e2] = 1e2 # Case NEE is near-zero
        
        return rel_err_NEE
    
    def get_emis_invgrid(self, prior, varb):
        """
        A slightly modified version for getting the emissions on the inversion grid,
        where I do want vector output but not for the whole state, just for one 
        prior. It also leaves the option to read GPP or Respiration instead of NEE
        for biosphere priors (and in principle to read specific Mahuika categories).
        """
        
        domains = self.get_domains_for_prior(prior)
        
        # Read as dictionary
        emis_dict = self.read_preprocessed_emis_and_regrid({d:[prior] for d in domains}, varb=varb)
        
        # Parse to vector
        emis_vec  = []
        nstep = self.get_ntimestep_opt()
        for itime in range(nstep):
            for domain in domains:
                nx,ny = self.rc.nx_inv[domain],self.rc.ny_inv[domain]
                for ix in range(nx):
                    for iy in range(ny):
                        emis_vec.append(emis_dict[domain][prior][itime,iy,ix])
                        
        return np.array(emis_vec)
        
    def get_scalars_for_unc_total_per_prior(self, Bmatrix, scalars_other):
        """
        In the standard set-up, if I prescribe a 50% prior error, that becomes
        the error on the per-gridcell, per-timestep prior value. Should you then
        aggregate over a full grid, and the whole timeperiod, then the aggregate uncertainty
        is much lower than 50%. Sometimes that is not what we want, and so this function gives 
        the option to scale the prior error matrix such that the 50% value reflects the 
        aggregate uncertainty over all gridcells, domains and timesteps. 
        
        The function generates one scalar per prior, and this is exactly the scalar
        that if the subpart of B that describes uncertainties of prior1 is multiplied
        by scalar[prior1], then the aggregate uncertainty for that prior is 
        (e.g.) 50% of total (absolute) emissions of that prior.
        
        The equation for the calculation is:
            scalar = sum( Bpri . (Emi x Emi) ) / sum(Emi^2),
            where I note that sum( Bpri . (Emi x Emi) ) is the aggregate uncertainty
        
        Two notes: 
        1) This is per prior. So it does work for one prior in multiple domains.
             But if you have Mahuika-Auckland in the inner domain correlated with
             e.g., EDGAR in the outer domain, then there's a 50% uncertainty on the
             aggregate of Mahuika-Auckland, 50% for EDGAR, and on top of that you
             have the cross-correlations between the two. So uncertainty on anthropogenic
             is >50%. But I think this is better than combining the two, since I'm
             mainly interested in scaling Mahuika-Auckland and UrbanVPRM correctly,
             and I don't want to convolute that with the whole of NZ.
             
        2) This gives a 50% uncertainty in the total aggregate. So if I do a 1 month
            inversion the uncertainty in the 1-month total will be 50%. If I do a 
            6 month inversion the uncertainty in the 6-month total will be 50%. 
            That makes it so that comparing a 1-month to a 6-month inversion with this
            option turned on will give weird results. The easiest solution if I want to 
            consistently compare inversions run over different time-windows is to turn 
            this scaling off and do some scaling "off-line" if necessary.
            (then again, if I want to compare different grid set-ups this option
             is almost a must)
            
        """
        
        scalars = {}
        for pri in self.priors_all:
            if self.rc.scale_unc_to_total[pri]:
                print("Scale to total,", pri)
                print(self.rc.scale_unc_to_total)
                idx    = self.get_prior_idx_in_state(pri)
                emis_i = self.emis_inv_vec[idx]
                B_i    = Bmatrix[idx,:][:,idx]
                
                # What I want the aggregate uncertainty to be
                emis_i = np.abs(emis_i) # Absolute emissions, otherwise for NEE it can get very small
                emis_i = emis_i*scalars_other[pri] # Take into account other scaling of the prior error
                esum = np.sum(emis_i)**2 
                
                # What the aggregate uncertainty is currently (note that I don't explicitly
                # calculate the outer product of emissions since I suspect it would use more memory?
                # I tested and this gives the same result)
                Bsum = np.sum(B_i*emis_i[:,np.newaxis]*emis_i[np.newaxis,:])
                
                scalars[pri] = esum/Bsum
            
            else:
                scalars[pri] = 1.0*scalars_other[pri]
            
        return scalars
    
    def get_gridcell_centers(self):
        self.lonc_2d, self.latc_2d = np.array([], dtype=float), np.array([], dtype=float)
        self.dx_priors = np.array([], dtype=object)
        for prior in self.priors_all:
            domains_i = self.get_domains_for_prior(prior)
            for domain in domains_i:
                lonc_i, latc_i = self.get_gridcell_centers_domain(domain, out_2d=True)
                lonc_i = lonc_i.flatten()
                latc_i = latc_i.flatten()
                self.lonc_2d = np.append(self.lonc_2d, lonc_i)
                self.latc_2d = np.append(self.latc_2d, latc_i)
                self.dx_priors = np.append(self.dx_priors, [prior]*len(lonc_i))
                
    def calculate_gridcell_distances(self):
        
        ngrid = len(self.lonc_2d)
        self.dx_2d = np.zeros((ngrid,ngrid))
        for i in range(ngrid):
            lon1, lat1 = self.lonc_2d[i], self.latc_2d[i]
            self.dx_2d[i] = st.calc_distance_between_coords(lon1, lat1, self.lonc_2d, self.latc_2d)
            
        self.ensure_minimum_gridcell_distances()
    
    def get_gridcell_centers_domain(self, domain, out_2d=False):
        
        lonb,latb = self.get_gridcell_bounds_domain(domain, out_2d=out_2d)
        if out_2d:
            lonc = 0.5*(lonb[1:,1:]+lonb[:-1,:-1])
            latc = 0.5*(latb[1:,1:]+latb[:-1,:-1])
        else:
            lonc = 0.5*(lonb[1:]+lonb[:-1])
            latc = 0.5*(latb[1:]+latb[:-1])
        
        return lonc,latc
    
    def ensure_minimum_gridcell_distances(self):
               
        self.calculate_gridcell_diag_length()
        ngrid = len(self.gridcell_diag_length)
        diag2d_a = np.tile(  self.gridcell_diag_length, ngrid).reshape(ngrid,ngrid)
        diag2d_b = np.repeat(self.gridcell_diag_length, ngrid).reshape(ngrid,ngrid)
        diag2d = 0.5*(diag2d_a+diag2d_b)
        
        # 1. Distance between grid cells can never be smaller than the average length of their diagonals x 0.5
        self.dx_2d = np.max([diag2d*0.5, self.dx_2d], axis=0)
        # 2. Except of course on the diagonal (where distance is always 0)
        np.fill_diagonal(self.dx_2d, 0.0)
        
    def calculate_gridcell_diag_length(self):
        '''
        Create a 1-D vector with the diagonal length of each grid cell in each
        domain. This is used as a lower threshold for distance between grid cells.
        '''
        
        self.gridcell_diag_length = np.array([], dtype=float)
        # Retrieve grid cell size
        for prior in self.priors_all:
            domains_i = self.get_domains_for_prior(prior)
            for domain in domains_i:
                lonb_i, latb_i = self.get_gridcell_bounds_domain(domain, out_2d=True)
                # Top left corner to bottom right corner of each grid cell
                dist_diag = st.calc_distance_between_coords(lonb_i[:-1,:-1], latb_i[:-1,:-1], lonb_i[1:,1:], latb_i[1:,1:])
                self.gridcell_diag_length = np.append(self.gridcell_diag_length, dist_diag)
        
    def write_dx(self):
        fname = self.get_filename_dx()
        np.save(fname, self.dx_2d)
        fname = self.get_filename_dx('_priors')
        np.save(fname, self.dx_priors)
        
    def write_dt(self):
        fname = self.get_filename_dt('diurnal')
        np.save(fname, self.dt_diurnal_2d)
        fname = self.get_filename_dt('longterm')
        np.save(fname, self.dt_longterm_2d)
        
    def write_Bmatrix(self):
        fname = self.get_filename_Bmatrix()
        np.save(fname, self.Bmatrix)
        
    def write_Bmatrix_inv(self):
        fname = self.get_filename_Bmatrix_inv()
        np.save(fname, self.Bmatrix_inv)
    
    def get_domains_for_prior(self, prior):
        '''
        We prescribe and save as a dictionary domains as keys and priors as items.
        Here we invert this: Select the domains in which we optimize a prior.
        '''
        
        domains = []
        for domain,priors in self.rc.priors_to_opt.items():
            if prior in priors:
                domains.append(domain)
                    
        return domains
    
    
    def get_gridcell_bounds_domain(self, domain, out_2d=False):
        
        lonb, latb = self.get_domain_bounds(domain)
        lonb = np.linspace(lonb[0], lonb[1], self.rc.nx_inv[domain]+1)
        latb = np.linspace(latb[0], latb[1], self.rc.ny_inv[domain]+1)
        
        if not out_2d:
            return lonb,latb
        else:
            nlon, nlat = len(lonb), len(latb)
            
            lonb2d = np.zeros((nlon, nlat))
            for i in range(nlat):
                lonb2d[:,i] = lonb
                
            latb2d = np.zeros((nlon,nlat))
            for i in range(nlon):
                latb2d[i,:] = latb
            
            return lonb2d, latb2d
    
    
    
    
    
if __name__ == "__main__":
    
            
    inversion_name = 'baseAKLNWP_base_1month'
    inversion_label = '1month'
    
    from construct_x import ConstructorX
    constr = ConstructorX(inversion_name)
    constr.rc.inversion_label = inversion_label
    constr.construct_x()
    
    constrB = ConstructorB(inversion_name)
    constrB.rc.inversion_label = inversion_label
    constrB.construct_dx_dt()
    
    dt_diurn, dt_long, dx = constrB.dt_diurnal_2d, constrB.dt_longterm_2d, constrB.dx_2d
    Bmatrix = constrB.create_B_from_dt_dx(dt_diurn, dt_long, dx)
    
    priors = constrB.priors_all
    nprior = len(priors)
    
    fig,ax = plt.subplots(nprior, nprior, figsize=(10,10))
    for ipri1,pri1 in enumerate(priors):
        idx1a = constrB.get_prior_idx_in_state(pri1) 
        for ipri2,pri2 in enumerate(priors):
            idx2a = constrB.get_prior_idx_in_state(pri2) 
            Bi = Bmatrix[idx1a,:][:,idx2a]
            axi = ax[ipri1,ipri2] if len(priors)>1 else ax
            axi.imshow(Bi, vmin=0, vmax=0.25)

    plt.tight_layout()
    plt.savefig("Figures/plot_Bmatrix.png")
    
    # for i,B in enumerate([Bt_diurn,Bt_long,Bt]):
    #     fig, ax = plt.subplots(1,1, figsize=(10,10))
        
    #     ax.imshow(B, vmin=0, vmax=1)

