# -*- coding: utf-8 -*-
"""

Postprocessing code for the inversion.
Useful so I can keep it out of the analysis scripts, and have the generic operations
I might want to do with each inversion cleanly coded in one place.

It's mainly for doing aggregation.

Created on Wed May  8 15:42:54 2024

@author: stijn
"""

import numpy as np
from datetime import datetime, timedelta
from inversion_main import Inversion
import os
import pickle

class Postprocessing(object):
    """
    Postprocessing routine for the inversion. This is so I can keep the inversion
    itself clean, focused on the calculating the full posterior matrices.
    
    The postprocessing will:
        - Convert state vectors to prior, posterior and, optionally, true emissions, aggregated
            at different, hard-coded spatial and temporal resolutions
        - Aggregate prior, posterior and, optionally, true B matrices in the same way
        - Calculate prior, posterior, and true observations and write to file
        - Calculate some required inversion statistics, such as chi-sq, aggregated correlation factors, etc
    """
    
    def __init__(self, inversion_name, rc_kwargs={}, dt_spinup=timedelta(days=0), dt_spindown=timedelta(days=0)):
        self.inversion_name = inversion_name
        self.inversion = Inversion(inversion_name, rc_kwargs)
        
        self.path_out = '%s/aggregated/'%self.inversion.get_path_inversion_output()
        if not os.path.exists(self.path_out):
            os.makedirs(self.path_out)
        
        self.rc = self.inversion.rc # shortcut
        
        # Spinup/down periods are completely dropped in the postprocessing
        self.dt_spinup   = dt_spinup
        self.dt_spindown = dt_spindown
        self.make_masks_spinupdown_state()
        
        self.define_aggregate_resolutions()
        
    def make_masks_spinupdown_state(self):
        '''
        Make masks that can be used to mask out the spinup and spindown period in state vector space 
        (so also applicable to Bmatrix, but not to obs).
        These are applied to weights_time, which means it will directly propogate both to emissions and Bmatrix.
        There's two masks we need: one to apply to the dimension with all timesteps and 
        one to apply to the dimension with between-day timesteps (so number of days).
        The third resolution (diurnal cycle) isn't affected by spin-up/down.
        '''
        
        # Start and end of simulation after cutting off spinup/down
        # date_start is first obs used, but emissions go back nstepNAME
        start_emis = self.rc.date_start - timedelta(seconds=3600*self.rc.nstepNAME)
        self.start_after_spinup  = start_emis       + self.dt_spinup
        self.end_before_spindown = self.rc.date_end - self.dt_spindown
        
        # We want to include the whole final day, not just the first hour
        self.end_before_spindown = datetime(self.end_before_spindown.year, self.end_before_spindown.month, self.end_before_spindown.day, 23, 59)
        
        # Mask for full timesteps
        tsteps_tot = self.inversion.get_timesteps_opt_full()
        self.mask_spinupdown_tot  = (tsteps_tot  >= self.start_after_spinup) & (tsteps_tot  <= self.end_before_spindown)
        
        # Mask for day-to-day steps
        tsteps_long = self.inversion.get_timesteps_opt_long()
        self.mask_spinupdown_long = (tsteps_long >= self.start_after_spinup) & (tsteps_long <= self.end_before_spindown)
        
    def run_standard_postprocessing(self):
        
        self.inversion.read_inversion_input(which_bmatrix=None)
        self.inversion.read_inversion_output(read_bopt=False)
        self.inversion.setup_inversion_grid()
        self.inversion.get_emissions_on_inversion_grid()
        self.postprocess_emissions()
        self.postprocess_Bmatrices()
        self.postprocess_Gmatrix()
        self.postprocess_observations()
        self.remove_unaggregated_matrices()
        self.calculate_inversion_statistics()
        
    def define_aggregate_resolutions(self):
        '''
        Define the resolutions I want to aggregate to. This is a list with combinations
        of spatial resolutions and temporal resolutions.aa
        '''
        
        self.aggregate_resolutions = [['per_domain','per_timestep'], 
                                      ['per_domain','diurnal'], 
                                      ['per_domain','daily'], 
                                      ['per_domain','one_timestep'], 
                                      
                                      ['all_domains','per_timestep'], 
                                      ['all_domains','diurnal'], 
                                      ['all_domains','daily'], 
                                      ['all_domains','one_timestep'],
                                      
                                      ['per_gridcell','one_timestep'], 
                                      ]
        
    
        
    def postprocess_emissions(self):
        """
        Postprocess / aggregate emissions to a dictionary with nesting as:
            - prior/posterior/truth
                - xres
                    -tres
                        - inventory
                            Array of shape (nt, nx)
                        
        Emissions will be in [kg/timestep/gridcell or domain], since this is the unit
        that makes it easy to aggregate and combine with the B matrix. I can always
        convert to per area / per second later.
        
        The reason for this awkward structuring of postprocessed emissions is that
        the Bmatrix has to be structured like this. I.e., I can't just save uncertainties
        per grid cell, per timestep, because to aggregate to e.g., yearly, you would need 
        cross-correlations, and you'd need to go back to the original Bmatrix. 
        
        For easy combination and comparison, I structure both emis and B in the same way, 
        even though emissions are much easier to aggregate.
        """
        
        # Do we also postprocess the truth or not?
        labels = self.get_emission_labels() # prior, posterior, and optionally truth
        
        self.emis_agg = self.setup_container_dict_emis()
        for v in labels:
            
            xi               = self.get_state_from_label(v)
            emis_i           = xi*self.inversion.emis_inv_vec
            
            for xres,tres in self.aggregate_resolutions:
                for prior in self.get_unique_priornames():
                    self.emis_agg[v][xres][tres][prior] = self.aggregate_array_onedim(emis_i, xres, tres, prior)
        
        self.save_aggregated_emissions()
        
    def postprocess_Gmatrix(self):
        """
        Postprocess/aggregate Gmatrix. This one is fairly easy since there's only one Gmatrix,
        and Gmatrix has dimensions (nstate, nobs), so aggregation only needed along the 0-axis.
        
        Remember : xopt = xpri + G (y-H.x)
        So G gives the adjustment to xpri of 1 ppm model-obs mismatch (and xpri is scaled!)
        
        I choose to leave Gmatrix scaled by emissions instead of converting back to relative.
        Note that I take the absolute value, because I am interested in which observations are
        important for which parts of the grid regardless of whether they cancel out and end up
        being unimportant.
        """
        
        # We need to aggregate Gmatrix weighted by emissions, because it matters if an observation reduces
        # relative uncertainties in gridcells with high or low emissions 
        # I scale by absolute emissions, because if 
        Gmatrix_sc = np.abs(self.inversion.Gmatrix[:,:]*self.inversion.emis_inv_vec[:,np.newaxis])
        
        self.Gmatrix_agg = self.setup_container_dict_Gmatrix()
        for xres,tres in self.aggregate_resolutions:
            for prior in self.get_unique_priornames():
                self.Gmatrix_agg[xres][tres][prior] = self.aggregate_array_onedim(Gmatrix_sc, xres, tres, prior)
        
        self.save_aggregated_Gmatrix()
        
    def get_emission_labels(self):
        '''
        Basically just a conditional to check if we only postprocess (prior, posterior), or also the truth.
        Postprocessing the truth is only needed when we're in an OSSE and the truth was created from a perturbation of the state.
        '''
        
        if self.rc.inversion_is_osse and self.rc.osse_create_obs_method=='from_state':
            return ['prior', 'posterior', 'true']
        else:
            return ['prior', 'posterior']
        
    def get_state_from_label(self, label):
        
        if label=='prior':
            return self.inversion.xprior
        elif label=='posterior':
            return self.inversion.xopt
        elif label=='true':
            return self.inversion.xtrue
        
    def aggregate_array_onedim(self, array, xres, tres, prior):
        '''
        Aggregate an array for one xres, tres, inventory combination. 
        This function can be used for arrays that need to be aggregated along one
        dimension (e.g., emissions or Gmatrix, but not Bmatrix). 
        '''
        
        priornames = self.inversion.read_state()[1]
        priormask  = (priornames==prior)
        
        array_sel  = array[priormask]
        
        # Get weights for aggregation
        weights_time  = self.create_aggregate_weights_time(tres)
        weights_space = self.create_aggregate_weights_space(xres, self.inversion.get_domains_for_prior(prior))
        
        ntime_tot  = weights_time.shape[1]
        nspace_tot = weights_space.shape[1]
        
        # Reshape so we can separately aggregate time and space
        # Differentiate in reshape between 1-D array (e.g., emis = (nstate)) and 2-D (e.g., Gmatrix = (nstate,nobs))
        ndim = len(array_sel.shape)
        if ndim == 1:
            array_sel = array_sel.reshape(ntime_tot, nspace_tot)
        elif ndim == 2:
            array_sel = array_sel.reshape(ntime_tot, nspace_tot, -1)
            
        # Aggregate
        array_agg = self.aggregate_dimension_with_weights(array_sel, weights_time , ax_agg=0)
        array_agg = self.aggregate_dimension_with_weights(array_agg, weights_space, ax_agg=1)
        
        return array_agg
    
    def postprocess_Bmatrices(self):
        labels  = self.get_Bmatrix_labels() # prior, posterior, and optionally truth
        upriors = self.get_unique_priornames()
        
        self.B_agg = {'rel':self.setup_container_dict_Bmatrix(), 
                      'abs':self.setup_container_dict_Bmatrix()}
        
        for v in labels:
            Bmatrix = self.get_Bmatrix_from_label(v)
            for rel_or_abs in ['rel','abs']:
                # I want to save both relative and absolute uncertainties
                for xres,tres in self.aggregate_resolutions:
                    for prior1 in upriors:
                        for prior2 in upriors:
                            Bagg_i = self.aggregate_Bmatrix_cross(Bmatrix, tres, xres, prior1, prior2, rel_or_abs)
                            self.B_agg[rel_or_abs][v][xres][tres][prior1][prior2] = Bagg_i
            del(Bmatrix)
        
        self.save_aggregated_Bmatrix()
        del(self.B_agg)
        
    def get_Bmatrix_labels(self):
        '''
        Basically just a conditional to check if we only postprocess (prior, posterior), or also the truth.
        Postprocessing the true error matrix is only needed when we're in an OSSE AND the truth was created 
        from a perturbation of the state AND the perturbation of the state uses a different error matrix than the inversion.
        '''
        
        if self.rc.inversion_is_osse and self.rc.osse_create_obs_method=='from_state' and self.rc.osse_Btrue_is_Bpri==False:
            return ['prior', 'posterior', 'true']
        else:
            return ['prior', 'posterior']
        
    def get_Bmatrix_from_label(self, label):
        
        if label=='prior':
            return self.inversion.read_Bmatrix()
        elif label=='posterior':
            return self.inversion.read_Bopt()
        elif label=='true':
            return self.inversion.read_Btrue()
    
    def postprocess_observations(self):
        '''
        Relatively simple, write prior posterior and optionally true observations to 
        a file. We organize it a little bit, to make it easier to use.
        '''
        
        # For masking out spinup/down:
        start = self.rc.date_start + self.dt_spinup
        end   = self.rc.date_end   - self.dt_spindown
        
        self.obs_all = self.setup_container_dict_obs()
        err = np.sqrt(np.diag(self.inversion.Rmatrix))
        for label in ['prior','posterior','true']:
            y = self.get_yvector_from_label(label)
            sites, dates, _ = self.inversion.read_yvector()
            
            mask_spin = (dates>=start) & (dates<=end)
            for site in np.unique(sites):
                mask_site = (sites==site)
                mask = (mask_spin & mask_site)
                
                self.obs_all[label][site]['dates']   = dates[mask]
                self.obs_all[label][site]['co2']     = y[mask]
                self.obs_all[label][site]['co2_err'] = err[mask]
                self.obs_all[label][site]['co2_err_2d'] = self.inversion.Rmatrix[np.outer(mask,mask)]
                
        self.save_postprocessed_observations()
        
    def get_yvector_from_label(self, label):
        if label=='prior':
            return self.inversion.Hmatrix @ self.inversion.xprior
        elif label=='posterior':
            return self.inversion.Hmatrix @ self.inversion.xopt
        elif label=='true':
            return self.inversion.yvector
        
    def calculate_inversion_statistics(self):
        pass
    
    def aggregate_Bmatrix_cross(self, Bmatrix, tres, xres, inventory1, inventory2, rel_or_abs):
        '''
        Aggregate B matrix dimensions to a specific temporal resolution and spatial
        resolution (tres, xres). Since B is defined as a relative uncertainty, the
        aggregation requires weighting with emissions. For conceptual ease, I first
        convert the relative uncertainties to absolute uncertainties, so that aggregation
        is simply addition. 
        Optionally, I can convert back to relative by dividing by total (absolute) emissions 
        based on rel_or_abs argument.
        If inventory1==inventory2 it's simply the uncertainty of the inventory, otherwise
        it presents the cross-correlations.
        '''
        
        # Select the part of the Bmatrix that we're interested in first
        _, priornames, domainnames = self.inversion.read_state()
        
        mask1 = priornames==inventory1
        mask2 = priornames==inventory2
        
        Bsel = Bmatrix[mask1, :][:,mask2]
                
        # Get weights to aggregate time
        weights_time = self.create_aggregate_weights_time(tres)
        
        # Get weights to aggregate space, one per set of domains
        domains1       = self.inversion.get_domains_for_prior(inventory1)
        weights_space1 = self.create_aggregate_weights_space(xres, domains1)
        domains2       = self.inversion.get_domains_for_prior(inventory2)
        weights_space2 = self.create_aggregate_weights_space(xres, domains2)
        
        nspace_agg1, nspace_tot1 = weights_space1.shape
        nspace_agg2, nspace_tot2 = weights_space2.shape
        ntime_agg , ntime_tot  = weights_time.shape
        
        # Scale B by emissions, since aggregation doesn't work for relative uncertainties
        emis_sel1 = np.abs(self.inversion.emis_inv_vec[mask1]) # Weigh with absolute emissions
        emis_sel2 = np.abs(self.inversion.emis_inv_vec[mask2]) # Weigh with absolute emissions
        Bsel    *= np.outer(emis_sel1, emis_sel2)
        Bsel     = Bsel.reshape(ntime_tot, nspace_tot1, ntime_tot, nspace_tot2)
        emis_sel1 = emis_sel1.reshape(ntime_tot, nspace_tot1)
        emis_sel2 = emis_sel2.reshape(ntime_tot, nspace_tot2)
        
        weights_time                   = weights_time.astype(bool)
        weights_space1, weights_space2 = weights_space1.astype(bool), weights_space2.astype(bool)
        Bagg = np.zeros((ntime_agg, nspace_agg1, ntime_agg, nspace_agg2))
        
        # Aggregate
        Bagg = self.aggregate_dimension_with_weights(Bsel, weights_time , ax_agg=0)
        Bagg = self.aggregate_dimension_with_weights(Bagg, weights_time , ax_agg=2)
        Bagg = self.aggregate_dimension_with_weights(Bagg, weights_space1, ax_agg=1)
        Bagg = self.aggregate_dimension_with_weights(Bagg, weights_space2, ax_agg=3)
        
        if rel_or_abs=='rel':
            emis_agg1 = self.aggregate_dimension_with_weights(emis_sel1, weights_time , ax_agg=0)
            emis_agg1 = self.aggregate_dimension_with_weights(emis_agg1, weights_space1, ax_agg=1)
            emis_agg2 = self.aggregate_dimension_with_weights(emis_sel2, weights_time , ax_agg=0)
            emis_agg2 = self.aggregate_dimension_with_weights(emis_agg2, weights_space2, ax_agg=1)
            Bagg /= np.outer(emis_agg1.flatten(), emis_agg2.flatten()).reshape(ntime_agg, nspace_agg1, ntime_agg, nspace_agg2)
                            
        return Bagg
    
    def create_aggregate_weights_time(self, tres):
        '''
        Weights are of shape (new dimension, old dimension), and have ones where
        we want to add up the old dimension (since the error matrix is already emission-weighted).
        
        We implement spinup/down here. The way we do it is 
         a) by dropping the spinup/down from the aggregated, new dimension: i.e., the first axis of weights, 
             specific to tres; e.g., diurnal cycle not affected, but daily weights are.
         b) by setting weights to zero in the full, old dimension (i.e., second axis of weights, same for all tres).
        '''
        
        nstep_long  = len(self.inversion.get_timesteps_opt_long()) # between-day steps
        dt_short    = self.rc.opt_freq_diurnal
        nstep_short = int(24/dt_short)                             # diurnal cycle steps
        nstep_tot   = self.inversion.get_ntimestep_opt()           # nstep_long*nstep_short
        
        if tres=='per_timestep':
            # No time aggregation
            weights = np.eye(nstep_tot)
            weights = weights[self.mask_spinupdown_tot]
            
        elif tres=='diurnal':
            # Aggregate to one diurnal cycle
            weights = np.zeros((nstep_short, nstep_tot))
            for i in range(nstep_short):
                weights[i, i::nstep_short] = 1
            
        elif tres=='daily':
            # Aggregate to one value per day; assume daily optimization, otherwise it should have been caught earlier
            weights = np.zeros((nstep_long, nstep_tot))
            for i in range(nstep_long):
                weights[i, (i*nstep_short):((i+1)*nstep_short)] = 1
            weights = weights[self.mask_spinupdown_long]
                
        elif tres=='one_timestep':
            # Aggregate to one value per day; assume daily optimization, otherwise it should have been caught earlier
            weights = np.ones((1, nstep_tot))
            
        else:
            raise ValueError("Unknown temporal resolution: %s"%tres)
        
        # Don't include spinup/down in aggregation; same for all
        weights[:, ~self.mask_spinupdown_tot] = 0 
        
        return weights
                
    def create_aggregate_weights_space(self, xres, domains):
        nspace = self.get_nspace_tot(domains)
            
        if xres=='per_gridcell':
            # Leave error matrix unchanged
            return np.eye(nspace)
            
        elif xres=='per_domain':
            weights = np.zeros((len(domains), nspace))
            i = 0
            for idom,domain in enumerate(domains):
                nspace_i = self.rc.nx_inv[domain]*self.rc.ny_inv[domain]
                weights[idom][i:i+nspace_i] = 1.0
                i += nspace_i
            return weights
        
        elif xres=='all_domains':
            # Add up all elements to one value
            return np.ones((1,nspace))
        
        else:
            raise ValueError("Unknown spatial resolution: %s"%xres)
        
    def get_nspace_tot(self,domains):
        nspace = 0
        for domain in domains:
            nspace += self.rc.nx_inv[domain]*self.rc.ny_inv[domain]
        return nspace
        
    def get_obs_sites(self):
        sites, _, _ = self.inversion.read_yvector()
        return np.unique(sites)
    
    def calc_weights_star(self, weights):
        return np.matmul(weights.T, np.linalg.inv(np.matmul(weights, weights.T)))
    
    def save_aggregated_emissions(self):
        fname_out = self.inversion.get_filename_emis_agg()
        if os.path.isfile(fname_out):
            os.remove(fname_out)
            
        with open(fname_out, 'wb') as f:
            pickle.dump(self.emis_agg, f)
            
    def save_aggregated_Gmatrix(self):
        fname_out = self.inversion.get_filename_Gmatrix_agg()
        if os.path.isfile(fname_out):
            os.remove(fname_out)
            
        with open(fname_out, 'wb') as f:
            pickle.dump(self.Gmatrix_agg, f)
            
    def save_aggregated_Bmatrix(self):
        
        for rel_or_abs, B in self.B_agg.items():
        
            fname_out = self.inversion.get_filename_Bmatrices_agg(rel_or_abs)
            if os.path.isfile(fname_out):
                os.remove(fname_out)
                
            with open(fname_out, 'wb') as f:
                pickle.dump(B, f)
            
    def save_postprocessed_observations(self):
        
        fname_out = self.inversion.get_filename_y_postpr()
        if os.path.isfile(fname_out):
            os.remove(fname_out)
            
        with open(fname_out, 'wb') as f:
            pickle.dump(self.obs_all, f)
    
    def read_aggregated_emissions(self):
        """
        Write aggregated emissions, defined in a nested dictionary, to a pickled file.
        """
        
        fname_out = self.inversion.get_filename_emis_agg()
        with open(fname_out, 'rb') as f:
            return pickle.load(f)
            
    def read_aggregated_Bmatrix(self, rel_or_abs='rel'):
        
        fname_out = self.inversion.get_filename_Bmatrices_agg(rel_or_abs)
        with open(fname_out, 'rb') as f:
            return pickle.load(f)
            
    def read_aggregated_Gmatrix(self):
        
        fname_out = self.inversion.get_filename_Gmatrix_agg()
        with open(fname_out, 'rb') as f:
            return pickle.load(f)
            
    def read_postprocessed_observations(self):
        
        fname_out = self.inversion.get_filename_y_postpr()
        with open(fname_out, 'rb') as f:
            return pickle.load(f)
                
    def get_unique_priornames(self):
        return np.unique(self.inversion.read_state()[1])
             
    def aggregate_dimension_with_weights(self, ar, weights, ax_agg):
        '''
        Aggregate array "ar" along axis "ax_agg" using the prescribed weights.
        Weights need to have shape (ndim_new, ndim_old), and ax_agg of ar needs 
        to have size ndim_old; after aggregation, it will have size ndim_new.
        '''
        
        ax_w = 1 # Weights need to have shape (ndim_new, ndim_old)
        ar_agg = np.tensordot(ar, weights, axes=((ax_agg), (ax_w)))
        # tensordot moves aggregated dimension to last axis, but I want to retain original dimension order
        ar_agg = np.moveaxis(ar_agg, -1, ax_agg)
        
        return ar_agg
           
    def setup_container_dict_emis(self):
        '''
        Setup a nested, empty dictionary for emissions, so that it's easier to fill
        in the different resolutions and priors.
        '''
        
        container = {}
        for v in self.get_emission_labels():
            container[v] = {}           
            for xres,tres in self.aggregate_resolutions:
                if xres not in container[v].keys():
                    container[v][xres] = {}
                container[v][xres][tres] = {}
                for inventory in self.get_unique_priornames():
                    container[v][xres][tres][inventory] = None
                    
        return container
    
    def setup_container_dict_Gmatrix(self):
        
        container = {}  
        for xres,tres in self.aggregate_resolutions:
            if xres not in container.keys():
                container[xres] = {}
            container[xres][tres] = {}
            for inventory in self.get_unique_priornames():
                container[xres][tres][inventory] = None
                    
        return container
        
    
    def setup_container_dict_Bmatrix(self):
        
        container = {}
        for v in self.get_Bmatrix_labels():
            container[v] = {}           
            for xres,tres in self.aggregate_resolutions:
                if xres not in container[v].keys():
                    container[v][xres] = {}
                container[v][xres][tres] = {}
                for inventory1 in self.get_unique_priornames():
                    container[v][xres][tres][inventory1] = {}
                    for inventory2 in self.get_unique_priornames():
                        container[v][xres][tres][inventory1][inventory2] = None
                        
        return container
    
    def setup_container_dict_obs(self):
        
        container= {}
        for label in ['prior','posterior','true']:
            container[label] = {}
            for site in self.get_obs_sites():
                container[label][site] = {'dates' : None, 
                                              'co2' : None,
                                              'co2_err' : None}
                
        return container
                
    def setup_container_dict_dates(self):
        container = {}
        tress = np.unique(np.array(self.aggregate_resolutions)[:,1])
        for tres in tress:
            container[tres] = None
        return container

    def remove_unaggregated_matrices(self):
        '''
        A lot of storage space goes to the full Bpri, Bopt matrices. This is
        not really necessary since for plotting I only need aggregrated matrices,
        and if I do need the full matrices I can always recreate them.
        '''
        
        import os
        
        fnames  = [self.inversion.get_filename_Bmatrix()]
        fnames += [self.inversion.get_filename_Bmatrix_inv()]
        fnames += [self.inversion.get_filename_Bopt()]
        
        for fname in fnames:
            if os.path.isfile(fname):
                os.remove(fname)
        
        
