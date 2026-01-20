#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Here I set up the class that everything that does part of the inversion 
(constructing matrices, the actual inversion) can inherit from. It includes
shared functions of the different parts of the inversion, that often require
all the details of the configuration, so if I would define them as separate
functions outside of a class, I would constantly have to re-read the configuration.

It also enables me to define e.g., filenames in one place. Generally, it's better
for clean code I think.''

Rule of thumb is that everything I use in multiple places ends up here, and everything
that is more specific than that doesn't (so that I don't end up with one big class).

In hindsight I'm not sure if this was the best way to do this but here we are..
"""

import inversion as inv
import functions_stijn as st
import read_priors as rp
import numpy as np
import os,time
from datetime import datetime,timedelta
import pickle
from base_paths import path_inv_base, path_inv

class InversionParameters(object):
    '''
    This is the class in which I store all the parameters from the yaml file.
    I do it this way because:
        a) I think it's cleaner than constantly referring to a dictionary.
        b) Doing it this way makes the attributes traceable to the yaml setup file, 
    '''
    
    def __init__(self, inversion_name, modification_kwargs={}):
        config_dict = inv.load_inversion_setup(inversion_name)
        for k,v in config_dict.items():
            setattr(self, k, v)
            
        # Optionally modify some of the YAML entries
        self.modify_rc_dict(modification_kwargs)
        
        
    def modify_rc_dict(self, kwargs):
        """
        As default we read the rc arguments from a yaml prescribed by inversion_name.
        Sometimes, it's nice to have a shortcut if I want to run test OSSE's modifying
        only one or two entries in the configuration, to do that inside the python
        script I'm running, without creating new yaml files. This gives me that option.
        Kwargs is a dictionary simular to config_dict, except it's items are a list,
        where the first entry describes what to modify:
            direct -> modify the entire rc value
            dict   -> modify only a subpart of the rc value, which has to be a nested
                        dictionary in that case. For example: only modify the temporal
                        length scale of the prior error entry, but conserve the rest
                        of the prior entry.
        In short, it's a quick way to do a targeted modification of the rc config in-script.
        """
        
        for k,v in kwargs.items():
            if v[0]=='direct':
                # Directly set an rc value
                setattr(self, k, v[1])
            elif v[0]=='dict':
                # Set only an rc dictionary entry, not the whole rc value
                # It's a bit awkward, but that's because I wanted to generalize
                # for an infinitely nested dictionary, since some of them get pretty deep
                curr_dict = getattr(self, k) # Select the rc entry
                keys = v[1:-1]               # Drop 'dict' and the last argument, which is the to-be-assigned value
                
                nest_dict = curr_dict
                for key in keys[:-1]: # Loop up to last dictionary key
                    nest_dict = nest_dict[key]
                    
                # Assign value
                value = v[-1]
                nest_dict[keys[-1]] = value
                setattr(self, k, curr_dict)
                
            else:
                raise ValueError("First list item should be dict or direct, instead it's:"%v[0])
                

class InversionBase(object):
    
    '''
    This is the base class for the inversion. All the paths and filenames are
    prescribed here, relative to the base paths in base_paths.py. 
    It includes any function that I use in more than one subclass
    (e.g., the constructors), and that requires config variables. This makes it
    a bit cluttered, but I didn't really know how else to do it.
    '''
    
    def __init__(self, inversion_name, modification_kwargs={}):
        self.inversion_name = inversion_name
        self.modification_kwargs = modification_kwargs
        
        self.unpack_YAML()
        self.setup_inversion_grid()
        
    def unpack_YAML(self):
        self.rc = InversionParameters(self.inversion_name, self.modification_kwargs)
        self.priors_all = np.unique([prior for _,priors in self.rc.priors_to_opt.items() for prior in priors])
        
    def get_path_inversion(self):
        start = self.rc.date_start.strftime('%Y%m%d')
        end   = self.rc.date_end.strftime('%Y%m%d')
        timeperiod = '%s_%s'%(start,end)
        return '%s/%s/%s/'%(path_inv, self.rc.inversion_label, timeperiod)
        
    def get_path_inversion_input(self):
        path = '%s/input/'%(self.get_path_inversion())
        self.ensure_path_exists(path)
        return path
    
    def get_path_inversion_output(self):
        path = '%s/output/'%(self.get_path_inversion())
        self.ensure_path_exists(path)
        return path
    
    def get_path_inversion_postprocess(self):
        path = '%s/aggregated/'%(self.get_path_inversion())
        self.ensure_path_exists(path)
        return path
            
    def get_path_enhancements_basegrid(self):
        # Path for the intermediate enhancements (nx_base, ny_base)
        path = '%s/enhancements/basegrid/%s/'%(path_inv_base, self.rc.basegrid_label)
        self.ensure_path_exists(path)
        return path
    
    def write_enhancements_basegrid(self, date, domain, emis_inv, enh_i):
        for isite, site in enumerate(self.rc.sites):
            fname = self.get_filename_intermediate_enhancements(date, domain, emis_inv, site)
            np.save(fname, enh_i[isite])
            
    def get_filename_enhancements_basegrid(self, date, domain, emis_inv, site):
        path = self.get_path_enhancements_basegrid()
        emis_cat = 'total' # hard-coded for now
        path = '%s/%s/%s/%s/'%(path, domain, emis_inv, emis_cat)
        self.ensure_path_exists(path)
        date_str = date.strftime("%Y%m%d")
        return "%s/enh_%s_%s.npy"%(path, site, date_str)
        
    def read_enhancements_basegrid(self, date, domain, emis_inv, site):
        fname = self.get_filename_enhancements_basegrid(date, domain, emis_inv, site)
        return np.load(fname)
    
    def get_path_enhancements_integrated(self, domain, inventory):
        path_enh    = self.get_path_enhancements_basegrid()
        path_integr = '%s/integrated/%s/%s/'%(path_enh, domain, inventory)
        self.ensure_path_exists(path_integr)
        return path_integr
        
    def get_filename_enhancements_integrated(self, domain, inventory, site, date):
        # Monthly files
        path = self.get_path_enhancements_integrated(domain, inventory)
        date_str = date.strftime('%Y%m')
        return '%s/%s_%s.npy'%(path,site,date_str)
    
    def read_yvector(self):
        path = self.get_path_inversion_input()
        yvector = np.load('%s/yvector.npy'%path)
        ydates = np.load('%s/ydates.npy'%path, allow_pickle=True)
        ysites = np.load('%s/ysites.npy'%path, allow_pickle=True)
        return ysites,ydates,yvector
        
    def get_filename_Hmatrix(self):
        path = self.get_path_inversion_input()
        return '%s/Hmatrix.npy'%(path)
    
    def read_Hmatrix(self):
        fname = self.get_filename_Hmatrix()
        return np.load(fname)
    
    def read_Rmatrix(self):
        return np.load(self.get_filename_Rmatrix())
    
    def read_Bmatrix_inv(self):
        return np.load(self.get_filename_Bmatrix_inv())
    
    def read_Bmatrix(self):
        return np.load(self.get_filename_Bmatrix())
    
    def read_Bopt(self):
        return np.load(self.get_filename_Bopt())
    
    def read_Btrue(self):
        return np.load(self.get_filename_Btrue())
    
    def get_filename_Bmatrix(self):
        return '%s/Bmatrix.npy'%self.get_path_inversion_input()
    
    def get_filename_Bmatrix_inv(self):
        return '%s/Bmatrix_inv.npy'%self.get_path_inversion_input()
    
    def get_filename_Bopt(self):
        return '%s/Bopt.npy'%self.get_path_inversion_output()
    
    def get_filename_Btrue(self):
        return '%s/Btrue.npy'%self.get_path_inversion_input()
        
    def get_filename_Rmatrix(self):
        path = self.get_path_inversion_input()
        return '%s/Rmatrix.npy'%(path)
    
    def get_filename_xtrue(self):
        path = self.get_path_inversion_input()
        return '%s/xtrue.npy'%(path)
    
    def get_filename_dx(self, label=''):
        path = self.get_path_inversion_input()
        return '%s/dx_2d%s.npy'%(path,label)
    
    def get_filename_dt(self, label):
        path = self.get_path_inversion_input()
        return '%s/dt_%s_2d.npy'%(path, label)
    
    def get_filename_true_enhancements(self, truth_label):
        invi = InversionBase(truth_label)
        return '%s/true_obs.nc4'%invi.get_path_inversion_input()
    
    def get_filename_emis_agg(self):
        return '%s/emis_agg.pkl'%self.get_path_inversion_postprocess()
        
    def get_filename_Bmatrices_agg(self, rel_or_abs='rel'):
        return '%s/Bmatrices_agg_%s.pkl'%(self.get_path_inversion_postprocess(), rel_or_abs)
    
    def get_filename_Gmatrix_agg(self):
        return '%s/Gmatrix_agg.pkl'%self.get_path_inversion_postprocess()
        
    def get_filename_y_postpr(self):
        return '%s/obs_all.pkl'%self.get_path_inversion_postprocess()      
    
    def get_ntimestep_opt(self):
        nstep_long = inv.determine_num_timesteps_long(self.rc.date_start, self.rc.date_end, self.rc.opt_freq_longterm, self.rc.nstepNAME)
        nstep_diurn = 24/self.rc.opt_freq_diurnal
        return int(nstep_long*nstep_diurn)
    
    def get_timesteps_opt_long(self, pos='start'):
        return inv.get_opt_timesteps(self.rc.date_start, self.rc.date_end, self.rc.opt_freq_longterm, self.rc.nstepNAME, pos=pos)
    
    def get_timesteps_opt_full(self):
        '''
        Get the timesteps that we optimize for, taking into account both the long and the in-day timesteps.
        '''
        
        tsteps_long = self.get_timesteps_opt_long()
        tsteps_diurnal = np.arange(0,24,self.rc.opt_freq_diurnal)
        tsteps_full = []
        for t1 in tsteps_long:
            for hour in tsteps_diurnal:
                tsteps_full.append(datetime(t1.year, t1.month, t1.day, hour))
                
        return np.array(tsteps_full)
        
    def get_nspace_opt(self):
        nspace = 0
        for domain in self.rc.domains:
            nxi, nyi = self.rc.nx_inv[domain], self.rc.ny_inv[domain]
            for prior in self.rc.priors_to_opt[domain]:
                nspace += nxi*nyi
        return nspace
        
    def get_nstate_opt(self):
        ntime = self.get_ntimestep_opt()
        nspace = self.get_nspace_opt()
        return ntime*nspace
    
    def get_domainname_NAME(self, domain):
        if   domain=='Mah0p3':
            return 'Mah0p3'
        elif domain=='In1p5':
            return 'In'
        elif domain=='Out7p0':
            return 'Out'
        else:
            raise KeyError("Unknown domain %s"%domain)
            
    def get_lonlat_invgrid(self, domain, bounds=False):
        
        nxi, nyi = self.rc.nx_inv[domain], self.rc.ny_inv[domain]
        return self.get_lonlat_domain(domain, nxi, nyi, bounds)
        
    def get_lonlat_basegrid(self, domain, bounds=False):
        
        nxi, nyi = self.rc.nx_base[domain], self.rc.ny_base[domain]
        return self.get_lonlat_domain(domain, nxi, nyi, bounds)
        
    def get_lonlat_domain(self, domain, nx, ny, bounds=False):
        
        lonb_full, latb_full = self.get_domain_bounds(domain)
        lonb = np.linspace(lonb_full[0], lonb_full[1], nx+1)
        latb = np.linspace(latb_full[0], latb_full[1], ny+1)
        
        if bounds:
            return lonb, latb
        else:
            lonc = 0.5*(lonb[:-1]+lonb[1:])
            latc = 0.5*(latb[:-1]+latb[1:])
            return lonc, latc
        
    def get_domain_bounds(self, domain):
        if self.rc.outer_bounds[domain] is None:
            lonb,latb = inv.getLonLatNAME(domain, bounds=True)
            lonb,latb = [lonb.min(),lonb.max()], [latb.min(),latb.max()]
        else:
            lonb = np.array(self.rc.outer_bounds[domain]['lon'])
            latb = np.array(self.rc.outer_bounds[domain]['lat'])
        return lonb, latb
            
    def read_state(self):
        path = self.get_path_inversion_input()
        xprior = np.load('%s/xprior.npy'%path)
        xpriornames = np.load('%s/xpriornames.npy'%path, allow_pickle=True)
        xpriordomains = np.load('%s/xpriordomains.npy'%path, allow_pickle=True)
        return xprior, xpriornames, xpriordomains
    
    def read_xtrue(self):
        fname = self.get_filename_xtrue()
        return np.load(fname)
    
    def read_dx(self, label=''):
        fname = self.get_filename_dx(label)
        return np.load(fname, allow_pickle=True)
    
    def read_dt(self, label):
        fname = self.get_filename_dt(label)
        return np.load(fname, allow_pickle=True)
    
    def get_prior_idx_in_state(self, prior):
        xpriornames = self.read_state()[1]
        idx = np.arange(len(xpriornames))
        return idx[xpriornames==prior]
                
    def get_prior_idx_in_Bspatial(self, prior):
        Bs_priors = self.read_dx('_priors')
        idx = np.arange(len(Bs_priors))
        return idx[Bs_priors==prior]
    
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
    
    def ensure_path_exists(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        
    
    def load_basegrid_enhancements(self, domain, inventory, site, date):
        # We load enhancements as gridded on the inversion grid. To get the integrated
        # enhancement we just have to sum over spatial grid and backwards timestep
        fname = self.get_filename_enhancements_basegrid(date, domain, inventory, site)
        enh = np.load(fname)
        return enh
            
    def setup_inversion_grid(self):
        
        self.lonb_inv, self.latb_inv, self.area_per_gridcell_inv = {}, {}, {}
        for domain in self.rc.priors_to_opt.keys():
            # Inversion grid
            nx,ny    = self.rc.nx_inv[domain], self.rc.ny_inv[domain]
            outerb   = self.get_outerbounds_inversion_domain(domain)
            self.lonb_inv[domain] = np.linspace(outerb['lon'][0], outerb['lon'][1], nx+1)
            self.latb_inv[domain] = np.linspace(outerb['lat'][0], outerb['lat'][1], ny+1)
            self.area_per_gridcell_inv[domain] = st.calc_area_per_gridcell(self.latb_inv[domain], self.lonb_inv[domain], bounds=True)
    
    def get_emissions_on_inversion_grid(self, overwrite=False):
        '''
        I have preprocessed emissions on the NAME grid, now I convert those to
        emission totals per inversion grid cell. Helpful for (error) aggregation
        and calculating emission totals.
        Parse both to a domain:inventory dictionary and to a vector of length state.
        Note: These are in principle prior emissions, if the state vector is filled
            with ones.
            
        '''
        
        if self.check_if_emis_on_invgrid_exist_onfile() and not overwrite:
            self.emis_inv_dict = self.read_emis_invgrid_from_file()
        else:
            self.emis_inv_dict = self.read_preprocessed_emis_and_regrid(self.rc.priors_to_opt)
            self.save_emis_inv_dict_to_file()
            
        self.scale_prior_emissions()
        self.emis_inv_vec  = self.parse_emis_dict_to_statevector(self.emis_inv_dict)
        
    def scale_prior_emissions(self):
        # Sometimes I want to scale certain priors by some constant scalar, here I implement that
        for domain, emii in self.emis_inv_dict.items():
            for prior, emii_i in emii.items():
                self.emis_inv_dict[domain][prior] *= self.get_scaling_factor_prior(prior)
        
    def get_scaling_factor_prior(self, prior):
        '''
        Get the optional scaling factor to be applied to a prior. This is applied
        both in construction of the Jacobian and in reading prior emissions.
        '''
        
        if hasattr(self.rc, 'scale_priors'):
            if prior in self.rc.scale_priors.keys():
                return self.rc.scale_priors[prior]
            else:
                return 1.0
        else:
            return 1.0
        
    def check_if_emis_on_invgrid_exist_onfile(self):
        return os.path.isfile(self.get_filename_emis_invgrid())
        
    def read_emis_invgrid_from_file(self):
        fname = self.get_filename_emis_invgrid()
        with open(fname, 'rb') as f:
            return pickle.load(f)
        
    def save_emis_inv_dict_to_file(self):
        fname = self.get_filename_emis_invgrid()
        if os.path.isfile(fname):
            os.remove(fname)
            
        with open(fname, 'wb') as f:
            pickle.dump(self.emis_inv_dict, f)
        
    def get_filename_emis_invgrid(self):
        return '%s/emis_invgrid.pkl'%self.get_path_inversion_input()
    
    def read_preprocessed_emis_and_regrid(self, priors_per_domain_dict, cats='all', varb='NEE'):
        '''
        Reading preprocessed emis can eat up a lot of memory (>20 Gb, even if only temporarily), 
        so I read each domain,inventory separately, regrid, and then move onto the next 
        so that I never have more than one domain,inventory combination in memory.
        
        Note that cats is applicable to Mahuika (industrial, traffic etc) and varb to
        biosphere priors (NEE, respiration, GPP).
        '''
        
        
        print("Reading preprocessed emissions...")
        udays         = self.get_unique_days_in_inversion()
        emis_inv_dict = {}
        for domain, inventories in priors_per_domain_dict.items():
            
            emis_inv_dict[domain] = {}
            domain_NAME = self.rc.domains_NAME[domain] # Which NAME domain does the inversion domain correspond to?
            
            for inventory in inventories:                
                emis_prepr = rp.read_preprocessed_1inventory(inventory, udays, domain_NAME, cats=cats, varb=varb)          # [kg/m2/s]
                emis_inv_dict[domain][inventory] = self.emis_aggregate_to_invgrid(emis_prepr, domain, inventory) # [kg/timestep/gridcell]
                del(emis_prepr)
                
        return emis_inv_dict
    
    def parse_emis_dict_to_statevector(self, emis_dict):
        '''
        Convert the per-domain, per-inventory dictionary to something of length
        state vector. The dictionary is already on inversion resolution, so
        it's just parsing it in the right order.
        '''
        
        self.xprior = self.read_state()[0]
        emis_vec = np.zeros_like(self.xprior, dtype=float)
        idx = 0
        for prior in self.priors_all:
            domains_i = self.get_domains_for_prior(prior)
            nstep = self.get_ntimestep_opt()
            for itime in range(nstep):
                for domain in domains_i:
                    nx,ny = self.rc.nx_inv[domain],self.rc.ny_inv[domain]
                    for ix in range(nx):
                        for iy in range(ny):
                            emis_vec[idx] = emis_dict[domain][prior][itime,iy,ix]
                            idx += 1
        return emis_vec
        
    def emis_aggregate_to_invgrid(self, emis_prepr, domain, inventory):
        '''
        Aggregate preprocessed emissions to inversion grid, also temporal aggregation
        frequency. We calculate as kg total per grid cell / time step.        
        '''
        
        mask_inner = self.make_mask_inner(domain)
        
        # NAME grid
        lonb_prepr, latb_prepr = inv.getLonLatNAME(self.rc.domains_NAME[domain], bounds=True)
        
        regr = st.make_xesmf_regridder(lonb_prepr, latb_prepr, 
                                       self.lonb_inv[domain], self.latb_inv[domain], 
                                       method='conservative', bounds=True)
        
        # Spatial aggregation
        emis_prepr[:,:,mask_inner] = 0.0 # Set inner bounds (= nested domain) to zero
        emis_regr = regr(emis_prepr) # kg/m2/s
        
        # Temporal aggregation
        emis_regr = self.emis_aggregate_temporal(emis_regr, inventory) # [kg/m2/timestep], shape (nt, nlat_inv, nlon_inv)
        
        # kg/m2/timestep to kg/gridcell/timestep
        emis_regr = emis_regr*self.area_per_gridcell_inv[domain]
        
        return emis_regr
                
    def make_mask_inner(self, domain):
        # Create mask to set to zero nested domains
        innerb = self.rc.inner_bounds[domain]
        lonc_prepr, latc_prepr = inv.getLonLatNAME(self.rc.domains_NAME[domain], bounds=False)
        if innerb is None:
            return np.zeros((len(latc_prepr), len(lonc_prepr)), dtype=bool)
        else:
            mask_lon = (lonc_prepr>innerb['lon'][0]) & (lonc_prepr<innerb['lon'][1])
            mask_lat = (latc_prepr>innerb['lat'][0]) & (latc_prepr<innerb['lat'][1])
            
            return np.outer(mask_lat, mask_lon)
                
    def get_outerbounds_inversion_domain(self, domain):
        
        outerb = self.rc.outer_bounds[domain]
        if outerb is None:
            # If set to None, default behaviour is use NAME domain bounds
            lonb,latb = inv.getLonLatNAME(self.rc.domains_NAME[domain], bounds=True)
            return {'lon':[lonb[0],lonb[-1]], 'lat':[latb[0],latb[1]]}
        else:
            return outerb
                
    def emis_aggregate_temporal(self, emis, inventory):
        '''
        Emissions are now of shape (nday, nhour, nlat_inv, nlon_inv). We want
        the first dimension to be the timestep in the inversion. This is equivalent
        to how we aggregate in e.g., ConstructorH.aggregate_time_to_opt_freq
        '''
        
        days_emis = self.get_unique_days_in_inversion()
        tsteps_emis = np.array([datetime(d.year, d.month, d.day, hour) for d in days_emis for hour in range(24)])
        hours_emis = np.array([d.hour for d in tsteps_emis])
        
        emis = emis*3600 # from per sec to per timestep
        nday,_,nlat,nlon = emis.shape
        emis = emis.reshape(-1, nlat,nlon) # Combine nday, nhour
        
        timesteps_long = self.get_timesteps_opt_long(pos='start')
        nt_opt = self.get_ntimestep_opt()
        dt_diurnal = self.rc.opt_freq_diurnal # In hours
        
        emis_tagg = np.zeros((nt_opt, nlat, nlon))
        istep = 0
        for tstep in timesteps_long: # Between days
            dt_long = inv.get_timestep_from_freq(tstep, self.rc.opt_freq_longterm)
            mask_long = (tsteps_emis>=tstep) & (tsteps_emis<(tstep+dt_long))
            
            for hour in range(0,24, dt_diurnal): # Within days
                mask_diurnal = (hours_emis>=hour) & (hours_emis<(hour+dt_diurnal))
                mask = (mask_long & mask_diurnal)
                
                emis_tagg[istep,:,:] += emis[mask].sum(axis=0)
                istep += 1
                
        return emis_tagg
        
    def get_unique_days_in_inversion(self):
        dstart = self.rc.date_start-timedelta(seconds=3600*self.rc.nstepNAME)
        dend   = self.rc.date_end
        day_start = datetime(dstart.year, dstart.month, dstart.day)
        day_end   = datetime(dend.year,   dend.month  , dend.day)
        days_to_read = []
        day = day_start
        while day<=day_end:
            days_to_read.append(day)
            day += timedelta(days=1)
        return np.array(days_to_read)
    
    def cleanup_before_inversion(self):
        '''
        Bit of a placeholder, for now remove emissions on invgrid at the start
        of each inversion so that we don't have old stuff sticking around.
        '''
        
        self.remove_file_emis_on_invgrid()
        
    def remove_file_emis_on_invgrid(self):
        fname = self.get_filename_emis_invgrid()
        if os.path.isfile(fname):
            os.remove(fname)
    
        
        
        
        
        
        
        



        
        
        
        
        
        
        
