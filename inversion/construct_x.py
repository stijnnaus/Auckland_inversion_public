#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 11:31:08 2023

@author: nauss
"""

import inversion as inv
import functions_stijn as st
import numpy as np
import os
from datetime import datetime,timedelta
from inversion_base import InversionBase

class ConstructorX(InversionBase):
    
    '''
    Construct state. In the base case, this is just a 1-D vector of ones, because
    the Jacobian already contains both the footprints and the emission information.
    But in the future we might want to do something more interesting with it, so
    it's good to have it explicitly coded.
    '''
    
    def __init__(self, inversion_name, modification_kwargs={}):
        super().__init__(inversion_name, modification_kwargs)
        
    def construct_x(self):
        '''
        The main thing of importance here is the order of dimensions. I think it makes most sense
        to have prior on the outside, then time, and then the spatial dimensions. Having the 
        spatial dimensions subsequentially is helpful, because we want to split the B matrix
        in a spatial and temporal component and then later combine those. And both temporal
        and spatial might have different correlations for different priors, so priors on the
        outside.
        '''
        
        self.xprior        = np.array([], dtype=float)
        self.xpriornames   = np.array([], dtype=object)
        self.xpriordomains = np.array([], dtype=object)
        
        for prior in self.priors_all:
            domains_i = self.get_domains_for_prior(prior)
            nstep = self.get_ntimestep_opt()
            for itime in range(nstep):
                for domain in domains_i:
                    nx,ny = self.rc.nx_inv[domain],self.rc.ny_inv[domain]
                    self.xprior        = np.append(self.xprior       , [1.0]*nx*ny)
                    self.xpriornames   = np.append(self.xpriornames  , [prior]*nx*ny)
                    self.xpriordomains = np.append(self.xpriordomains, [domain]*nx*ny)
            
        self.write_state()
        
    def write_state(self):
        path = self.get_path_inversion_input()
        
        print("Saving state to ", '%s/xprior.npy'%path)
        np.save('%s/xprior.npy'%path, self.xprior)
        np.save('%s/xpriornames.npy'%path, self.xpriornames, allow_pickle=True)
        np.save('%s/xpriordomains.npy'%path, self.xpriordomains, allow_pickle=True)
        
        
if __name__ == '__main__':
    constr = ConstructorX('baseAKLNWP_base_2month')
    constr.construct_x()

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        