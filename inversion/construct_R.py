#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 12:39:34 2023

@author: nauss

Constructing the observational error correlation matrix.
"""

from inversion_base import InversionBase
import numpy as np
from datetime import datetime,timedelta

class ConstructorR(InversionBase):
    
    def __init__(self, inversion_name, modification_kwargs={}):
        super().__init__(inversion_name, modification_kwargs)
            
    def construct_R(self):
        self.make_Rmatrix()
        self.write_Rmatrix()
        
    def make_Rmatrix(self):
        nobs = self.get_nobs()
        self.Rmatrix_dt = self.make_dt_matrix_obs()
        self.Rmatrix    = np.zeros((nobs, nobs), dtype=float)
        for error_dict in self.rc.obs_errors:
            self.Rmatrix += self.make_Rmatrix_one(error_dict)
            
    def make_Rmatrix_one(self, error_dict):
        
        error       = error_dict["error_value"]
        error_type  = error_dict["error_type"] # sim_enh, hour_std, or absolute
        corlen      = error_dict['correlation_length'] # In hours
        
        Runscaled   = np.exp( - self.Rmatrix_dt / corlen )
        Runscaled[np.isnan(Runscaled)] = 0. # Different sites are uncorrelated, I put nans there
        Rdiag       = self.make_Rmatrix_diagonal(error, error_type)
        Rmatrix     = Rdiag[:,np.newaxis]*Rdiag[np.newaxis,:] * Runscaled
        
        return Rmatrix
        
    def make_Rmatrix_diagonal(self, error, error_type):
        '''
        Make the diagonal of the Rmatrix, i.e., the per-observation observational 
        errors. There's three options for this: 
            - A fixed ppm error
            - An error relative to the simulated enhancements
            - An error relative to the within-hour-std in the observations.
        Since I'm working in the OSSE space for now, I haven't implemented the third one yet.
        '''
        
        if   error_type == 'absolute':
            # Absolute error
            nobs = self.get_nobs()
            Rdiag    = error*np.ones(nobs)
            
        elif error_type == 'sim_enh':
            # Error relative to simulated enhancement
            Hmatrix  = self.read_Hmatrix()
            xprior   = self.read_state()[0]
            ymodel   = np.abs((Hmatrix @ xprior))  # Simulated enhancements
            Rdiag    = error*ymodel
            
        elif error_type == 'model_data_mismatch':
            # Error relative to difference between simulated enhancements and observations
            Hmatrix  = self.read_Hmatrix()
            xprior   = self.read_state()[0]
            ymodel   = np.abs((Hmatrix @ xprior))  # Simulated enhancements
            yobs     = self.read_yvector()[2]
            Rdiag    = error*np.abs(ymodel-yobs)
            
        else:
            raise KeyError("Obs error type not implemented: %s"%error_type)
            
        return Rdiag
    
    def write_Rmatrix(self):
        fname = self.get_filename_Rmatrix()
        np.save(fname, self.Rmatrix)
    
    def get_nobs(self):
        ysites = self.read_yvector()[0]
        return len(ysites)
    
    def make_dt_matrix_obs(self):
        '''
        Return 2-D matrix of shape (nobs , nobs) that gives the timestep in hours
        between each observation, with nans for dt's between different sites 
        ( so that obs errors of different sites are not correlated ).
        '''
        
        ysites, ydates, _ = self.read_yvector()
        nobs              = len(ysites)
        dt_matrix         = np.zeros((nobs,nobs))
        dt_matrix[:,:]    = np.nan
        for site in np.unique(ysites):
            idx = np.where(ysites==site)[0]
            dt_i = np.abs(ydates[idx][:,np.newaxis] - ydates[idx][np.newaxis,:])
            for i1,j1 in enumerate(idx):
                for i2,j2 in enumerate(idx):
                    dt_ii = dt_i[i1,i2]
                    dt_matrix[j1,j2] = dt_ii.days*24 + dt_ii.seconds/3600.
                    
        return dt_matrix
        
            

if __name__ == '__main__':
    inversion_name = 'baseAKLNWP_base'
    constrR = ConstructorR(inversion_name)
    constrR.construct_R()

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
        