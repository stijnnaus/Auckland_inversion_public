#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 15:31:36 2023

The main inversion code doing the calculations.

@author: nauss
"""

import inversion as inv
import numpy as np
import os,time
from datetime import datetime,timedelta
from construct_x import ConstructorX
from construct_y import ConstructorY
from construct_H import ConstructorH
from construct_B import ConstructorB
from construct_R import ConstructorR

from inversion_base import InversionBase

def construct_all(inversion_name):
    
    cx = ConstructorX(inversion_name)
    cx.remove_file_emis_on_invgrid()
    cx.construct_x()
    cy = ConstructorY(inversion_name)
    cy.construct_y()
    cR = ConstructorR(inversion_name)
    cR.construct_R()
    cH = ConstructorH(inversion_name)
    cH.construct_H()
    cB = ConstructorB(inversion_name)
    cB.construct_dx_dt()
    
    
class Inversion(InversionBase):
    
    def __init__(self, inversion_name, modification_kwargs):
        super().__init__(inversion_name, modification_kwargs)
        
    def run_inversion(self):
        self.cleanup_before_inversion()
        t0 = time.time()
        self.read_inversion_input(which_bmatrix='inv')
        t1 = time.time()
        self.do_inversion_calculations()
        t2 = time.time()
        self.save_inversion_output()
        t3 = time.time()
        self.calculate_costs()
        t4 = time.time()
        
        print("Read inv input %2.2fs"%(t1-t0), flush=True)
        print("Do inv calcula %2.2fs"%(t2-t1), flush=True)
        print("Save inv outpu %2.2fs"%(t3-t2), flush=True)
        print("Calculate cost %2.2fs"%(t4-t3), flush=True)
        
    def read_inversion_input(self, which_bmatrix='inv'):
        '''
        Read inversion necessary for input. 
        The Bmatrix is by far the largest matrix in most cases, so I leave it
        optional whether we read B, Binv, or none of them. Note that for the
        inversion we only need Binv.
        '''
        
        self.xprior      = self.read_state()[0]
        self.ysites, self.ydates, self.yvector = self.read_yvector()
        self.Hmatrix     = self.read_Hmatrix()
        self.Rmatrix     = self.read_Rmatrix()
        
        if   which_bmatrix is None:
            pass
        elif which_bmatrix == 'inv':
            self.Bmatrix_inv = self.read_Bmatrix_inv()
        else:
            self.Bmatrix = self.read_Bmatrix()
        
        if self.rc.inversion_is_osse and self.rc.osse_create_obs_method=='from_state':
            self.xtrue = np.load(self.get_filename_xtrue())
        
    def read_inversion_output(self, read_bopt=True):
        path = self.get_path_inversion_output()
        
        self.xopt = np.load('%s/xopt.npy'%path)
        
        if read_bopt:
            self.Bopt = self.read_Bopt()
            
        self.Gmatrix = np.load('%s/Gmatrix.npy'%path)
        
    def do_inversion_calculations(self):
        
        t0 = time.time()
        self.calculate_Bopt()
        t1 = time.time()
        self.calculate_xopt()
        t2 = time.time()
        
        print("Calculate Bopt %2.2fs"%(t1-t0), flush=True)
        print("Calculate xopt %2.2fs"%(t2-t1), flush=True)
        
    def calculate_Bopt(self):
        
        self.Rmatrix_inv = self.invert_matrix(self.Rmatrix)
        self.HxR         = np.dot(self.Hmatrix.T, self.Rmatrix_inv) 
        self.Bopt        = self.invert_matrix( np.dot(self.HxR, self.Hmatrix) + self.Bmatrix_inv )
        
    def calculate_xopt(self):
        self.Gmatrix = np.dot(self.Bopt, self.HxR)
        self.xopt    = self.xprior + np.dot( self.Gmatrix, (self.yvector - np.dot(self.Hmatrix, self.xprior)))
        
    def save_inversion_output(self):
        path = self.get_path_inversion_output()
        np.save('%s/xopt.npy'%path, self.xopt)
        np.save('%s/Bopt.npy'%path, self.Bopt)
        np.save('%s/Gmatrix.npy'%path, self.Gmatrix)
        
    def calculate_costs(self):
        self.cost_prior = self.calculate_cost_obs( np.dot(self.Hmatrix, self.xprior))
        self.cost_poste = self.calculate_cost_obs( np.dot(self.Hmatrix, self.xopt))
        self.cost_bg    = self.calculate_cost_bg(self.xopt)
        
    def calculate_cost_bg(self, xnew):
        return np.sum(np.dot( (xnew-self.xprior)**2, self.Bmatrix_inv))
    
    def calculate_cost_obs(self, ynew):
        return np.sum(np.dot( (self.yvector-ynew)**2, self.Rmatrix_inv))
    
    def get_scaling_domain(self, xfull, domain):
        # Convert 1-D vector back to a multi-dimensional array for specific domain
        ntime = self.get_ntimestep_opt()
        nx, ny = self.rc.nx_inv[domain], self.rc.ny_inv[domain]
        priors = self.rc.priors_to_opt[domain]
        nprior = len(priors)
        
        priornames = self.read_state()[1]
        priordomains = self.read_state()[2]
        xsel = np.zeros((nprior, ntime, nx, ny))
        for ipri0,pri in enumerate(priors):
            xsel_i = xfull[((priornames==pri) & (priordomains==domain))]
            idx = 0
            for it in range(ntime):
                for ix in range(nx):
                    for iy in range(ny):
                        xsel[ipri0,it,ix,iy] = xsel_i[idx]
                        idx += 1
        return xsel
    
    def invert_matrix(self, matrix):
        """
        Apparently for inverting matrix, as a rule of thumb, it's faster to not
        use np.linalg.inv(matrix), but better to rewrite and let np.linalg.solve
        solve a A.x = 1 matrix for x. I did some testing and result is identical.
        
        Additional testing showed that linalg.inv worked better for me, so I reverted
        back to that.
        """
        
        #matrix_inv = np.linalg.solve(matrix, np.eye(len(matrix)))
        matrix_inv = np.linalg.inv(matrix)
        return matrix_inv
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
