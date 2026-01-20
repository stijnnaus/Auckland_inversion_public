#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions for the run_inversion scripts, so as not to clutter the run-script.
"""

from construct_x import ConstructorX
from construct_y_synthetic import ConstructorY_synthetic as ConstructorY
from construct_H import ConstructorH
from construct_B import ConstructorB
from construct_R import ConstructorR

import psutil
import time

from inversion_main import Inversion
from postprocessing import Postprocessing

import sys

def construct_input_matrices(inversion_name, rc_kwargs):
    """
    Construct input matrices for the inversion.
    """
    
    t0 = time.time()
    
    cx = ConstructorX(inversion_name, rc_kwargs)
    cx.remove_file_emis_on_invgrid()
    cx.construct_x()
    print(sys.getsizeof(cx))
    print(cx.get_filename_Bmatrix())
    del(cx)
    
    t1 = time.time()
    print('construct x took : %2.2fs'%(t1-t0), flush=True)
    
    cB = ConstructorB(inversion_name, rc_kwargs)
    cB.construct_B()
    print('Bmatrix: %2.2f mb'%(sys.getsizeof(cB.Bmatrix)/1e6), flush=True)
    del(cB)
    
    
    t2 = time.time()
    print('construct B took : %2.2fs'%(t2-t1), flush=True)
    
    # A placeholder so I can construct H and R, which I need to construct the "real" ysynthetic
    cy = ConstructorY(inversion_name, rc_kwargs)#, barebones=True)
    cy.construct_y()
    del(cy)
    
    t3 = time.time()
    print('construct y bb took : %2.2fs'%(t3-t2), flush=True)
    
    cH = ConstructorH(inversion_name, rc_kwargs)
    cH.construct_H()
    print('cH: %2.2f kb'%(sys.getsizeof(cH)/1e3), flush=True)
    del(cH)
    
    t4 = time.time()
    print('construct H took : %2.2fs'%(t4-t3), flush=True)
    
    cR = ConstructorR(inversion_name, rc_kwargs)
    cR.construct_R()
    del(cR)
    
    t5 = time.time()
    print('construct R took : %2.2fs'%(t5-t4), flush=True)
    
    cy = ConstructorY(inversion_name, rc_kwargs, barebones=False)
    cy.construct_y()
    del(cy)
    
    t6 = time.time()
    print('construct y full took : %2.2fs'%(t6-t5), flush=True)
    
def run_inversion(inversion_name, rc_kwargs):
    
    inv = Inversion(inversion_name, rc_kwargs)
    inv.run_inversion()
    
    memuse_1 = (psutil.virtual_memory()[3]/1e6)
    del(inv)
    memuse_2 = (psutil.virtual_memory()[3]/1e6)
    print("inv: Memory freed up %2.2f mb"%(memuse_1-memuse_2))

def run_postprocessing(inversion_name, rc_kwargs, dt_spinup, dt_spindown):
    postpr = Postprocessing(inversion_name, rc_kwargs, dt_spinup, dt_spindown)
    postpr.run_standard_postprocessing()
    
    memuse_1 = (psutil.virtual_memory()[3]/1e6)
    del(postpr)
    memuse_2 = (psutil.virtual_memory()[3]/1e6)
    print("postpr: Memory freed up %2.2f mb"%(memuse_1-memuse_2))
    
    