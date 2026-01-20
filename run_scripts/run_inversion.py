#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run one inversion with input arguments.
This includes setting up input matrices, running the inversion, and postprocessing.
"""

import sys
from datetime import timedelta,datetime

import run_inversion_helpers as he

# Read input arguments
# inversion_name = sys.argv[1]
# start          = datetime.strptime(sys.argv[2], '%Y%m%d')
# end            = datetime.strptime(sys.argv[3], '%Y%m%d')
# spinup_days    = int(sys.argv[4]) # in days
# spindown_days  = int(sys.argv[5]) # in days
# task           = sys.argv[6] # preprocess, inversion, postprocess or all

# Test set-up
inversion_name = 'baseAKLNWP_test'
start          = datetime(2022,1,1)
end            = datetime(2022,1,10)
spinup_days    = 1
spindown_days  = 1
task = 'all'


nstepNAME = 25
dt_spinup   = timedelta(days=spinup_days)
dt_spindown = timedelta(days=spindown_days)
start_full = start - dt_spinup
end_full   = end   + dt_spindown


# Startdate denotes the date of the first NAME simulation that is used, but since this
# simulation is run backwards, actual emissions are optimized nstepNAME backwards
# Therefore, to have a whole number of days in the simulation we add nstepNAME
# Note that since we use a spin-up period the effect of this treatment will always be small
start_full += timedelta(seconds=nstepNAME*3600)

start_str = start_full.strftime("%Y-%m-%d")
end_str   = end_full.strftime("%Y-%m-%d")

# Modification arguments to default settings defined by inversion_name - only time period
mod_args = {}
mod_args['date_start']      = ['direct', start_full]
mod_args['date_end']        = ['direct', end_full]

#%%

# Run inversion and postprocessing
import time

if task=='preprocess' or task=='all':

    print("Constructing matrices..", flush=True)
    t0 = time.time()

    he.construct_input_matrices(inversion_name, mod_args)

    t1 = time.time()
    print("Constructing matrices took %2.2fs"%(t1-t0), flush=True)


if task=='inversion' or task=='all':

    print("Running inversion %s-%s"%(start_str,end_str), flush=True)
    t0 = time.time()

    he.run_inversion(inversion_name, mod_args)

    t1 = time.time()
    print("Running inversion took %2.2fs"%(t1-t0), flush=True)


if task=='postprocess' or task=='all':

    print("Running postprocessing..", flush=True)
    t0 = time.time()
    
    he.run_postprocessing(inversion_name, mod_args, dt_spinup, dt_spindown)
    
    t1 = time.time()
    print("Postprocessing took %2.2fs"%(t1-t0), flush=True)



#%%

# from postprocessing import Postprocessing
# postpr = Postprocessing(inversion_name, mod_args, dt_spinup, dt_spindown)
# e = postpr.read_aggregated_emissions()

# from construct_y_synthetic import ConstructorY_synthetic as ConstructorY
# from construct_H import ConstructorH
# import numpy as np

# cy = ConstructorY(inversion_name, mod_args, barebones=False)
# cH = ConstructorH(inversion_name, mod_args)

# ysites, ydates, yvector = cy.read_yvector()
# Hmatrix = cH.read_Hmatrix()
# yprior = np.sum(Hmatrix, axis=1)

# import matplotlib.pyplot as plt

# usites = np.unique(ysites)

# fig,ax = plt.subplots(1,len(usites),figsize=(5*len(usites),4.5))

# for i,site in enumerate(usites):
#     mask = ysites==site
    
#     ax[i].set_title(site)
#     # ax[i].plot(ydates[mask], yprior[mask], '.', label='prior')
#     # ax[i].plot(ydates[mask], yvector[mask], '.', label='truth')
#     ax[i].scatter(yprior[mask], yvector[mask])
    
#     ax[i].set_xlim(-10,10)
#     ax[i].set_ylim(-10,10)

# fig.tight_layout()
