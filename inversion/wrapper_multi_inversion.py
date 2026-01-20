#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here, we run/submit multiple parallel inversions to cover a longer time period.
Important settings are start, end, spin-up, spin-down and length of each inversion.
"""

from datetime import datetime,timedelta
import numpy as np


start       = datetime(2022,3,1)
end         = datetime(2023,3,1)

dt_inversion = timedelta(weeks=4) # does not include spin-up/down
dt_spinup    = timedelta(weeks=2)
dt_spindown  = timedelta(weeks=2)

start_inversion = start
while start_inversion<end:
    
    end_inversion = start_inversion + dt_inversion
    
    run_inversion(start_inversion, end_inversion, dt_spinup, dt_spindown)
    
    start_inversion += dt_inversion
    
    
    
    

