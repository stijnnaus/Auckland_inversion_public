#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate CO2 enhancements from NAME footprints and preprocessed emission inventories
at the highest grid resolution, and then convert to the (generally much coarser)
inversion grid. I can still do my inversion at a coarser resolution, but obviously
I can't go back to a finer resolution. Therefore, the "coarse" inversion grid is
the highest possible resolution at which I would want to do my inversion.

There's some specific ways in which I deal with my NAME nested rc.domains. 
  1. I stick quite close to the three nested NAME rc.domains I have, but I make the inner  
   domain a bit smaller so that it aligns exactly with MahuikaAuckland. This ensures  
   that I can just have one prior per domain. 
  2. In each outer nested domain I set the inner rc.domains to 0 before regridding to
   to the inversion grid, so you can never choose later to e.g., only use the outer
   domain (but you can still use only the inner one). This is because the inversion
   grid can be coarse, and so (setting inner domain to zero, then regridding) is not
   the same as (regridding, then setting the inner domain to zero).

"""

from datetime import datetime,timedelta
import calendar

import sys

import enhancement_calculator as enhc

#%%

year, month = int(sys.argv[1]), int(sys.argv[2])
nday = calendar.monthrange(year,month)[1]
dates = [datetime(year,month,i) for i in range(1,nday+1)]

#%%

domains = 'Mah0p3_in','Mah0p3_out', 'In1p5', 'Out7p0'
name_runs = 'baseAKLNWP', 'baseNZCSM'
name_runs = 'baseNZCSM',

# inventories = 'MahuikaAuckland', 'UrbanVPRM', 'EDGARv8', 'ODIAC', 'BiomeBGC', 'UrbanVPRM_new_x', 'UrbanVPRM_new_t'
inventories = 'EDGARv8','MahuikaAuckland','UrbanVPRM','BiomeBGC',

inner = ['UrbanVPRM','MahuikaAuckland','ODIAC']
outer = ['BiomeBGC','EDGARv8']
inventories_per_domain = {'Mah0p3_in':inner,
                          'Mah0p3_out':inner,
                          'In1p5':outer,
                          'Out7p0':outer}

#dates = []
#dates += [datetime(2021,12,10) + timedelta(days=i) for i in range(22)]
#dates += [datetime(2023,1,14) + timedelta(days=i) for i in range(20)]
for name_run in name_runs:
    for domain in domains:
        
        # print("Processing %i/%i; run %s; domain %s"%(year, month, name_run, domain), flush=True)
        
        inventories = inventories_per_domain[domain]
        
        Prepr = enhc.CalculatorEnhancementsForInversion('%s_base'%name_run)
        Prepr.process_enhancements(dates, domain, inventories, remove_diurnal=False)     
        





