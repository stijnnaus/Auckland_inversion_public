#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here, we run/submit multiple parallel inversions to cover a longer time period.
Import settings are start, end, spin-up, spin-down and length of each inversion.
"""

from datetime import datetime,timedelta
import subprocess
import shutil

def submit_inversion(inversion_name, start_inversion, end_inversion, dt_spinup, dt_spindown, task):
    
    start_str = start_inversion.strftime("%Y%m%d")
    end_str   = end_inversion.strftime("%Y%m%d")
    
    fname_template = "submit_inversion_template.j"
    fname_script   = "submit_jobs/submit_inversion_%s_%s.j"%(inversion_name, start_str)
    
    shutil.copyfile(fname_template, fname_script)
    subprocess.run(["sed", "-i", "s/inversion_name/%s/g"%inversion_name,   fname_script])
    subprocess.run(["sed", "-i",      "s/startdate/%s/g"%start_str,        fname_script])
    subprocess.run(["sed", "-i",        "s/enddate/%s/g"%end_str,          fname_script])
    subprocess.run(["sed", "-i",         "s/spinup/%s/g"%dt_spinup.days,   fname_script])
    subprocess.run(["sed", "-i",       "s/spindown/%s/g"%dt_spindown.days, fname_script])
    subprocess.run(["sed", "-i",           "s/rtask/%s/g"%task,             fname_script])
    
    subprocess.run(["sbatch", fname_script])


#inversion_names = 'baseAKLNWP_mahk', 'baseAKLNWP_mahk_new_x', 'baseAKLNWP_mahk_new_t'
inversion_names  = ['baseAKLNWP_odiac','baseNZCSM_base','baseAKLNWP_100m']
inversion_names += ['baseAKLNWP_double_pri_err', 'baseAKLNWP_only_AUT']
inversion_names = ['baseAKLNWP_afternoon']

#inversion_names = ['baseAKLNWP_vprm_std',] #"baseAKLNWP_vprm_afternoon"]

# start       = datetime(2022,1,1)
start       = datetime(2022,7,16)
end         = datetime(2022,12,30)
# end         = datetime(2022,12,30)

task = 'all' # preprocess, inversion, postprocess, or all

dt_inversion = timedelta(weeks=4) # does not include spin-up/down
dt_spinup    = timedelta(weeks=1)
dt_spindown  = timedelta(weeks=1)

for inversion_name in inversion_names:
    start_inversion = start
    while start_inversion<end:
        end_inversion = start_inversion + dt_inversion
        end_inversion = end_inversion - timedelta(days=1) # end_inversion is included, so e.g., for dt=1d we only want to run one day, not two
        print('%s: %s to %s'%(inversion_name,start_inversion.strftime('%Y-%m-%d'),end_inversion.strftime('%Y-%m-%d')))
        submit_inversion(inversion_name, start_inversion, end_inversion, dt_spinup, dt_spindown, task=task)
        
        # We don't need to run end_inversion anymore, so skip one additional day ahead
        start_inversion = end_inversion + timedelta(days=1)
        
    
    
    

