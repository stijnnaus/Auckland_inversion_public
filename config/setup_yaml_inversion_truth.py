#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
We define our inversion in a YAML file. Since I'm not too sure about how to 
format a YAML file, I'm just putting the setup in a dictionary here, and then
saving that as a yaml file.
"""

import yaml
import inversion as inv
from datetime import datetime,timedelta
from base_paths import path_code
import numpy as np

def get_inner_bounds_all(inversion_setup):
    
    innerb = {}
    domains = inversion_setup['domains']
    domains_NAME = inversion_setup['domains_NAME']
    for idom,dom in enumerate(domains):
        innerb[dom] = get_inner_bounds_i(idom, domains, domains_NAME, inversion_setup['outer_bounds'])
    return innerb
        
    
    
def get_inner_bounds_i(i, domains, domains_NAME, outer_bounds):
    # Get bounds of inner nested domain
    if i==0:
        # Innermost domain
        return None
    else:
        domain = domains[i-1] # Previous domain in list is largest nested domain
        lon,lat = inv.getLonLatNAME(domains_NAME[domain], bounds=True)
        # Take into account that the nested domain may have modified dimensions
        # described in its "outer_bounds"
        lonb = [ float(np.round(lon.min(),5)), float(np.round(lon.max(),5)) ] if outer_bounds[domain] is None else outer_bounds[domain]['lon']
        latb = [ float(np.round(lat.min(),5)), float(np.round(lat.max(),5)) ] if outer_bounds[domain] is None else outer_bounds[domain]['lat']
        return {'lon':lonb, 'lat':latb}

# Standard, first iteration of inversion

# for label in ['mahk','mahk_new_x', 'mahk_new_t']:
    
label_add = 'truth2'
    
inventory_mahk = 'MahuikaAuckland'
inventory_vprm = 'UrbanVPRM'
for run in ['baseAKLNWP','baseNZCSM']:
    inversion_setup = {}
    
    print('%s_%s'%(run,label_add))
    
    # -------------------- GENERIC INVERSION SET-UP -----------------------
    # These are things I won't often have to change unless I want to fundamentally change how I do the inversion
    inversion_setup['basegrid_label'] = '%s_v1'%run # Where to put the intermediate-grid enhancements
    inversion_setup['name_run'] = run
    inversion_setup['samplelayer'] = '40m'
    inversion_setup['nstepNAME'] = 25
    
    inversion_setup['inversion_is_osse'] = True
    
    inversion_setup['domains'] = ['Mah0p3_in', 'Mah0p3_out', 'In1p5', 'Out7p0'] # From inner to outer
    # Which NAME domain each inversion domain draws from:
    inversion_setup['domains_NAME'] = {'Mah0p3_in':'Mah0p3', 'Mah0p3_out':'Mah0p3', 'In1p5':'In1p5', 'Out7p0':'Out7p0'}
    inversion_setup['nx_base'] = {'Mah0p3_in':36, 'Mah0p3_out':36, 'In1p5':24, 'Out7p0':12}
    inversion_setup['ny_base'] = {'Mah0p3_in':36, 'Mah0p3_out':48, 'In1p5':24, 'Out7p0':12}
    inversion_setup['sites']   = ['MKH','AUT','NWO','TKA']
    
    # -------------------- SPECIFIC INVERSION SET-UP -----------------------
    # These are things I can play with within a given inversion framework
    inversion_setup['inversion_label'] = '%s_%s'%(run,label_add) # Label of the specific inversion set-up (e.g., nx_inv, time window, priors etc)
    inversion_setup['nx_inv'] = {'Mah0p3_in':12, 'Mah0p3_out':6, 'In1p5':4, 'Out7p0':2} # Should be a divisor of nx_base
    inversion_setup['ny_inv'] = {'Mah0p3_in':12, 'Mah0p3_out':6, 'In1p5':4, 'Out7p0':2} # Should be a divisor of ny_base
    # inversion_setup['nx_inv'] = {'Mah0p3_in':12, 'Mah0p3_out':4, 'In1p5':2, 'Out7p0':2} # Should be a divisor of nx_base
    # inversion_setup['ny_inv'] = {'Mah0p3_in':12, 'Mah0p3_out':4, 'In1p5':2, 'Out7p0':2} # Should be a divisor of ny_base
    
    inversion_setup['date_start'] = datetime(2021,12,15) + timedelta(seconds=3600*25)
    inversion_setup['date_end']   = datetime(2023,1,16)
    
    # How do we optimize observations? For now only site-to-site gradients
    inversion_setup['obs_method']     = 'per_site'
    # inversion_setup['obs_gradients']  = { 'NE':[['AUT','MKH'], ['NWO','MKH'], ['TKA','MKH']],
    #                                       'SW':[['AUT','MKH'], ['NWO','MKH'], ['TKA','MKH']] }
    inversion_setup['obs_gradients']  = { 'NE':[['AUT'], ['NWO'], ['TKA'], ['MKH']],
                                          'SW':[['AUT'], ['NWO'], ['TKA'], ['MKH']] }
    
    # List of observational errors that are added together in construct_R:
    # Note that absolute error_type is in ppm, whereas sim_enh is a fraction of the simulated enhancement
    inversion_setup['obs_errors'] = [{'error_type':'absolute', 'error_value':0.5, 'correlation_length':0.001}, \
                                     {'error_type':'sim_enh' , 'error_value':0.3, 'correlation_length':3.0  }]
    
    # Observation filters
    inversion_setup['obs_filters']      = [] #['windspeed','winddirection',]#'time_of_day']
    
    # Time of day, only include prescribed hours (e.g., afternoon)
    inversion_setup['obs_filter_hours'] = list(range(12,18)) # NZST, time_of_day filter
    
    # Windspeed
    inversion_setup['winddata_fill_method'] = 'no_obs_only_model' # How do we gap-fill observed winds?
    inversion_setup['windspeed_limits'] = {'mangere':3, 'mkh':3}
    inversion_setup['afternoon_hours']  = list(range(12,18)) # Don't apply windspeed mask to afternoon hours
    
    # Wind direction: Nested dictionary which tells us which wind directions to include for each group
    # Note that groups match groups in obs_gradients
    inversion_setup['obs_filter_winddir_windows'] =     {
                                                        'SW':{'skytower_se':[200,300], 'MKH':[200,320]}, \
                                                        'NE':{'skytower_nw':[0,90], 'MKH':[0,90]}
                                                        }
    
    # OSSE things
    inversion_setup['inversion_is_osse'] = True
    # inversion_setup['osse_create_obs_method'] = 'from_inv_dict'
    inversion_setup['osse_create_obs_method'] = 'from_file'
    inversion_setup['osse_truth_label']       = 'baseALKNWP_truth'
    
    # Whether we use the prior error matrix to perturb the state, or create a different, "True" error matrix
    inversion_setup['osse_Btrue_is_Bpri'] = True 
    inversion_setup['osse_xpert_B_scale'] = 1.0
    inversion_setup['osse_ypert_R_scale'] = 1.0
    
    inversion_setup['osse_Mah0p3_in_UrbanVPRM_scale'] = 1.5
    inversion_setup['osse_Mah0p3_out_UrbanVPRM_scale'] = 1.5
    
    inversion_setup['osse_Mah0p3_in_%s_scale'%inventory_mahk] = 1.2
    inversion_setup['osse_Mah0p3_out_%s_scale'%inventory_mahk] = 1.2
            
    # Which priors do we subtract from the observations before optimization, 
    # and which do we optimize?
    inversion_setup['priors_fixed']  = {} 
    inversion_setup['priors_to_opt'] = {'Mah0p3_in':[inventory_mahk, inventory_vprm], 
                                        'Mah0p3_out':[inventory_mahk, inventory_vprm],
                                        'In1p5' :['EDGARv8', 'BiomeBGC'], 
                                        'Out7p0':['EDGARv8', 'BiomeBGC']
                                        }
    
    inversion_setup['inventories_for_synth_obs'] = {'Mah0p3_in':[inventory_mahk, inventory_vprm], 
                                        'Mah0p3_out':[inventory_mahk, inventory_vprm],
                                        #'In1p5' :[],#'ODIAC'],            #'BiomeBGC'], 
                                        #'Out7p0':[]#,'EDGARv7']}          #'BiomeBGC']
                                        }
    
    inversion_setup['opt_freq_diurnal']     = 3 # How many hours in one diurnal cycle bin? (nt_diurnal = 24/this)
    inversion_setup['opt_freq_longterm']    = 'daily' # Monthly, weekly, daily or daily_n
    
    inversion_setup['scale_unc_to_total']   = {inventory_mahk:False, inventory_vprm:False, 'BiomeBGC':False, 'EDGARv8':False}
    # inversion_setup['scale_priors']         = {'MahuikaAuckland_new_x':4.78, 'MahuikaAuckland_new_t':0.124}
    inversion_setup['scale_priors']         = {'MahuikaAuckland_new_x':4.78, 'MahuikaAuckland_new_t':0.124,
                                               'UrbanVPRM_new_x':1/4.78, 'UrbanVPRM_new_t':1/0.124}
    
    # Prior error settings
    inversion_setup['scale_err_to_GPP_Resp'] = {'UrbanVPRM':{'GPP':0.5,'Re':1.0}, 
                                                'BiomeBGC' :{'GPP':0.5,'Re':1.0}}
    
    inversion_setup['prior_errors'] = {}
    
    inversion_setup['prior_errors'][inventory_mahk] = {'rel_error':1.0, 
                                                       'min_abs_error':0, 
                                                       'L_temp_long':7, 'L_temp_diurnal':3, 
                                                       'L_spatial':20e3, 'scale_to_total':False}
    
    inversion_setup['prior_errors']['EDGARv8'] = {'rel_error':1.0, 
                                                  'min_abs_error':0, 
                                                  'L_temp_long':7, 'L_temp_diurnal':3, 
                                                  'L_spatial':20e3, 'scale_to_total':False}
    
    inversion_setup['prior_errors'][inventory_vprm] = {'rel_error':1.0, 
                                                    'min_abs_error':0e6, 
                                                    'L_temp_long':7, 'L_temp_diurnal':3, 
                                                    'L_spatial':20e3, 'scale_to_total':False}
      
    inversion_setup['prior_errors']['BiomeBGC'] = {'rel_error':1.0, 
                                                   'min_abs_error':0e6, 
                                                   'L_temp_long':7, 'L_temp_diurnal':3, 
                                                   'L_spatial':20e3, 'scale_to_total':False}
    
    # Option to separate between weekday and weekends in error correlation structure
    # It's based on prior_error_labels, not on prior name
    inversion_setup['separate_weekend_error'] = {}
    inversion_setup['separate_weekend_error'][0] = {'flag':True, 'corr':0.}
    
    inversion_setup['prior_error_labels'] = {inventory_mahk:0, 
                                             'EDGARv8':0, 
                                             inventory_vprm:1, 
                                             'BiomeBGC':1} # Same-label priors are 1:1 correlated
    
    # Define domain for the inner city, as well as aligning a middle domain with Mahuika-Auckland / UrbanVPRM
    inversion_setup['outer_bounds'] = {'Mah0p3_in': {'lon':[174.54, 174.98], 'lat':[-37.1,-36.72]}, 
                                       'Mah0p3_out':{'lon':[174.2,175.1], 'lat':[-37.25,-36.3]}, 
                                       'In1p5' :None, 
                                       'Out7p0':None, 
                                       }
    
    inversion_setup['inner_bounds'] = get_inner_bounds_all(inversion_setup)
    # inversion_setup['inner_bounds']['Out7p0'] = {k:[float(v[0]),float(v[1])] for k,v in inversion_setup['inner_bounds']['Out7p0'].items()}
    # #inversion_setup['inner_bounds']['daemon_12p0'] = None
    
    inversion_setup['yaml_filename'] = "%s/config/inversion_setups/%s.yml"%(path_code,inversion_setup['inversion_label'])
    with open(inversion_setup['yaml_filename'], 'w') as f:
        yaml.dump(inversion_setup, f)

