# -*- coding: utf-8 -*-
"""
Since I use the code on different systems, it's handy to have one central place
where I save the base paths where the core data for the inversion is located.
(footprints, raw priors, observations)
"""

path_code         = '/home/stijn/NIWA/Auckland_inversion/'
path_base         = '/media/stijn/T9/NIWA/'
path_figs         = '%s/Figures/'%path_base

path_footprints   = "%s/cylc-run/"%path_base

path_inv_base     = "%s/"%path_base
path_inv          = "%s/inversions/"%path_inv_base
path_obs_co2      = "%s/observations/Measurement_data_NZ_hourly/"%path_inv_base
path_obs_meteo    = "%s/observations/Wind/"%path_inv_base
path_priors_raw   = "%s/priors/"%path_inv_base
