#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 13:12:58 2023

Helper functions for the inversion that are not included in inversion_base.
It grew a bit organically what's included where, so the distinction between
what's in inversion_base and what's here isn't always clear, but the idea was
that here only functions that do not require too many inversion/rc variables are
included.

@author: nauss
"""

import numpy as np
from datetime import datetime,timedelta
import glob,os
import functions_stijn as st
import calendar
import tarfile
from base_paths import path_code

def load_inversion_setup(inversion_name):
    fname = '%s/config/inversion_setups/%s.yml'%(path_code, inversion_name)
    if os.path.exists(fname):
        return st.load_yaml_file(fname)
    else:
        raise ValueError("Inversion set-up not found: %s (%s)"%(inversion_name, fname))

def get_path_name():
    from base_paths import path_footprints
    return path_footprints

def get_path_name_output(subfolder):
    path = '%s/%s/share/OUTPUT/'%(get_path_name(), subfolder)
    return path

def get_path_name_input(subfolder):
    path = '%s/%s/share/INPUT/'%(get_path_name(), subfolder)
    return path

def get_name_tar_preamble(subfolder):
    # I tarred the NAME output a bit awkwardly such that I always need to provide
    # the full path where the data was originally tarred
    return 'home/nauss/cylc-run/%s/share/OUTPUT/'%subfolder

def get_path_in_tar(subfolder, date):
    # I tarred the NAME output a bit awkwardly such that I always need to provide
    # the full path where the data was originally
    path0 = get_name_tar_preamble(subfolder)
    date_str = date.strftime('%Y%m%dT%H')
    path = os.path.join(path0, date_str, 'nameiii/data')
    return path

def get_filename_winddata_NAME(subfolder, date, input_timezone):
    
    if input_timezone=='nzst':
        # NAME output is in UTC so pick the file corresponding to..
        date = date - timedelta(seconds=12*3600)
    
    path_base = get_path_name_output(subfolder)
    date_str = date.strftime('%Y%m%dT%H')
    fname = '%s/%s_Met.tar.gz'%(path_base, date_str)
    return fname

def get_filename_in_tar_meteo(subfolder, date, site, input_timezone):
    if input_timezone=='nzst':
        # NAME output is in UTC so pick the file corresponding to..
        date = date - timedelta(seconds=12*3600)
    path = get_path_in_tar(subfolder, date)
    fname = os.path.join(path, 'MetOutput_C1_%s.txt'%site)
    return fname




def read_footprint_hourly(date_start, nsite, nstepNAME, dt=3600, domain='standard_in',subfolder='2022', path0=None,
                          field_label='', tarred=False, tar_label='', path0_tar=None, timezone='NZST'):
    
    '''
    Read footprints from hourly files. E.g., for 26 hours backwards run, 25 files
    will be read and put in one array.
    Input arguments:
        - date_start : The datetime corresponding to one specific simulation
        - nsite      : Number of sites (= number of columns in footprint file)
        - nstepNAME  : Number of NAME timesteps
        - dt         : Timestep between output files in secs (as the function name implies, this is mostly 3600)
        - domain     : Domain name so I know lon,lat dimensions
        - subfolder  : Basically the simulation name, subfolder in cylc-run
        - path0      : Very optional, but in case the NAME output is not in the standard place
        - field_label: Label(/addendum) of footprint files (e.g., grid1, grid2 etc)
        - tarred     : Are the output files tarred or not? If so need the next two arguments:
          - tar_label: Similar to field_label, but for the tarfile (normally contains domain + samplelayer)
          - path0_tar: The path inside the tarfile to where the footprint files are
        - timezone   : Which timezone the startdate is in
        
    
    '''
    
    if timezone.lower()=='utc':
        pass
    elif timezone.lower()=='nzst':
        # NZST to UTC
        date_start -= timedelta(seconds=12*3600)
    else:
        raise KeyError("Unknown timezone for reading footprints: %s"%timezone)
    
    
    date_str = date_start.strftime('%Y%m%dT%H')
    if path0 is None:
        path0 = get_path_name_output(subfolder)
    
    if path0_tar is None:
        # Default path preamble inside tar file
        path0_tar = get_name_tar_preamble(subfolder)
    path1 = '%s/nameiii/data'%date_str
    
    lons, lats = getLonLatNAME(domain=domain) 
    
    if tarred:
        fname_tar = '%s%s.tar.gz'%(date_str, tar_label)
        tarf = tarfile.open('%s/%s'%(path0,fname_tar))
    
    timesteps  = np.zeros(nstepNAME, dtype=object)
    footprints = np.zeros((nsite, nstepNAME, len(lats), len(lons)))
    
    # Loop over reading all footprint files in one simulation
    for istep in range(0,nstepNAME): 
        # Note: The T1 footprint file doesn't correspond to a timestep, so we skip it.
        #     If there are 26 footprint files (so up to T26) there are actually only 25 timesteps,
        #     and the first timestep is represented by the T2 file. The T1 file is meaningless.
        ifile = istep+1
        
        date_i = date_start - timedelta(seconds=dt*ifile)
        timesteps[istep] = date_i
        date_str = date_i.strftime('%Y%m%d%H%M')
        
        fname = 'Fields_%s_C1_T%i_%s.txt'%(field_label, ifile+1, date_str)
        if tarred:
            try:
                tarname = os.path.join(path0_tar, path1, fname)
                f = tarf.extractfile(tarname)
            except:
                print('Tarfile %s'%fname_tar)
                raise
        else:
            f = open('%s/%s/%s'%(path0,path1,fname))
            
        for i,line in enumerate(f.readlines()[37:]):
            if tarred:
                # Is read as binary
                line = str(line)[2:]
            
            try:
                line = line.split(',')
                ix = int(line[0])-1
                iy = int(line[1])-1
                for isite in range(nsite):
                    footprints[isite,istep,iy,ix] = float(line[4+isite])
            except:
                print(fname)
                print(line)
                raise
                
    if tarred:
        tarf.close()
        
    if timezone.lower()=='utc':
        pass
    elif timezone.lower()=='nzst':
        # UTC to NZST
        timesteps += timedelta(seconds=12*3600)
    else:
        raise KeyError("Unknown timezone for reading footprints: %s"%timezone)
                    
    return timesteps,footprints


def getLonLatNAME(domain, bounds=False):
    # Hard-coded, should be the same as in the NAME config file
    domain = domain.lower()
    if   domain == 'in1p5' or domain == 'in':
        # Inner domain ~1.5km2
        xlo,xhi,nx = 172.2,177.3,306
        ylo,yhi,ny = -38.9,-34.8,303
        
    elif domain == 'out7p0' or domain == 'out':
        # Outer domain with increased resolution ~7km2
        xlo,xhi,nx = 164.0, 185.8, 294
        ylo,yhi,ny = -48.5, -32.5, 264
        
    elif domain == 'akl_0p3':
        # The Auckland model domain / grid (approx)
        xlo,xhi,nx  = 173.8455, 175.6455, 300
        ylo,yhi,ny  = -37.7309, -35.9309, 300
        
    elif domain == 'glenbrook':
        # Glenbrook small domain (for Molly's simulations)
        xlo,xhi,nx  = 174.40,175.00,36
        ylo,yhi,ny  = -37.35,-36.90,34
        
    elif domain == 'mah':
        # The Mahuika domain / grid (approx)
        xlo,xhi,nx  = 174.053,175.388,297
        ylo,yhi,ny  = -37.351,-35.97,306
        
    elif domain == 'mah0p3':
        # A grid that covers both Mahuika and AKL0p3 at 0p3 resolution
        xlo,xhi,nx  = 173.8455,175.388,514
        ylo,yhi,ny  = -37.7309, -35.975, 588
        
    elif domain == 'daemon_12p0':
        xlo,xhi,nx  = 160, 185.8, 259
        ylo,yhi,ny  = -50, -30  , 201
        
    elif domain == 'daemon_1p5':
        xlo,xhi,nx  = 160, 185.8, 1912
        ylo,yhi,ny  = -50, -30  , 1482
        
    elif domain == 'flasks_tim':
        xlo,xhi,nx  = 172, 179.2, 533
        ylo,yhi,ny  = -39.5, -33, 481
        
    else:
        raise KeyError("Unknown NAME domain for retrieving lon,lat: %s"%domain)
        
    lonc = np.linspace(xlo, xhi, nx)
    latc = np.linspace(ylo, yhi, ny)
    
    dlon = np.mean(np.abs(lonc[1:]-lonc[:-1]))
    dlat = np.mean(np.abs(latc[1:]-latc[:-1]))
    
    lonb = np.linspace(xlo-dlon/2., xhi+dlon/2., nx+1)
    latb = np.linspace(ylo-dlat/2., yhi+dlat/2., ny+1)
    if bounds:
        return lonb, latb
    else:
        return lonc, latc

def determine_num_timesteps_long(date_start, date_end, freq, nstepNAME):
    '''
    Determine the number of timesteps for an inversion that assimilates observations
    from date_start to date_end. 
    This is only for the timestepping between days (hence "long"), not for the timesteps within day.
    '''
    
    timesteps = get_opt_timesteps(date_start, date_end, freq, nstepNAME)
    return len(timesteps)

def get_opt_timesteps(date_start, date_end, freq, nstepNAME, pos='mid'):
    
    # Sensitivity to emissions goes back nstepNAME hours from first observation
    date0 = (date_start-timedelta(seconds=3600*nstepNAME))
    if date0.hour!=0:        
        raise ValueError("Optimization period is not a whole number of days. \n" + 
                         "This gives problems in diurnal v daily correlations. \n" +
                         "Note that emissions are optimized backwards from start of simulation, \n" + 
                         "so (startdate-nstepNAME) should be start of a day.")
        
    dates = []
    date_curr = date0
    while date_curr<=date_end:
        dt = get_timestep_from_freq(date_curr, freq)
        
        if   pos.lower()=='start':
            date_i = date_curr
        elif pos.lower()=='mid':
            date_i = date_curr + dt/2
        elif pos.lower()=='end':
            date_i = date_curr + dt
        else:
            raise KeyError("Unknown argument for pos: has to be mid,start or end")
            
        # Especially in case of pos==mid/end, we want to make sure that the final timestep
        # is the end date. So if the frequency is weekly, we don't want to include the full
        # final week if it continues beyond the end date
        date_i = np.min([date_i, date_end])
        dates.append( date_i )
            
        date_curr += dt
        
    return np.array(dates)


def add_up_correlated_uncertainties(tsteps, unc, corlen):
    """
    Function to add up correlated uncertainties (i.e., uncertainty on the sum)
    tsteps and unc are 1-d arrays describing timesteps and uncertainty on each
    timestep, where corlen is the correlation length of the uncertainties.
    
    I think I only need this function for adding up uncertainty totals of the different 
    inversions covering different parts of the time periods (hence the naming), 
    but in principle it works also in space as long as it's 1-D vectors.
    """
    
    if type(corlen) == timedelta:
        corlen = corlen.days + corlen.seconds/(24*3600)
        
    nt = len(tsteps)
    unc_matrix = np.zeros((nt,nt))
    for i1,t1 in enumerate(tsteps):
        for i2,t2 in enumerate(tsteps):
            dt = np.abs(t1-t2)
            
            if type(dt) == timedelta:
                dt = dt.days + dt.seconds/(24*3600)
                
            r = np.exp(-dt/corlen)
            unc_matrix[i1,i2] = r * unc[i1] * unc[i2]
    
    unc_tot = np.sqrt(np.sum(unc_matrix))
    return unc_tot

def get_timestep_from_freq(date, freq):
    '''
    Determine the timestep from current date and frequency.
    I.e., the inversion timestep covers "date" to "date+dt" 
    First timestep can be a bit awkward sometimes due to the 26 hour backwards
    run, so we correct for this. E.g., if the first timestep doesn't start at hour 0,
    I prefer to cut the first timestep short, so that all other timesteps do start
    at hour 0 (as opposed to e.g., starting in the daily optimization every step at hour 22)
    '''
            
    if freq=='daily':
        # Number of hours until the next day
        dt = timedelta(days=1)-timedelta(seconds=3600*date.hour)
    elif freq=='monthly':
        nday_in_month = calendar.monthrange(date.year,date.month)[1]
        # Number of days until the next month
        nday = nday_in_month - date.day 
        nhour = 24-date.hour
        dt = timedelta(days=nday,seconds=3600*nhour)
    elif freq=='weekly':
        dt = timedelta(days=7)-timedelta(seconds=3600*date.hour)
    else:
        raise KeyError("Unknown obs frequency: %s"%freq)
        
    return dt

def calculate_enhancements_from_emis_footprints(emis, footprints, area_per_gridcell, time_integrated=False, fpunit='conc'):
    '''
    Input units of emissions and footprints HAVE TO BE:
          emis: [kg m-2 s-1] ; dimensions (time, lat, lon)        -> if time-integrated no time dimension!
    footprints: [g s m-3]    ; dimensions (nsite, time, lat ,lon) -> if time-integrated no time dimension!
    Footprints can now also be in ppm, since NAME can output this unit too!
    '''
    
    Mco2 = 44.01 # gCO2 mol-1
    R = 8.31     # J K-1 mol-1
    P = 1e5      # kg m-1 s-2
    T = 290      # K
    conv = (R*T)/(Mco2*P) # m3 gCO2-1 (converting between volume (e.g., ppm) and mass CO2)
    
    nsite = len(footprints)
    xco2 = np.zeros(nsite, dtype=float)
    cco2 = np.zeros(nsite, dtype=float)
    
    # Footprints from [g.s/m3] to [s/m]
    fp = footprints/3600.*area_per_gridcell
    
    # Emissions from [kg/m2/s] to [g/m2/s]
    emis = np.copy(emis)*1e3
    
    if time_integrated:
        # If footprints and emissions are time-integrated / averaged, then there is one less dimension
        cco2 = np.sum(fp[:,:,:]*emis[np.newaxis,:,:], axis=(1,2))
    else:
        cco2 = np.sum(fp[:,:,:,:]*emis[np.newaxis,:,:,:], axis=(1,2,3))
        
    if fpunit=='conc':
        xco2 = cco2 * conv * 1e6 # [gCO2/m3] to [ppm]
    elif fpunit=='ppm':
        xco2 = np.copy(cco2)
        cco2 = xco2 / (conv * 1e6)
    else:
        raise KeyError("Unknown footprint unit: %s"%fpunit)
        
    return cco2,xco2
