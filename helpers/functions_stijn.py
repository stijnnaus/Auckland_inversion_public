#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A collection of helper functions that don't fit anywhere else.
They are generic and not specific to the inversion, I use these
for all code analysis I do.

So it might also include some stuff that isn't used anywhere in the
inversion specifically.
"""

import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xar
import math
from datetime import datetime,timedelta
import calendar
import yaml
from scipy.interpolate import griddata,RectBivariateSpline


def get_idx_weekday_weekend(dates):
    # For a list of dates, return a list with 
    #  a) Indices of the dates that are weekdays
    #  b) Indices of the dates that are weekends or holidays
    idx_weekday, idx_weekend_holiday = [], []
    for i,date in enumerate(dates):
        if (date.weekday()>=5) or (is_nz_holiday(date)): 
            idx_weekend_holiday.append(i)
        else:                
            idx_weekday.append(i)
    return idx_weekday, idx_weekend_holiday

def is_nz_holiday(date):
    # Check if date is a NZ holiday. I've hard-coded it for 2021-2025
    
    newyears       = [datetime(2021,1,1),   datetime(2022,1,3),   datetime(2023,1,3),   datetime(2024,1,1),   datetime(2025,1,1)]
    newyears_plus1 = [datetime(2021,1,4),   datetime(2022,1,4),   datetime(2023,1,2),   datetime(2024,1,2),   datetime(2025,1,2)]
    auckland_day   = [datetime(2021,1,29),  datetime(2022,1,31),  datetime(2023,1,30),  datetime(2024,1,29),  datetime(2025,1,27)]
    waitangi       = [datetime(2021,2,8),   datetime(2022,2,7),   datetime(2023,2,6),   datetime(2024,2,6),   datetime(2025,2,6)]
    good_friday    = [datetime(2021,4,2),   datetime(2022,1,4),   datetime(2023,1,2),   datetime(2024,1,2),   datetime(2025,1,2)]
    easter_monday  = [d+timedelta(days=3) for d in good_friday]
    anzac          = [datetime(2021,4,26),  datetime(2022,4,25),  datetime(2023,4,25),  datetime(2024,4,25),  datetime(2025,4,25)]
    kings_day      = [datetime(2021,6,7),   datetime(2022,6,6),   datetime(2023,6,5),   datetime(2024,6,3),   datetime(2025,6,2)]
    # NOTE: Matariki not a public holiday before 2022!
    matariki       = [                      datetime(2022,7,14),  datetime(2023,6,24),  datetime(2024,6,28),  datetime(2025,6,20)]
    labour_day     = [datetime(2021,10,25), datetime(2022,10,24), datetime(2023,10,23), datetime(2024,10,28), datetime(2025,10,27)]
    christmas      = [datetime(2021,12,27), datetime(2022,12,27), datetime(2023,12,25), datetime(2024,12,25), datetime(2025,12,25)]
    boxing_day     = [datetime(2021,12,28), datetime(2022,12,26), datetime(2023,12,26), datetime(2024,12,26), datetime(2025,12,26)]
    
    holidays = newyears + newyears_plus1 + auckland_day + waitangi + good_friday + easter_monday + \
               anzac + kings_day + matariki + labour_day + christmas + boxing_day
                        
    if date.year<2021 or date.year>2025:
        print("Warning: Dates for holidays in year %i have not been hard-coded yet!"%date.year)           
    
    return (date in holidays)

def get_nday_in_year(year):
    nday = 0
    for month in range(1,13):
        nday += calendar.monthrange(year,month)[1]
    return nday

def get_unique_values_from_dict(dicti):
    vals = []
    for _,vals_i in dicti.items():
        if type(vals_i)==np.ndarray or type(vals_i)==list:
            vals += list(vals_i)
        else:
            vals.append(vals_i)
    return np.unique(vals)

def invert_dictionary(dicti):
    vals_unq = get_unique_values_from_dict(dicti)
    dict_inv = {v:[] for v in vals_unq}
    for k,v in dicti.items():
        dict_inv[v].append(k)
        
    return dict_inv

def load_yaml_file(filename):
    with open(filename,'r') as f:
        contents = yaml.full_load(f)
    return contents

def calculate_emission_total(lat, lon, emis,axis=0):
    area_per_gridcell = calc_area_per_gridcell(lat,lon)
    if axis==0:   emitot = emis*area_per_gridcell[:,:]
    elif axis==1: emitot = emis*area_per_gridcell[np.newaxis,:,:]
    elif axis==2: emitot = emis*area_per_gridcell[np.newaxis,np.newaxis,:,:]
    return np.sum(emitot, axis=(-1,-2))

def calc_area_per_gridcell(lats, lons, bounds=False):
    '''
    Calculate area per grid cell in m2
    If lat, lon are grid cell bounds, then bounds=True
    If they are grid cell centers we calculate first grid cell boundaries (approx)
    '''
    
    if not bounds:
        lats = get_gridcell_bounds_from_centers(lats)
        lons = get_gridcell_bounds_from_centers(lons)
    
    dlat = np.abs(np.sin(lats[1:]*np.pi/180.) - np.sin(lats[:-1]*np.pi/180.))
    dlon = np.abs(lons[1:]-lons[:-1])*np.pi/180.
    R_EARTH = 6371e3 # [m]
    area = np.outer(dlat,dlon)*R_EARTH**2
    
    return area

def calc_distance_between_coords(lon1, lat1, lon2, lat2):
    '''
    Calculate distance between 2 lat,lon coordinates in meters. Haversine formula.
    '''
    
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2-lon1
    dlat = lat2-lat1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371e3 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

def mid(a):
    return 0.5*(a[1:] + a[:-1])

def get_gridcell_bounds_from_centers(ll):
    '''
    An approximate way to get from e.g., latitude gridcell centers the boundaries.
    We assume that bounds are in the middle between grid cell centers, 
    and extrapolate the two outer boundaries by the mean grid spacing
    '''
    
    dl = np.abs(np.mean(ll[1:]-ll[:-1]))
    ll_mid = list((ll[1:] + ll[:-1])/2.)
    ll = [ll[0]-dl/2.] + ll_mid + [ll[-1]+dl/2.]
    return np.array(ll)

def adjust_xticks_dates_to_md(ax, xticks=None):
    '''
    Datetime ticks on the x-axis sometimes needlessly show Y-m-d, needlessly
    cluttering the x-axis since almost everything I plot is only for one year.
    This routine removes the year from all but the first x tick.
    '''
    
    if xticks is None:
        # Use current labels
        xticks = [datetime.strptime(t.get_text(),'%Y-%m-%d') for t in ax.get_xticklabels()]
    
    xticklabels = [xticks[0].strftime('%Y-%m-%d')] + [xtick.strftime('%m-%d') for xtick in xticks[1:]]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    
def adjust_xticks_dates_to_m(ax, xticks=None, month_fmt='%m'):
    '''
    Same as above, but for only showing the month, so if I plot a whole year this is the way.
    
    month_fmt sets whether the month is shown as a number or as a word (%m = 01; %b = Jan)
    '''
    
    if xticks is None:
        # Use current labels
        xticks = [datetime.strptime(t.get_text(),'%Y-%m') for t in ax.get_xticklabels()]
    
    xticklabels = [xticks[0].strftime('%Y-'+month_fmt)] + [xtick.strftime(month_fmt) for xtick in xticks[1:]]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

def getLonLatFromLocation(location):
    # Some pre-defined domains for plotting
    location = location.lower()
    if location=='nz_name':
        lonrange = 164.0, 185.8
        latrange = -48.5, -32.5
    elif location=='auckl_name':
        lonrange = 172.2,177.3
        latrange = -38.9,-34.8
    elif location=='auckl_zoom':
        lonrange = 174.2,175.2
        latrange = -37.3,-36.2
    else:
        raise KeyError("Invalid location %s"%location)
    
    return lonrange,latrange

def regrid(lon_in,lat_in,flux,lon_out,lat_out,method_overall='griddata',method1='linear',force_pos=False,conserve_mass=True):
    
    flux[np.isnan(flux)] = 0.
    
    if method_overall=='griddata':
        # Option 1: Use scipy griddata. Slower, but works for non-rectangular grids
        x,y = np.meshgrid(lon_in,lat_in)
        flux_new = griddata((x.ravel(),y.ravel()), flux.ravel(), (lon_out, lat_out), method=method1,fill_value=0)
        
    elif method_overall=='RectBivariateSpline':
        # Option 2: Spline fit
        # Not sure which works better, but this one is faster
        ip = RectBivariateSpline(lat_in, lon_in, flux)
        flux_new = ip(lat_out, lon_out)
        
        # The spline fit sometimes puts non-zero values when it extrapolates
        # To me that's weird, so where it extrapolates I just put everything to zero
        flux_new[lat_out<lat_in.min(),:] = 0.0
        flux_new[:,lon_out<lon_in.min()] = 0.0
        flux_new[lat_out>lat_in.max(),:] = 0.0
        flux_new[:,lon_out>lon_in.max()] = 0.0
        
        if force_pos:
            flux_new[flux_new<0] = 0.0
        if conserve_mass:
            flux_new = calc_conserve_mass(lon_in,lat_in,flux, lon_out,lat_out,flux_new)
            
    elif method_overall=='xesmf':
        # Option 3: Use the xesmf regridder, which can be set to conservative, useful for e.g., emissions
        # For this we first need to convert to xarray
        regr = make_xesmf_regridder(lon_in, lat_in, lon_out, lat_out, method1)
        flux_new = regr(flux)
        
    return flux_new

def make_xesmf_regridder(lon_in, lat_in, lon_out, lat_out, method='conservative', bounds=False):
    
    import xesmf
    if bounds:
        
        ds_in  = {"lat":mid(lat_in) , "lon":mid(lon_in),  "lat_b":lat_in , "lon_b":lon_in}
        ds_out = {"lat":mid(lat_out), "lon":mid(lon_out), "lat_b":lat_out, "lon_b":lon_out}
        
    else:
        ds_in  = xar.Dataset({"lat": (["lat"], lat_in, {"units": "degrees_north"}),
                             "lon": (["lon"], lon_in, {"units": "degrees_east"})})
        ds_out = xar.Dataset({"lat": (["lat"], lat_out, {"units": "degrees_north"}),
                             "lon": (["lon"], lon_out, {"units": "degrees_east"})})
    
    return xesmf.Regridder(ds_in, ds_out, method)

def calc_conserve_mass(lon_ori, lat_ori, mass_ori, lon_new, lat_new, mass_new):
    """
    Make sure that mass_ori and mass_new are the same (e.g., after regridding).
    Take into account that the new grid can be smaller than the old grid, so that we should
    only scale the relevant part. If this is the case then the simple implementation here will
    not be perfect for all cases, but at least it'll hopefully reduce the errors made.
    If the new grid is larger than the old grid, then that's fine, because the new grid will be
    zero where the old grid does not reach.
    Mass has dimensions (nlat,nlon) and is in units per area
    """
    
    area_ori = calc_area_per_gridcell(lat_ori, lon_ori)
    masklon = (lon_ori>=lon_new.min()) & (lon_ori<=lon_new.max())
    masklat = (lat_ori>=lat_new.min()) & (lat_ori<=lat_new.max())
    mask = np.outer(masklat,masklon)
    mass_ori_tot = np.nansum((mass_ori*area_ori)[mask])
    
    area_new = calc_area_per_gridcell(lat_new, lon_new)
    mass_new_tot = np.nansum(mass_new*area_new)    
    
    # Ensure that if mass is 0 it doesn't do something strange
    # (e.g. sometimes one timestep doesn't have emissions)
    if mass_ori_tot==0:
        scalar = 0.
    else:
        scalar = (mass_ori_tot/mass_new_tot)
    
    return scalar*mass_new

def getSiteCoords():
    coords = {}
    coords['MKH'] = 174.5615, -37.0507
    coords['AUT'] = 174.7657, -36.8543
    coords['NWO'] = 174.8239, -36.8621
    coords['TAK'] = 174.7993, -36.8265
    return coords

def plotSquare(ax, xx, yy, color='b', lw=5, alpha=1.0, linestyle='-', zorder=5):
    args = {'linewidth':lw, 'color':color, 'alpha':alpha, 'linestyle':linestyle, 'zorder':zorder}
    [ax.plot([xx[0],xx[1]], [yy[i],yy[i]], **args) for i in [0,1]]
    [ax.plot([xx[i],xx[i]], [yy[0],yy[1]], **args) for i in [0,1]]
    
def plotSiteLocations(ax,text=False,s=500):
    coords = getSiteCoords()
    xoffsets = [-0.035, -0.085, -0.04, 0.055]
    yoffsets = [+0.032, -0.02, +0.07, -0.04]
    for ii,(site,(xi,yi)) in enumerate(coords.items()):
        ax.scatter(xi,yi, marker='X', linewidth=4, s=500, edgecolor='k',facecolor='w')
        if text:
            ax.text(xi+xoffsets[ii],yi+yoffsets[ii],site, fontsize=20, color='k', bbox={'alpha':0.8,'color':'white'})
        
def calc_distance_latlon(lon1,lon2, lat1,lat2):
    # Depreciated, I now use Haversine
    return calc_distance_between_coords([lat1,lon1], [lat2,lon2])

def makeMapCartopy(lonRange=(-180,+180), latRange=(-90, 90), location=None, ax=None, landcolor='gray', 
                   oceancolor='lightblue', bordercolor='white', meridians=True, fig_scale=1.0, zorder=0):
    
    if location is not None:
        lonRange, latRange = getLonLatFromLocation(location)
    
    if ax is None:
        returnFig = True
        size = calcAspectratioFromLonLat(lonRange, latRange)
        size = (size[0]*fig_scale,size[1]*fig_scale)
        fig = plt.figure(figsize=size)
        central_longitude = 0.5*(lonRange[0]+lonRange[1])
        proj = ccrs.PlateCarree(central_longitude=central_longitude)
        ax = fig.add_subplot(1,1,1,projection=proj)
    else:
        returnFig = False
    
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cartopy.feature.OCEAN,  facecolor=oceancolor)
    ax.add_feature(cartopy.feature.LAND,   facecolor=landcolor)
    ax.add_feature(cartopy.feature.BORDERS, edgecolor=bordercolor, linewidth=2.)
    ax.set_extent([lonRange[0],lonRange[1],latRange[0],latRange[1]], crs=ccrs.PlateCarree())
    if meridians:
        drawMeridians(ax)
    
    if returnFig: 
        return fig, ax, central_longitude
    else:         
        return ax
    
def calcAspectratioFromLonLat(lonRange=None, latRange=None, location=None):
    if location is not None:
        lonRange, latRange = getLonLatFromLocation(location)
    sizeY = 5 # Standard
    dlon  = np.abs( lonRange[1] - lonRange[0] )
    dlat  = np.abs( latRange[1] - latRange[0] )
    sizeX = 2 + (sizeY * (dlon/dlat))
    return (sizeX, sizeY)
    
def drawMeridians(ax):
    
    grid = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0, color='gray', alpha=0.5, linestyle='--')
    
    grid.right_labels = False
    grid.top_labels = False
    
def clipVals(vals, valsRange=None, minstd=-2, plusstd=+3):
    if valsRange is None:
        low, high = calcClipRange(vals, minstd, plusstd)
    else:
        low, high = valsRange
    return np.clip(vals, low, high)

def calcClipRange(vals, minstd=2, plusstd=3):
    
    valsWithoutExtremes = removeExtremes(vals, nstd=3)
    mean = np.nanmean(valsWithoutExtremes)
    std  = np.nanstd(valsWithoutExtremes)
    low  = mean - minstd*std
    high = mean + plusstd*std
    return low, high
    
def removeExtremes(vals, nstd=3):
    # Remove values that fall outside nstd number of standard deviations.
    # We do this iteratively until all values are in the correct range.
    maskInside = np.zeros_like(vals, dtype=bool)
    valsNew = np.copy(vals)
    while (~maskInside).sum()>0:
        mean = np.nanmean(valsNew)
        std  = np.nanstd (valsNew)
        maskInside = (valsNew < (mean+nstd*std)) & (valsNew > (mean-nstd*std))
        valsNew = valsNew[maskInside]
    return valsNew

def sron_cmap(colormap="rainbow_PiBr",nan_transparent=True,bad_def=None):
    """docstring for sron_colormap
    """
    import matplotlib

    if colormap == "rainbow_PiBr": # rainbow extended from red to brown
        part_1=[[232, 236, 251], [221, 216, 239], [209, 193, 225], [195, 168, 209], [181, 143, 194], [167, 120, 180], 
                [155, 98, 167], [140, 78, 153]]
        part_2=[[111, 76, 155], [96, 89, 169], [85, 104, 184], [78, 121, 197], [77, 138, 198], [78, 150, 188], [84, 158, 179], 
                [89, 165, 169], [96, 171, 158], [105, 177, 144], [119, 183, 125], [140, 188, 104], [166, 190, 84], [190, 188, 72],
                [209, 181, 65], [221, 170, 60], [228, 156, 57], [231, 140, 53], [230, 121, 50], [228, 99, 45], [223, 72, 40], 
                [218, 34, 34]]
        part_3=[[184, 34, 30], [149, 33, 27], [114, 30, 23], [82, 26, 19]]
        cmap_def=part_2 + part_3

        if bad_def is None:
            bad_def=[119.0,119.0,119.0]
            
        result = matplotlib.colors.LinearSegmentedColormap.from_list(colormap, np.array(cmap_def)/256.0)
        if nan_transparent == False:
            result.set_bad(np.array(bad_def)/256.0,1.)

    return result


def make_mask_bounds_out(arr, lims):
    return (arr<lims[0]) & (arr>lims[1])

def make_mask_bounds_in(arr, lims):
    return (arr>lims[0]) & (arr<lims[1])

def select_hours_from_fulldays(dates_hour, dates_fullday, arr):
    '''
    A function to select a selection of hours from full daily cycles.
     - dates_hour is a 1-D array with datetime objects for the hours we want
     - dates_fullday is a 1-D array with the unique days present in arr
     - arr is some array with a full daily cycle for each day in dates_fullday.
        It can have dimensions (len(dates_fullday)*24) or (len(dates_fullday),24)
    '''
    
    arr = arr.reshape(len(dates_fullday),24)
    arr_selection = np.zeros(len(dates_hour))
    for i,date in enumerate(dates_hour):
        iday = np.where(np.array(dates_fullday)==datetime(date.year,date.month,date.day))[0][0]
        arr_selection[i] = arr[iday,date.hour]
    return arr_selection
    
def find_unique_months(dates):
    '''
    Input list of datetime objects, select unique months (y,m combinations)
    and return as datetime objects.
    '''
    
    return np.unique([datetime(d.year,d.month,1) for d in dates])
    
    
def make_dates_array(date_start, date_end, dt=timedelta(days=1)):
    date_i = date_start
    dates = []
    while date_i<=date_end:
        dates.append(date_i)
        date_i += dt
    return np.array(dates)
    
    
    
    
    
    
    
    
    