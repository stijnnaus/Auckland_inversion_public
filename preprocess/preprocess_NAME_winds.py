from read_meteo_NAME import RetrievorWinddataNAME
from datetime import datetime,timedelta

sites = ['ManukauHeads', 'AucklandAirport', 'MOTAT', 'SkyTower', 'Mangere', 'AucklandUni', 'Takarunga', 'Pourewa']
invnames = ['baseAKLNWP_base']
dates = [datetime(2022,1,1) + timedelta(days=i) for i in range(365)]
for invname in invnames:
    retr = RetrievorWinddataNAME(invname)
    ws,wd,pbl,temp,rh = retr.retrieve_winddata_NAME(dates, sites, overwrite=True)
