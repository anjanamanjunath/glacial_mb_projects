import math as math
import pandas as pd
import numpy as np
import geopandas as gpd
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import rioxarray
import cartopy
import salem
import os

import oggm 
from oggm import cfg, utils, workflow, tasks, graphics
from oggm import entity_task, global_tasks
from oggm.utils import compile_climate_input
from oggm.core import gis

import warnings
warnings.filterwarnings('ignore')


cfg.initialize(logging_level='WARNING')
cfg.PARAMS['border'] = 10
cfg.PARAMS['use_multiprocessing'] = True 

cfg.PATHS['working_dir'] = utils.gettempdir('OGGM_Trial')
cfg.PATHS['working_dir']

# Greenland is the Randoply Glacier Inventory's 5th region and 6.0 is the current version as of May 2023
rgi_region = '05'
rgi_version = '6'
rgi_dir = utils.get_rgi_dir(version=rgi_version)

path = utils.get_rgi_region_file(region=rgi_region, version=rgi_version)
rgidf = gpd.read_file(path)


# organizing all the MB data (accumalation, ablation, runoff, etc.)
# to query for the paritcular region of south greenland
rgidf = rgidf.query('-55 <= CenLon <= -47')
rgidf = rgidf.query('70 <= CenLat <= 75')

#data check 
print(rgidf.head(n=5))
print(np.shape(rgidf))

base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.1/elev_bands/W5E5_w_data/'

gdirs_hold = []
# removing glaciers for which data doesn't exist in the RGI directory
for glacierID in rgidf.RGIId:
    try:
        glacier_data = workflow.init_glacier_directories(glacierID, from_prepro_level=3, prepro_base_url=base_url, prepro_border=10)
    except:
        print(glacierID+"data directory doesn't exist") 
    else: 
        gdirs_hold.append(glacier_data)


# optimal to save dataset as it takes a while to download! note: path here is hardcoded
gdirs = pd.DataFrame(gdirs_hold)
gdirs.to_csv('/Users/anjanamanjunath/Desktop/classwork/STAT 254 project/data/gdir_data.csv')

gdirs

RGI_ids = []

gdirs = pd.read_csv('/Users/anjanamanjunath/Desktop/classwork/STAT 254 project/data/gdir_data.csv')
for i in range(len(gdirs)):
    idx1 = gdirs.iloc[i,1].index('RGI id:')
    idx2 = gdirs.iloc[i,1].index('\n  Region')

    res = ''
    for idx in range(idx1 + len('RGI id:') + 1, idx2):
        res = res + gdirs.iloc[i,1][idx]

    RGI_ids.append(res)


# topographical data
topo_dict = []

for i in range(1687): 
    topo_dict.append(oggm.shop.bedtopo.add_consensus_thickness(gdirs[i]))

topo_dict = sum(topo_dict, [])
topo_df = pd.DataFrame.from_dict(topo_dict, orient='columns')

topo_df.index = topo_df.ID
topo_df.index.name = 'RGI_ID'
topo_df.drop(columns='ID')

topo_df.to_csv('/Users/anjanamanjunath/Desktop/classwork/STAT 254 project/topo_df.csv')

# climate data 
climate_data = global_tasks.compile_climate_input(gdirs)

climate_data['zmed'] = ('RGI id:', topo_df['zmed'])
climate_data = climate_data.sel(time=slice(2000.0,2020.0))

climate_data.to_csv('/Users/anjanamanjunath/Desktop/classwork/STAT 254 project/climate_gdir_data.csv')

monthly_climate_df = climate_data.to_dataframe()
monthly_climate_df = monthly_climate_df.sort_values(['rgi_id', 'hydro_year', 'hydro_month'], ascending=True)
monthly_climate_df = pd.concat([month_parsing(monthly_climate_df, i) for i in range(12)], axis=1, ignore_index=False)

# Temperature
climate_data.temp.data = climate_data.temp.data + 6/1000*(climate_data.zmed.data - climate_data.ref_hgt.data) 
annual_temp_df = climate_data.temp.where(climate_data.temp > 0.0).groupby('hydro_year').sum()
annual_temp_df = annual_temp_df.rename('Temp')
annual_temp_df = annual_temp_df.to_dataframe()

# Snowfall
annual_snow_df = climate_data.prcp.where(climate_data.temp <= 0.0).groupby('hydro_year').sum()
annual_snow_df = annual_snow_df.rename('Snow')
annual_snow_df = annual_snow_df.to_dataframe()

annual_climate_df = pd.concat([annual_temp_df, annual_snow_df], axis=1)
training_df = pd.merge(monthly_climate_df, annual_climate_df, on=["RGI id:", "hydro_year"])

training_df = training_df.reset_index().merge(topo_df.set_index('ID'), how='left', left_on='RGI id', right_index=True)
training_df = training_df.set_index(['"RGI id', 'hydro_year'])
training_df = training_df.reindex(sorted(training_df.columns), axis=1)
training_df.to_csv('/Users/anjanamanjunath/Desktop/classwork/STAT 254 project/training_df.csv')


Y_df = pd.read_csv('/Users/anjanamanjunath/Downloads/time_series_05/dh_05_rgi60_pergla_rates.csv')
Y_df = Y_df[Y_df.rgi_id.isin(training_df.rgi_id)]
Y_df['target_id'] = np.arange(0, Y_df.shape[0])

Y_df = Y_df.set_index(['rgi_id'])
Y_df = Y_df.sort_values(['rgi_id'])
Y_df = Y_df[['dmdtda', 'err_dmdtda', 'Time period']]

covariates_df = pd.read_csv('/Users/anjanamanjunath/Desktop/classwork/STAT 254 project/data/training_df.csv', index_col=['rgi_id', 'hydro_year'])

aggregation = True

all_data = covariates_df.merge(Y_df, 
                           left_on=['rgi_id'], 
                           right_on=['rgi_id'])

if aggregation:
    all_data = all_data.groupby(['rgi_id', 'Time period']).mean()


all_data

all_data.to_csv('/Users/anjanamanjunath/Desktop/classwork/STAT 254 project/data/all_data.csv')
