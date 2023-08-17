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

# run prior to running functions to establish directories

cfg.initialize(logging_level='WARNING')
cfg.PARAMS['border'] = 10
cfg.PARAMS['use_multiprocessing'] = True 

cfg.PATHS['working_dir'] = utils.gettempdir('OGGM_Trial')
cfg.PATHS['working_dir']


def get_glaciers(extent, region):
    rgi_region = region # greenland is the RGI's 5th region
    rgi_version = '6' # 6.0 is the current version as of May 2023  <- can be modified when a new version is released
    rgi_dir = utils.get_rgi_dir(version=rgi_version)

    path = utils.get_rgi_region_file(region=rgi_region, version=rgi_version)
    rgi_df = gpd.read_file(path)

    min_lon = extent[0]
    max_lon = extent[1]
    min_lat = extent[2]
    max_lat = extent[3]

    # to query for the particular region of south greenland based on extent provided

    rgi_roi = rgi_df.query(min_lon+' <= CenLon <= '+max_lon)
    rgi_roi = rgi_roi.query(min_lat+' <= CenLat <= '+max_lat)

    # print(rgi_roi.head(n=5)) <- to understand data structure
    # print(np.shape(rgi_roi))

    base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.1/elev_bands/W5E5_w_data/'

    gdirs_hold = []

    # removing glaciers for which data doesn't exist in the RGI directory

    for gIDs in rgi_df.RGIId:
        try:
            glacier_data = workflow.init_glacier_directories(gIDs, from_prepro_level=3, prepro_base_url=base_url, prepro_border=10)
        except:
            print(gIDs+"data directory doesn't exist") 
        else: 
            gdirs_hold.append(glacier_data)

    # write data to csv to store: pulling the data takes a very long time, 
    # so best to write data to csv to access in further steps, and only directly pull data from the OGGM as needed

    gd = pd.DataFrame(gdirs_hold)
    gd.to_csv('/**/data/gdir_all_data.csv')

def get_topo_climate(csv_path):
    glacier_id = pd.read_csv(csv_path)
    RGI_ids = []

    for i in range(len(glacier_id)):
        idx1 = glacier_id.iloc[i,1].index('RGI id:')
        idx2 = glacier_id.iloc[i,1].index('\n  Region')

        res = ''
    for idx in range(idx1 + len('RGI id:') + 1, idx2):
        res = res + glacier_id.iloc[i,1][idx]

    RGI_ids.append(res)

    # get all topographic data
    topo_dict = []

    for i in range(1687): 
        topo_dict.append(oggm.shop.bedtopo.add_consensus_thickness(glacier_id[i]))

    topo_dict = sum(topo_dict, [])
    topo_df = pd.DataFrame.from_dict(topo_dict, orient='columns')

    topo_df.index = topo_df.ID
    topo_df.index.name = 'RGI_ID'
    topo_df.drop(columns='ID')

    # get all climate data
    topo_df.to_csv('/**/data/topo_df.csv')

    climate_data = global_tasks.compile_climate_input(glacier_id)

    climate_data['zmed'] = ('RGI id:', topo_df['zmed'])
    climate_data = climate_data.sel(time=slice(2000.0,2020.0))

    climate_data.to_csv('/**/data/climate_gdir_data.csv')

    # seperating based on whether it was cold enough for the precipitation to be rain (accumulation) 
    # or snowfall (ablation)
    climate_data.temp.data = climate_data.temp.data + 6/1000*(climate_data.zmed.data - climate_data.ref_hgt.data) 
    temp_df = climate_data.temp.where(climate_data.temp > 0.0).groupby('hydro_year').sum()
    temp_df = temp_df.rename('temp')
    temp_df = temp_df.to_dataframe()

    snowfall_df = climate_data.prcp.where(climate_data.temp <= 0.0).groupby('hydro_year').sum()
    snowfall_df = snowfall_df.rename('snowfall')
    snowfall_df = snowfall_df.to_dataframe()

    climate_df = pd.concat([temp_df, snowfall_df], axis=1)


# note the data for albedo and debris cover is directly downloaded from https://nsidc.org/data/explore-data
# TO UPDATE: preprocessing code for albedo and debris data, ELA line

def train_test_data(climate_data, climate_df, topo_df, MB_file_path):
    training_df = pd.merge(climate_df, climate_data, on=["RGI id:"])

    training_df = training_df.reset_index().merge(topo_df.set_index('ID'), how='left', 
                                                  left_on='RGI id', right_index=True)
    training_df = training_df.set_index(['"RGI id']).reindex(sorted(training_df.columns), axis=1)

    training_df.to_csv('**/data/training_df.csv')

    # Mass Balance file path: /Users/anjanamanjunath/Downloads/time_series_05/dh_05_rgi60_pergla_rates.csv
    # data from Hugonnet et. al (2021) download link available with doi
    Y_df = pd.read_csv(MB_file_path)
    Y_df = Y_df[Y_df.rgi_id.isin(training_df.rgi_id)]
    Y_df['target_id'] = np.arange(0, Y_df.shape[0])

    Y_df = Y_df.set_index(['rgi_id'])
    Y_df = Y_df.sort_values(['rgi_id'])
    Y_df = Y_df[['dmdtda', 'err_dmdtda', 'Time period']]

    covariates_df = pd.read_csv('/**/data/training_df.csv', index_col=['rgi_id', 'hydro_year'])

    aggregation = True

    all_data = covariates_df.merge(Y_df, 
                           left_on=['rgi_id'], 
                           right_on=['rgi_id'])

    if aggregation:
        all_data = all_data.groupby(['rgi_id']).mean()

    all_data

    all_data.to_csv('/**/data/all_data.csv')






    





