# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from libs.modules.my_methods import read_insar_stations, convert_searise_csvs
import pandas as pd
import os

# read geodataframe of station locations. If not existing, build it again and store as gdf
insar_stations_gdf_fn = 'insar_stations_gdf'

fn = os.path.join(os.path.join(os.getcwd(), r'files\dataframes'), insar_stations_gdf_fn)
if not os.path.exists(fn):
    gdf_insar_stations = read_insar_stations(insar_stations_gdf_fn)
else:
    gdf_insar_stations = pd.read_pickle(fn)

print('insar station locations have been read')

# build dataframe of SLR with and without VLM and export to png, shp and tif files
convert_searise_csvs(gdf_insar_stations)
convert_searise_csvs(gdf_insar_stations, gdf_col_str='slr_no_vlm', write_shapefile=False, write_pickle=False)
convert_searise_csvs(gdf_insar_stations, gdf_col_str='vlm', write_shapefile=False, write_pickle=False)

print('done')