def read_insar_stations(insar_stations_gdf_fn, plot_figs=False, save_figs=False, save_to_pickle=True):
    import geopandas as gpd
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    # read xls of locations
    input_path = os.path.join(os.getcwd(), r'files\inputs')
    input_fn = 'NZ_2km_April2022_fixed.xlsx'
    xls_fn = os.path.join(input_path, input_fn)
    df = pd.read_excel(xls_fn, index_col=0)

    if plot_figs:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        s = ax.scatter(df.Lon, df.Lat, c=df["Uz (mm/yr)"], s=df["Area sampled (radius km)"])
        cb = plt.colorbar(s)
        cb.set_label('Uz (mm/yr)')
        if save_figs:
            plt.savefig(r'files/outputs/vlm_insar.png', dpi=300)
            plt.close()
        else:
            plt.show()

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Lon, df.Lat)) \
        .set_crs(4326, allow_override=True, inplace=True)
    # ax = gdf.plot("Uz (mm/yr)", legend=True) # not used for having trouble plotting the colorbar label

    if save_to_pickle:
        work_dir = os.getcwd()
        df_path = os.path.join(work_dir, r'files\dataframes')
        if not os.path.exists(df_path):
            os.mkdir(df_path)

        gdf_output_fn = os.path.join(df_path, insar_stations_gdf_fn)
        gdf.to_pickle(gdf_output_fn)

    return gdf

def convert_searise_csvs(gdf_in, gdf_col_str='rslr', out_path='files\outputs',
                         write_shapefile=True, write_pickle=True, plot_figs=True, save_figs=True):
    '''
    converts the csvs provided by the SeaRise projects to a geopandas geodataframe and saves as png, shp and tif raster
    :param gdf_in: the input geodataframe (insar stations)
    :param gdf_col_str: default = 'rslr'  # others are 'vlm' and 'slr_no_vlm'
    :param out_path: default = 'files\outputs'
    :param plot_figs: default = True
    :param save_figs: default = True
    :return: gdf_out: gdf_in + columns added with rslr, slr_no_vlm and vlm
    '''

    import glob
    import os
    import matplotlib.pyplot as plt
    from libs.modules.utils import only_numerics
    import numpy as np
    import warnings
    import pandas as pd
    import geopandas as gpd

    # read separate csv files in folder
    my_path = os.path.join(os.getcwd(), r'files\inputs\Takiwa_csvs')
    file_list = glob.glob(os.path.join(my_path, r'*med_confidence.csv'))

    my_path_no_vlm = os.path.join(os.getcwd(), r'files\inputs\without_vlm_by_site')
    file_list_no_vlm = glob.glob(os.path.join(my_path_no_vlm, r'*med_confidence.csv'))

    gdf_in['rslr'] = np.nan
    gdf_in['slr_no_vlm'] = np.nan
    pd.options.mode.chained_assignment = None

    # read csv in once to establish yearlist and scenariolist
    # read one csv to obtain year list and scenario list
    df = pd.read_csv(file_list[0], index_col=0)
    year_list = list(df.index.values)
    scenario_list = list(df.columns.values)

    for iyear in range(len(year_list)):
        my_year = year_list[iyear]

        for iscenario in range(len(scenario_list)):
            my_scenario = scenario_list[iscenario]
            print('year = ', str(my_year))
            print('scenario = ', str(my_scenario))

            # read all csvs to gather data for geotiff/shape
            for ifile in range(len(file_list)):
                fn = file_list[ifile]
                df = pd.read_csv(fn, index_col=0)
                my_idx = int(only_numerics(fn[-23:-19]))
                gdf_in['rslr'].iloc[my_idx] = df.loc[my_year][my_scenario]

                fn = file_list_no_vlm[ifile]
                df = pd.read_csv(fn, index_col=0)
                # my_idx = int(only_numerics(fn[-23:-19])) # not necessary because naming is exactly the same
                gdf_in['slr_no_vlm'].iloc[my_idx] = df.loc[my_year][my_scenario]

            gdf_in['vlm'] = gdf_in['slr_no_vlm'] - gdf_in['rslr']

            gdf = gdf_in.to_crs(epsg=2193)

            # save gdf to pickle
            if write_pickle:
                df_path = os.path.join(os.getcwd(), r'files\dataframes')
                out_gdf_fn = gdf_col_str + '_' + str(my_year) + '_' + str(my_scenario) + '.gdf'
                out_gdf_fn = os.path.join(os.path.join(os.getcwd(), df_path), out_gdf_fn)

                if not os.path.exists(df_path):
                    os.mkdir(df_path)
                gdf_output_fn = os.path.join(df_path, out_gdf_fn)
                gdf.to_pickle(gdf_output_fn)

            # save to ESRI shapefile
            if write_shapefile:
                warnings.filterwarnings("ignore")  # https://docs.python.org/3/library/warnings.html#the-warnings-filter
                out_shp_fn = gdf_col_str + '_' + str(my_year) + '_' + my_scenario + '.shp'
                out_shp_fn = os.path.join(os.path.join(os.getcwd(), out_path), out_shp_fn)
                gdf.to_file(out_shp_fn)

            # save to png figure
            if plot_figs:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                # s = ax.scatter(gdf.geometry(), c=gdf["rslr"], s=gdf["Area sampled (radius km)"])
                s = ax.scatter(gdf.Lon, gdf.Lat, c=gdf[gdf_col_str], s=gdf["Area sampled (radius km)"])
                cb = plt.colorbar(s)
                cb.set_label(gdf_col_str + ' (mm) in ' + str(my_year) + ' [' + str(my_scenario) + ']')
                if save_figs:
                    out_fig_fn = gdf_col_str + '_' + str(my_year) + '_' + my_scenario + '.png'
                    out_fig_fn = os.path.join(os.path.join(os.getcwd(), out_path), out_fig_fn)
                    plt.savefig(out_fig_fn, dpi=300)
                    plt.close()
                else:
                    plt.show()

            # save to geotif: gridding routine that builds a raster of outputs
            from libs.modules.my_methods import grid_slr_points
            out_tif_fn = gdf_col_str + '_grid_' + str(my_year) + '_' + my_scenario + '.tif'
            out_tif_fn = os.path.join(os.path.join(os.getcwd(), out_path), out_tif_fn)
            grid_slr_points(gdf, out_tif_fn, gdf_col_str, grid_option=1)

def grid_slr_points(gdf, out_tif_fn, gdf_col_str, grid_option=1, template_raster='dtm_5km.tif'):
    '''

    :param gdf: the geopandas geodataframe
    :param output_tif_fn: output filename for geotiff
    :param gdf_col_str: passed on from 'convert_searise_csvs: default = 'rslr'  # others are 'vlm' and 'slr_no_vlm'
    :param grid_option: option 0: uses geocube, but extrapolates; option 1 (default): uses a template and doesn't extrapolate
    :return: nothing, but it writes a tif file with name output_tif_fn
    '''
    import os
    import numpy as np
    import geopandas as gpd
    import shapely
    import rasterio as rio
    from rasterio import features
    import xarray as xr
    import rioxarray
    from geocube.api.core import make_geocube
    from geocube.rasterize import rasterize_points_griddata, rasterize_points_radial
    from functools import partial

    # option 0: use make_geocube. Doesn't work well enough yet with extrapolation, but might come in handy when nationwide measurements will come in.
    # https://corteva.github.io/geocube/stable/examples/rasterize_point_data.html
    if grid_option == 0:
        cube = make_geocube(
            gdf,
            measurements=[gdf_col_str],
            resolution=(-2000, 2000),
            rasterize_function=rasterize_points_griddata,
            # rasterize_function=partial(rasterize_points_griddata, method="cubic"),
        )
        cube.rslr.where(cube.rslr != cube.rslr.rio.nodata).rio.to_raster(out_tif_fn)

    if grid_option == 1:
        # template_raster = 'dtm_5km.tif'
        # https://gis.stackexchange.com/questions/151339/rasterize-a-shapefile-with-geopandas-or-fiona-python
        rst_fn = os.path.join(os.path.join(os.getcwd(), 'files/inputs'), template_raster) # template
        # rst_fn = os.path.join(os.getcwd(), 'files/inputs/si_50_s4_2000_heads.tif') # template
        template = rio.open(rst_fn)

        # total area for the grid
        xmin, ymin, xmax, ymax = template.bounds #gdf.total_bounds
        # how many cells across and down
        x_res = (xmax - xmin) / template.width
        y_res = (ymax - ymin) / template.height
        # projection of the grid
        crs = 'epsg:2193'
        # create the cells in a loop
        grid_cells = []
        for x0 in np.arange(xmin, xmax + x_res, x_res):
            for y0 in np.arange(ymin, ymax + y_res, y_res):
                # bounds
                x1 = x0 - x_res
                y1 = y0 + y_res
                grid_cells.append(shapely.geometry.box(x0, y0, x1, y1))

        cell = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=crs)
        merged = gpd.sjoin(gdf, cell, how='left', predicate='within')

        # Compute stats per grid cell -- aggregate to grid cells with dissolve
        dissolve = merged.dissolve(by="index_right")
        # put this into cell
        cell.loc[dissolve.index, gdf_col_str] = dissolve[gdf_col_str].values

        # rasterise and chuck to geotiff
        meta = template.meta.copy()
        meta.update(
            compress='lzw',
            count=1,
            dtype=rio.float32,
        )

        with rio.open(out_tif_fn, 'w+', **meta) as out:
            out_arr = out.read(1)
            # this is where we create a generator of geom, value pairs to use in rasterizing
            shapes = ((geom, value) for geom, value in zip(cell.geometry, cell[gdf_col_str]))
            my_raster = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
            out.write(my_raster.astype(rio.float32), 1)

    print('output tif written: '+ out_tif_fn)