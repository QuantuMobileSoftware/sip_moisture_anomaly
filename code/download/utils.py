import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio


def get_tiles(aoi_path, sentinel_tiles_path):
    """
    Returns Sentinel-2 tiles that intersects with specified AoI.

        Parameters:
            aoi_path (str): Path to geojson/shp file with AoI to process.
            sentinel_tiles_path (str): Path to geojson/shp file with all Sentinel-2 tiles.

        Returns:
            date_tile_info (GeoDataFrame): Filtered tiles (tileID, geometry, date).
    """
    aoi_file = gpd.read_file(aoi_path)
    sentinel_tiles = gpd.read_file(sentinel_tiles_path)
    sentinel_tiles.set_index("Name", drop=False, inplace=True)

    best_interseciton = {"tileID": [], "geometry": []}
    rest_aoi = aoi_file.copy()

    while rest_aoi.area.sum() > 0:
        res_intersection = gpd.overlay(rest_aoi, sentinel_tiles, how="intersection")
        biggest_area_idx = res_intersection.area.argmax()

        tileID = res_intersection.loc[biggest_area_idx, "Name"]
        this_aoi = res_intersection.loc[biggest_area_idx, "geometry"]

        best_interseciton["tileID"].append(tileID)
        best_interseciton["geometry"].append(this_aoi)

        biggest_intersection = sentinel_tiles.loc[[tileID]]
        rest_aoi = gpd.overlay(rest_aoi, biggest_intersection, how="difference")
        sentinel_tiles = sentinel_tiles.loc[res_intersection["Name"]]

    date_tile_info = gpd.GeoDataFrame(best_interseciton)
    date_tile_info.crs = aoi_file.crs

    return date_tile_info


def _check_folder(tile_folder, file, limit, nodata):
    with rasterio.open(os.path.join(tile_folder, file)) as src:
        # Read in image as a numpy array
        array = src.read(1)
        # Count the occurance of NoData values in np array
        nodata_count = np.count_nonzero(array == nodata)
        # Get a % of NoData pixels
        nodata_percentage = round(nodata_count / array.size * 100, 2)
        if nodata_percentage <= limit:
            return True
        else:
            return False


def check_nodata(loadings, product_type, limit=15.0, nodata=0):
    filtered = dict()

    for tile, folders in loadings.items():
        filtered_folders = set()
        for folder in folders:
            for file in os.listdir(folder):
                if file.endswith(".jp2") and "OPER" not in file:
                    if product_type == "L1C" and limit:
                        if _check_folder(folder, file, limit, nodata):
                            filtered_folders.add(folder)
                            break
                    else:
                        filtered_folders.add(folder)
        filtered[tile] = filtered_folders
    return filtered


def get_min_clouds(
    loadings,
    max_ptc=5,
    cloud_regex=r"\<CLOUDY_PIXEL_PERCENTAGE\>[0-9]*\.?[0-9]*</CLOUDY_PIXEL_PERCENTAGE>",
):
    filtered = dict()
    min_ptc = max_ptc

    for tile, folders in loadings.items():
        filtered_folders = set()
        for folder in folders:
            for file in os.listdir(folder):

                if "MTD_TL.xml" in file:  # MTD_TL.xml

                    with open(os.path.join(folder, file)) as f:
                        ptc = f.read()
                        ptc = re.search(cloud_regex, ptc)

                        if ptc is not None:
                            ptc = "".join(
                                [x for x in ptc.group(0) if x.isdigit() or x == "."]
                            )
                            filtered_folders.add((ptc, folder))

                else:
                    filtered_folders.add(("50", folder))

        filtered[tile] = sorted(filtered_folders)[0][1]

    return filtered


def transform_resolution(data_path, save_path, resolution=(10, 10)):
    with rasterio.open(data_path) as src:
        transform, width, height = rasterio.warp.aligned_target(
            transform=src.meta['transform'], width=src.height,
            height=src.height, resolution=resolution)

        kwargs = src.meta.copy()
        kwargs.update({'transform': transform,
                       'width': width,
                       'height': height})
        with rasterio.open(save_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    resampling=rasterio.warp.Resampling.nearest)
        dst.close()
    src.close()

    return save_path