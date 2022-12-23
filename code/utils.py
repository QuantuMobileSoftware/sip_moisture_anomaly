import os
import geojson
import rasterio
from rasterio.merge import merge
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
)


def transform_crs(data_path, save_path, dst_crs="EPSG:4326", resolution=(10, 10)):
    with rasterio.open(data_path) as src:
        if resolution is None:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
        else:
            transform, width, height = calculate_default_transform(
                src.crs,
                dst_crs,
                src.width,
                src.height,
                *src.bounds,
                resolution=resolution,
            )
        kwargs = src.meta.copy()
        kwargs.update(
            {"crs": dst_crs, "transform": transform, "width": width, "height": height}
        )
        with rasterio.open(save_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                )

    return save_path


def stitch_tiles(paths, out_raster_path):
    if not isinstance(paths[0], str):
        paths = [str(x) for x in paths]
    tiles = []
    tmp_files = []
    
    crs = None
    meta = None
    for i, path in enumerate(paths):
        if i == 0:
            file = rasterio.open(path)
            meta, crs = file.meta, file.crs
        else:
            tmp_path = path.replace(
                '.jp2', '_tmp.jp2').replace('.tif', '_tmp.tif')
            crs_transformed = transform_crs(path, tmp_path, 
                                            dst_crs=crs, 
                                            resolution=None)
            tmp_files.append(crs_transformed)
            file = rasterio.open(crs_transformed)
        tiles.append(file)
            
    tile_arr, transform = merge(tiles, method='last')
    
    meta.update({"driver": "GTiff",
                 "height": tile_arr.shape[1],
                 "width": tile_arr.shape[2],
                 "transform": transform,
                 "crs": crs})
    
    if '.jp2' in out_raster_path:
        out_raster_path = out_raster_path.replace('.jp2', '_merged.tif')
    else:
        out_raster_path = out_raster_path.replace('.tif', '_merged.tif')
    print(f'saved raster {out_raster_path}')

    for tile in tiles:
        tile.close()
        
    for tmp_file in tmp_files:
        try:
            os.remove(tmp_file)
        except FileNotFoundError:
            print(f'Tile {tmp_file} was removed or renamed, skipping')
        
    with rasterio.open(out_raster_path, "w", **meta) as dst:
        dst.write(tile_arr)
    
    return out_raster_path


def dump_no_data_geosjon(polygon, geojson_path, metadata):
    label = 'No data'
    style = dict(color='red')
    feature = geojson.Feature(geometry=polygon, properties=dict(label=label, style=style))
    feature['start_date'] = metadata["START_DATE"]
    feature['end_date'] = metadata["END_DATE"]
    feature['request_id'] = metadata["REQUEST_ID"]
    feature['name'] = f'{metadata["NAME"]}\nNo data available'

    with open(geojson_path, 'w') as f:
        geojson.dump(feature, f)
