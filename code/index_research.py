import cv2
import numpy as np
import rasterio
from pathlib import Path


def calculate_ndvi(b04_path, b08_path, out_path=None, nodata=0):

    with rasterio.open(b08_path) as src:
        b08 = src.read(1).astype(rasterio.float32)
        crs = src.crs
        meta = src.meta

    with rasterio.open(b04_path) as src:
        b04 = src.read(1).astype(rasterio.float32)

    ndvi = np.where((b08 + b04) == 0, nodata, (b08 - b04) / (b08 + b04))

    meta.update(driver="GTiff")
    meta.update(dtype=rasterio.float32)

    if out_path is None:
        out_path = b08_path.replace("_B08", "_NDVI").replace(".jp2", ".tif")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.meta["nodata"] = 0
        dst.meta["max"] = 1
        dst.meta["min"] = 0
        dst.write(ndvi.astype(rasterio.float32), 1)

    return out_path


def calculate_ndmi(b08_path, b12_path, out_path=None, nodata=0):

    with rasterio.open(b08_path) as src:
        b08 = src.read(1).astype(rasterio.float32)
        crs = src.crs
        meta = src.meta

    with rasterio.open(b12_path) as src:
        b12 = src.read(1).astype(rasterio.float32)

    b12 = cv2.resize(b12, (b08.shape[-1], b08.shape[-2]), interpolation=cv2.INTER_AREA)

    ndvi = np.where((b08 + b12) == 0, nodata, (b08 - b12) / (b08 + b12))

    meta.update(driver="GTiff")
    meta.update(dtype=rasterio.float32)

    if out_path is None:
        out_path = b08_path.replace("_B08", "_NDMI").replace(".jp2", ".tif")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.meta["nodata"] = 0
        dst.meta["max"] = 1
        dst.meta["min"] = 0
        dst.write(ndvi.astype(rasterio.float32), 1)

    return out_path
