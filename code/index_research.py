import cv2
import numpy as np
import rasterio
from pathlib import Path
from .download.utils import transform_resolution


def calculate_tcari(b03_path, b04_path, b05_path, out_path=None, nodata=0):

    with rasterio.open(b03_path) as src:
        b03 = src.read(1).astype(rasterio.float32)
        crs = src.crs
        meta = src.meta

    with rasterio.open(b04_path) as src:
        b04 = src.read(1).astype(rasterio.float32)
        
    with rasterio.open(b05_path) as src:
        b05 = src.read(1).astype(rasterio.float32)
        
    if b03.shape != b05.shape:
        transformed_path = transform_resolution(b05_path, b05_path.replace('_B05', '_B05_upscaled'))
        with rasterio.open(transformed_path) as src:
            b05 = src.read(1).astype(rasterio.float32)
            
    tcari = np.where(b05 / b04 == 0,
                    nodata,
                    3 * ( (b05 - b04) - 0.2 * (b05 - b03) ) * (b05 / b04)
                    )

    meta.update(driver="GTiff")
    meta.update(dtype=rasterio.float32)

    if out_path is None:
        out_path = b04_path.replace("_B04", "_TCARI").replace(".jp2", ".tif")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.meta["nodata"] = 0
        dst.meta["max"] = 1
        dst.meta["min"] = 0
        dst.write(tcari.astype(rasterio.float32), 1)

    return out_path


def calculate_msr_g(b04_path, b08_path, out_path=None):

    with rasterio.open(b08_path) as src:
        b08 = src.read(1).astype(rasterio.float32)
        crs = src.crs
        meta = src.meta

    with rasterio.open(b04_path) as src:
        b04 = src.read(1).astype(rasterio.float32)

    msr_g = np.where(
        np.sqrt((b08 / b04) + 1) == 0, 0, ((b08 / b04) - 1) / np.sqrt((b08 / b04) + 1)
    )

    meta.update(driver="GTiff")
    meta.update(dtype=rasterio.float32)

    if out_path is None:
        out_path = b08_path.replace("_B08", "_MSR-G").replace(".jp2", ".tif")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.meta["nodata"] = 0
        dst.meta["max"] = 1
        dst.meta["min"] = 0
        dst.write(msr_g.astype(rasterio.float32), 1)

    return out_path


def calculate_mndwi(b03_path, b11_path, out_path=None):

    with rasterio.open(b11_path) as src:
        b11 = src.read(1).astype(rasterio.float32)
        crs = src.crs
        meta = src.meta

    with rasterio.open(b03_path) as src:
        b03 = src.read(1).astype(rasterio.float32)
        
    b11 = cv2.resize(b11, (b03.shape[-2], b03.shape[-1]), interpolation=cv2.INTER_AREA)

    mndwi = np.where((b11 + b03) == 0, 0, (b11 - b03) / (b11 + b03))

    meta.update(driver="GTiff")
    meta.update(dtype=rasterio.float32)

    if out_path is None:
        out_path = b03_path.replace("_B03", "_MNDWI").replace(".jp2", ".tif")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.meta["nodata"] = 0
        dst.meta["max"] = 1
        dst.meta["min"] = 0
        dst.write(mndwi.astype(rasterio.float32), 1)

    return out_path

def calculate_ndwi(b03_path, b08_path, out_path=None):

    with rasterio.open(b08_path) as src:
        b08 = src.read(1).astype(rasterio.float32)
        crs = src.crs
        meta = src.meta

    with rasterio.open(b03_path) as src:
        b03 = src.read(1).astype(rasterio.float32)

    ndvi = np.where((b08 + b03) == 0, 0, (b08 - b03) / (b08 + b03))

    meta.update(driver="GTiff")
    meta.update(dtype=rasterio.float32)

    if out_path is None:
        out_path = b08_path.replace("_B08", "_NDWI").replace(".jp2", ".tif")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.meta["nodata"] = 0
        dst.meta["max"] = 1
        dst.meta["min"] = 0
        dst.write(ndvi.astype(rasterio.float32), 1)

    return out_path


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

def calculate_ndre(b07_path, b05_path, out_path=None, nodata=0):

    with rasterio.open(b07_path) as src:
        b07 = src.read(1).astype(rasterio.float32)
        crs = src.crs
        meta = src.meta

    with rasterio.open(b05_path) as src:
        b05 = src.read(1).astype(rasterio.float32)

    if b05.shape != b07.shape:
        b05 = cv2.resize(b05, (b07.shape[-2], b07.shape[-1]), interpolation=cv2.INTER_AREA)

    ndvi = np.where((b07 + b05) == 0, nodata, (b07 - b05) / (b07 + b05))

    meta.update(driver="GTiff")
    meta.update(dtype=rasterio.float32)

    if out_path is None:
        out_path = b07_path.replace("_B07", "_NDRE").replace(".jp2", ".tif")

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.meta["nodata"] = 0
        dst.meta["max"] = 1
        dst.meta["min"] = 0
        dst.write(ndvi.astype(rasterio.float32), 1)

    return out_path


def stack_layers(layers, out_path=None):

    arrays = []
    for layer in layers:

        with rasterio.open(layer) as src:
            img = src.read().astype(rasterio.float32)
            crs = src.crs
            meta = src.meta

        for channel in img:
            arrays.append(channel)

    img_size = max([x.shape for x in arrays])
    for i in range(len(arrays)):
        if arrays[i].shape != img_size:
            arrays[i] = cv2.resize(
                arrays[i], (img_size[-2], img_size[-1]), interpolation=cv2.INTER_AREA
            )

    stacked_img = np.stack(arrays, axis=0)

    meta.update(count=len(arrays))
    meta.update(driver="GTiff")
    meta.update(dtype=rasterio.float32)

    if out_path is not None:
        with rasterio.open(out_path, "w", **meta) as dst:
            for i, band in enumerate(stacked_img):
                dst.write(band, i + 1)

    return stacked_img


def combine_bands(band1, band2, op="+", out_path=None, out_shape=None):

    with rasterio.open(band1) as src:
        b1 = src.read(1).astype(rasterio.float32)
        crs = src.crs
        meta = src.meta

    with rasterio.open(band2) as src:
        b2 = src.read(1).astype(rasterio.float32)
    
    if out_shape is None:
        if sum(b1.shape) > sum(b2.shape):
            b2 = cv2.resize(b2, (b1.shape[-2], b1.shape[-1]), interpolation=cv2.INTER_AREA)
        elif sum(b1.shape) < sum(b2.shape):
            b1 = cv2.resize(b1, (b2.shape[-2], b2.shape[-1]), interpolation=cv2.INTER_AREA)
    else:
        b2 = cv2.resize(b2, out_shape, interpolation=cv2.INTER_AREA)
        b1 = cv2.resize(b1, out_shape, interpolation=cv2.INTER_AREA)
        meta['width'] = out_shape[-1]
        meta['height'] = out_shape[-2]
        
    if op == "+":
        calculated_img = b1 + b2
    elif op == "-":
        calculated_img = b1 - b2
    elif op == "/":
        calculated_img = b1 / b2

    meta.update(driver="GTiff")
    meta.update(dtype=rasterio.float32)

    if out_path is not None:
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.meta["nodata"] = 0
            dst.meta["max"] = calculated_img.max()
            dst.meta["min"] = calculated_img.min()
            dst.write(calculated_img.astype(rasterio.float32), 1)

    return calculated_img

