import os
import cv2
import numpy as np
import rasterio
import rasterio.mask as riomask

from .utils import draw_pseudocolor_raster


class PlantStress(object):
    def __init__(self, min_ndvi=0.3, noise_z_score=5, anomaly_z_score=1):
        self.min_ndvi = min_ndvi
        self.noise_z_score = noise_z_score
        self.anomaly_z_score = anomaly_z_score

    def _remove_noise(self):
        pass

    def _get_stats(self, array, exclude_nan=True):
        if exclude_nan:
            mu, sigma = np.nanmean(array), np.nanstd(array)
        else:
            mu, sigma = array.mean(), array.std()

        return mu, sigma

    def _get_anomalies_array(self, ndvi_image, z1, z2, border_mask=None):
        img_anom = ndvi_image[0]
        img_anom = np.where(img_anom < z2[0], 1, 0)
        if border_mask is not None:
            img_anom = np.where(border_mask == 1, 0, img_anom)
        raster_mask = np.where(
            (ndvi_image[0] > z1[0]) & (ndvi_image[0] < z1[1]), True, False
        )
        img_anom = np.where(raster_mask, img_anom, 0)

        return img_anom

    def segment_field(
        self, name, field, ndvi_path, start_date, end_date, request_id=10001, idx=0
    ):

        try:
            with rasterio.open(ndvi_path) as src:
                meta = src.meta
                ndvi_image, tfs = riomask.mask(
                    src, field.to_crs(src.crs).geometry, all_touched=False, crop=True
                )
        except Exception:
            with rasterio.open(ndvi_path) as src:
                meta = src.meta
                ndvi_image, tfs = riomask.mask(
                    src, [field], all_touched=False, crop=True
                )

        border, _ = cv2.findContours(
            np.where(ndvi_image[0] == 0, 255, 0).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        border_mask = cv2.drawContours(np.zeros(ndvi_image[0].shape), border, 0, 1, 2)

        ndvi_image = np.where(ndvi_image >= self.min_ndvi, ndvi_image, np.nan)
        mu, sigma = self._get_stats(ndvi_image)

        if mu < self.min_ndvi:
            raise SystemExit("No vegetation is observed")

        z1 = (
            mu - self.noise_z_score * sigma,
            mu + self.noise_z_score * sigma,
        )  # get rid of noise
        pure_ndvi = ndvi_image[0]
        pure_ndvi = pure_ndvi[pure_ndvi >= self.min_ndvi]
        pure_ndvi = pure_ndvi[(pure_ndvi >= z1[0]) & (pure_ndvi <= z1[1])]
        pure_mu, pure_sigma = self._get_stats(pure_ndvi, exclude_nan=False)
        z2 = (
            pure_mu - self.anomaly_z_score * pure_sigma,
            pure_mu + self.anomaly_z_score * pure_sigma,
        )  # find anomalies in "noise-free" image

        img_anom = self._get_anomalies_array(ndvi_image, z1, z2, border_mask)

        raster_path = ndvi_path.replace("_ndvi.tif", f"_field_{request_id}_{idx}.tif")
        assert raster_path != ndvi_path

        colors = {"Normal Growth": (0, 0, 0), "Anomaly": (182, 10, 28)}

        draw_pseudocolor_raster(
            image=img_anom,
            colors=colors,
            meta=meta,
            meta_name=name,
            out_raster_path=raster_path,
            start_date=start_date,
            end_date=end_date,
            request_id=request_id,
            annotated=False,
            transform=tfs
        )

        return raster_path

