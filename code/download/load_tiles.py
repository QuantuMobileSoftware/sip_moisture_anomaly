from pathlib import Path
from sentinel2download.downloader import Sentinel2Downloader

CONSTRAINTS = {'NODATA_PIXEL_PERCENTAGE': 0.0, 'CLOUDY_PIXEL_PERCENTAGE': 0.0, }

def load_images(api_key, tiles, start_date, end_date, output_dir, bands, constrains=None, product_type="L2A"):
    loader = Sentinel2Downloader(api_key)
    loadings = dict()
    for tile in tiles:

        if constrains is None:
            constrains = CONSTRAINTS
        loaded = loader.download(
            product_type,
            [tile],
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            bands=bands,
            constraints=constrains,
        )

        loadings[tile] = loaded

    tile_folders = dict()
    for tile, tile_paths in loadings.items():
        tile_folders[tile] = {
            str(Path(tile_path[0]).parent) for tile_path in tile_paths
        }
    return tile_folders
