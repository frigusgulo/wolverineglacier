import fiona
import rasterio
import rasterio.mask
import sys
import os

shapefile = sys.argv[1]
with fiona.open(shapefile, "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]

with rasterio.open(sys.argv[2]) as src:
    out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
    out_meta = src.meta

with rasterio.open(sys.argv[3], "w", **out_meta) as dest:
    dest.write(out_image)
