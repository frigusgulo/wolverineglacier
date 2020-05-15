import glimpse
from glimpse.imports import datetime,np,os
import glob
from PIL import Image
from PIL.ExifTags import TAGS
import numpy as np
import matplotlib.pyplot as plt

#if __name__ == "__main__":

params = dict(
  sensorsz=(35.9, 24),
  xyz=(393506.713, 6695855.64, 961.3370),
  viewdir=(6.55401513e+01, -1.95787507e+01,  7.90589866e+00),
  p=(1.42603100e-01,4.12163305e-01),
  k=(-4.78577912e-01,  1.08772049e+00, -1.27452460e+00,6.78479918e-01,  2.39852156e-02, -1.229e-01),
  imgsz=(7360, 4912),
  fmm=28
)
cliff_camera = glimpse.Camera(**params)



img_path = '/home/dunbar/Research/wolverine/data/cam_cliff/images'
images = glob.glob(os.path.join(img_path,'*.JPG'),recursive=True)
images = [glimpse.Image(path=imagepath,cam=cliff_camera.copy()) for imagepath in images]
images.sort(key= lambda img: img.datetime ) # sort by datetime
images = images[::2]
cliffobserver = glimpse.Observer(images,cache=True)
dem_path = '/home/dunbar/Research/wolverine/data/dem/wolverine_skyline32606.tif'
dem = glimpse.Raster.read(dem_path,d=5)
dem.crop(zlim=(0,np.inf))
dem.fill_crevasses(mask=~np.isnan(dem.Z), fill=True)
dem.fill_circle(cliffobserver.xyz,radius=100)
viewshed = dem.copy()
viewshed.Z = np.ones(dem.shape, dtype=bool)
viewshed.Z &= dem.viewshed(cliffobserver.xyz)
import itertools
xy0= np.array((394368,6696220))
xy = xy0 + np.vstack([xy for xy in
    itertools.product(range(-200, 200, 50), range(-200, 200, 50))])

motion_model = []
time_unit = datetime.timedelta(days=1)

for i in range(xy.shape[0]):
    motion_model.append(glimpse.CartesianMotionModel(
        xy[i,:], time_unit=time_unit, dem=dem, dem_sigma=3, n=5000, xy_sigma=(2, 2),
        vxyz_sigma=(5, 5, 0.2), axyz_sigma=(2, 2, 0.2)))



tracker = glimpse.Tracker(observers=[cliffobserver], viewshed=viewshed)
tracks = tracker.track(motion_models=motion_model, tile_size=(10, 10),parallel=26)
tracks.plot_xy(start=dict(color='green'), mean=dict(color='red'), sigma=dict(alpha=0.25))
'''