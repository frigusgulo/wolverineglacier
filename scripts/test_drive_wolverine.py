import pdb
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import glimpse
from glimpse.imports import datetime,np,os
import glob
import itertools
os.environ['OMP_NUM_THREADS'] = '1'
#==============================================

DATA_DIR = '/home/dunbar/Research/wolverine/data/test_images'
DEM_DIR = '/home/dunbar/Research/wolverine/data/dem'
IMG_DIR = 'images'
CAM_DIR = 'images_json'
MAX_DEPTH = 30e3
STATIONS = ('cam_cliff')

# ---- Prepare Observers ----
# ---- Prepare Observers ----

start = datetime.datetime(2017, 12, 30, 00)
end = datetime.datetime(2019, 8, 30, 00)

observers = []
images = []
# some finaglin
pathlist = glob.glob(os.path.join(DATA_DIR,"*.JPG"))
temp = pathlist[0]
del pathlist[0]
pathlist.append(temp)

for path in pathlist:
    image = glimpse.Image(path=path,cam=path.replace(".JPG",".json"))
    print(image.datetime)
    images.append(image)

observer = glimpse.Observer(list(np.array(images)))
print("==========================")
print("Position {}".format(observer.xyz))
#----------------------------
# Prepare DEM
''' 
boxes = [obs.images[0].cam.viewbox(MAX_DEPTH)
    for obs in observers]
box = glimpse.helpers.intersect_boxes(boxes)
'''
paths = glob.glob(os.path.join(DEM_DIR, '*.tif'))
paths.sort()
path = paths[0]
dem = glimpse.Raster.read(path)
print("DEM PATH: {}".format(path))
dem.crop(zlim=(0, np.inf))
dem.fill_crevasses(mask=~np.isnan(dem.Z), fill=True)


# ---- Prepare viewshed ----

dem.fill_circle(observer.xyz, radius=100)
viewshed = dem.copy()
viewshed.Z = np.ones(dem.shape, dtype=bool)
viewshed.Z &= dem.viewshed(observer.xyz)

# ---- Run Tracker ----

xy= np.array((394368,6696220))


time_unit = datetime.timedelta(days=0.5)
motion_model = glimpse.CartesianMotionModel(xy, time_unit=time_unit, dem=dem, dem_sigma=3, n=5000, xy_sigma=(2, 2),vxyz_sigma=(5, 5, 0.2), axyz_sigma=(2, 2, 0.2))

tracker = glimpse.Tracker(observers=observers, viewshed=viewshed)
tracks = tracker.track(motion_models=motion_model, tile_size=(15, 15),parallel=32)

# ---- Plot tracks ----
tracks.plot_xy(start=dict(color='green'), mean=dict(color='red'), sigma=dict(alpha=0.25))
plt.show()

