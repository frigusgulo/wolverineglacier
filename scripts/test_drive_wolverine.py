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

DATA_DIR = '/home/dunbar/Research/wolverine/data'
DEM_DIR = os.path.join(DATA_DIR, 'dem')
IMG_DIR = 'images'
CAM_DIR = 'images_json'
MAX_DEPTH = 30e3
STATIONS = ('cam_tounge','cam_weather','cam_cliff')

# ---- Prepare Observers ----
# ---- Prepare Observers ----

start = datetime.datetime(2017, 12, 30, 00)
end = datetime.datetime(2019, 8, 30, 00)

observers = []
for station in STATIONS:
    station_dir = os.path.join(DATA_DIR, station)
    path = os.path.join(station_dir,CAM_DIR,'*.json')
    cam_paths = glob.glob(path,recursive=True)
    cam_paths.sort()
    basenames = [glimpse.helpers.strip_path(path) for path in cam_paths]
    images = [glimpse.Image(
        path=os.path.join(station_dir, IMG_DIR, basename + '.JPG'),
        cam=os.path.join(station_dir, CAM_DIR, basename + '.json'))
        for basename in basenames]
    datetimes = np.array([img.datetime for img in images])
    inrange = np.logical_and(datetimes > start, datetimes < end)
    observers.append(glimpse.Observer(list(np.array(images)[inrange])))
#----------------------------
# Prepare DEM
''' 
boxes = [obs.images[0].cam.viewbox(MAX_DEPTH)
    for obs in observers]
box = glimpse.helpers.intersect_boxes(boxes)
'''
paths = glob.glob(os.path.join(DEM_DIR, '*.tiff'))
paths.sort()
path = paths[0]
dem = glimpse.Raster.read(path)#, xlim=box[0::3], ylim=box[1::3])
print("DEM PATH: {}".format(path))
dem.crop(zlim=(0, np.inf))
dem.fill_crevasses(mask=~np.isnan(dem.Z), fill=True)


observers_ = []
observers_.append(observers[0])
observers = observers_
observers_ = []
# ---- Prepare viewshed ----
for obs in observers:
    dem.fill_circle(obs.xyz, radius=100)
viewshed = dem.copy()
viewshed.Z = np.ones(dem.shape, dtype=bool)
for obs in observers:
    viewshed.Z &= dem.viewshed(obs.xyz)

# ---- Run Tracker ----

xy= np.array((394368,6696220))


time_unit = datetime.timedelta(days=0.5)
motion_models = [glimpse.CartesianMotionModel(
    xyi, time_unit=time_unit, dem=dem, dem_sigma=3, n=5000, xy_sigma=(2, 2),
    vxyz_sigma=(5, 5, 0.2), axyz_sigma=(2, 2, 0.2)) for xyi in xy]
# motion_models = [glimpse.CylindricalMotionModel(
#     xyi, time_unit=time_unit, dem=dem, dem_sigma=3, n=5000, xy_sigma=(2, 2),
#     vrthz_sigma=(np.sqrt(50), np.pi, 0.2), arthz_sigma=(np.sqrt(8), 0.05, 0.2))
#     for xyi in xy]
tracker = glimpse.Tracker(observers=observers, viewshed=viewshed)
tracks = tracker.track(motion_models=motion_models, tile_size=(15, 15),
    parallel=32)

# ---- Plot tracks ----
tracks.plot_xy(start=dict(color='green'), mean=dict(color='red'), sigma=dict(alpha=0.25))
plt.show()

