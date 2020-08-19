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

DATA_DIR = '/home/dunbar/Research/wolverine/data/'
DEM_DIR = os.path.join(DATA_DIR, 'dem')
IMG_DIR = 'images'
CAM_DIR = 'images_json'
MAX_DEPTH = 30e3
STATIONS = ('cam_cliff')

# ---- Prepare Observers ----

observers = []

campaths=  glob.glob(os.path.join("/home/dunbar/Research/wolverine/data/cam_cliff/images","*.JSON"))
images = [glimpse.Image(path=campath.replace(".JSON",".JPG"),cam=campath) for campath in campaths]
images.sort(key= lambda img: img.datetime)
observers.append(glimpse.Observer(list(np.array(images))))
#--------------------------


#----------------------------
# Prepare DEM 
path = "/home/dunbar/Research/wolverine/data/dem/wolverine_skyline32606.tif"
dem = glimpse.Raster.read(path,d=5)
print("DEM PATH: {}".format(path))
dem.crop(zlim=(0, np.inf))
dem.fill_crevasses(mask=~np.isnan(dem.Z), fill=True)



# ---- Prepare viewshed ----

for obs in observers:
    dem.fill_circle(obs.xyz, radius=100)
viewshed = dem.copy()
viewshed.Z = np.ones(dem.shape, dtype=bool)
for obs in observers:
    viewshed.Z &= dem.viewshed(obs.xyz)

print("\n *****Viewshed Done**** \n")
# ---- Run Tracker ----
xy = []
xy0 = np.array((394368,6696220))
xy.append(xy0)
#xy = xy0 + np.vstack([xy for xy in
  # itertools.product(range(-500, 500, 50), range(-500, 500, 50))])
time_unit = datetime.timedelta(days=0.5)
motion_models = [glimpse.CartesianMotionModel(
    xyi, time_unit=time_unit, dem=dem, dem_sigma=3, n=5000, xy_sigma=(2, 2),
    vxyz_sigma=(5, 5, 0.2), axyz_sigma=(2, 2, 0.2)) for xyi in xy]
# motion_models = [glimpse.CylindricalMotionModel(
#     xyi, time_unit=time_unit, dem=dem, dem_sigma=3, n=5000, xy_sigma=(2, 2),
#     vrthz_sigma=(np.sqrt(50), np.pi, 0.2), arthz_sigma=(np.sqrt(8), 0.05, 0.2))
#     for xyi in xy]
tracker = glimpse.Tracker(observers=observers, viewshed=viewshed)
print("\n****Tracking Now*****\n")
tracks = tracker.track(motion_models=motion_models, tile_size=(15, 15),
    parallel=2**5)

# ---- Plot tracks ----
tracks.plot_xy(start=dict(color='green'), mean=dict(color='red'), sigma=dict(alpha=0.25))
plt.show()
