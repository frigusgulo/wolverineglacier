import matplotlib
matplotlib.use('agg')
import glimpse
from glimpse.imports import (sys, datetime, matplotlib, np, os)
import glob


# ------------------------------
SNAP = datetime.timedelta(hours=1)
MAXDT = datetime.timedelta(days=1)
MATCH_SEQ = (np.arange(12) + 1)
MAX_RATIO = 0.6
MAX_ERROR = 0.03 # fraction of image width
N_MATCHES = 50
MIN_MATCHES = 50

# ---- Functions ----

def write_matches(matcher, **kwargs):
    matcher.build_matches(
        maxdt=MAXDT, seq=MATCH_SEQ,
        path=MATCHES_PATH, overwrite=False, max_ratio=0.75,
        max_distance=None, parallel=4, weights=True,
        clear_keypoints=True, clear_matches=True, **kwargs)

def read_matches(matcher, **kwargs):
    matcher.build_matches(
        maxdt=MAXDT, seq=MATCH_SEQ,
        path=MATCHES_PATH, overwrite=False, max_ratio=0.75,
        max_distance=None, parallel=True, weights=True,
        clear_keypoints=True, clear_matches=False,
        as_type=glimpse.optimize.RotationMatches,
        filter=dict(
            min_weight=1 / MAX_RATIO, max_error=MAX_ERROR,
            n_best=N_MATCHES, scaled=True),
        **kwargs)

def chext(infile,extension):
    """
    Changes a given file path to an identical file path but with a different extension
    ex: image_01.JPG to image_01.JSON

    Arguments:

        infile (str): Input file path to be changed
        extension (str): output file extension

    Return:
        outfile (str) file with new extension
    """
    if not os.path.isfile(infile):
        raise Exception("File {} Not Found".format(infile))
    token = infile.split(".")[-1] #get old extension
    outfile = infile.split(token)[0] # check for redundent extensions
    outfile = outfile.split("/")[-1]
    outfile += extension
    return outfile

def save_observercams(observer,directory,print_path=False):
    """
    Saves each camera model for the respective Image objects in an Observer object as a .JSON file

    Arguments:
        observer (glimpse.Observer(): Observer object with Images
        directory (path) : Path to the directory used to save camera models
        print_path (bool) : Print the respective path of each camera model saved
    """
    if not os.path.isdir(directory):
        raise Exception("Directory {} Not Found".format(directory))
    for images in observer.images:
        filename = images.path.split("/")[-1]
        path = os.path.join(directory,chext(filename,"JSON"))
        if print_path: print("Path: {}\n".format(path))
        try:
            images.cam.write(path)
        except:
            print("Image {} Has Undefined Camera".format(images.path))
# ----------------------------
root = '~/Research/wolverine/data'
MATCHES_PATH = os.path.join(root,'subdata/matches')
KEYPOINTS_PATH = os.path.join(root,'subdata/keypoints')

STATIONS = ('cam_cliff','cam_tounge')

'''
cliff.camdict =   dict(sensorsz=(35.9,24),xyz=(393506.713,6695855.64, 961.3370), viewdir=(6.55401513e+01, -1.95787507e+01,  7.90589866e+00))
cliff.imgpts = np.array([[5193, 549],[3101, 642],[6153.0, 2297.0]])
cliff.worldpts = np.array([[408245.86,6695847.03,1560 ],[416067.22,6707259.97,988],[394569.509, 6695550.678, 621.075]])
cliff.imagepaths = glob.glob(os.path.join(root, STATIONS[0],'images','*.JPG'),recursive=True)
cliff.images  = [glimpse.Image(path=cliff.imagepaths[0],exif=glimpse.Exif(cliff.imagepaths[0]),cam=cliff.camdict.copy())]
cliff.images[0].anchor=True
cliff.points = glimpse.optimize.Points(cliff.images[0].cam,cliff.imgpts,cliff.worldpts)
[cliff.images.append(glimpse.Image(path=imagepath,cam=cliff.images[0].cam.copy())) for imagepath in cliff.imagepaths[1:]]
cliff.matcher = glimpse.optimize.KeypointMatcher(cliff.images)
cliff.matcher.build_keypoints(contrastThreshold=0.02, overwrite=False,clear_images=True, clear_keypoints=True)
'''
if __name__ == "__main__":

	tounge_camdict =  dict(sensorsz=(35.9,24),xyz=(393797.3785,6694756.62, 767.029), viewdir=(2.85064110e-01,2.54395619e-02, 6.17540651e-03))
	tounge_worldpts = np.array([[393610.609, 6695578.333, 782.287],[393506.713, 6695855.641, 961.337],[393868.946, 6695316.571,644.398]])
	tounge_imgpts = np.array([[479, 2448],[164, 1398],[2813, 3853]])
	tounge_imagepaths = glob.glob(os.path.join("/home/dunbar/Research/wolverine/data/cam_tounge/images",'*.JPG'),recursive=True)
	tounge_images = [glimpse.Image(path=tounge_imagepaths[0],exif=glimpse.Exif(tounge_imagepaths[0]),cam=tounge_camdict.copy())]
	tounge_images[0].anchor=True
	tounge_points = glimpse.optimize.Points(tounge_images[0].cam,tounge_imgpts,tounge_worldpts)


	Cameras = glimpse.optimize.Cameras([tounge_images[0].cam],[tounge_points],dict(viewdir=True,f=True,p=True))
	Cameras.set_cameras(Cameras.fit())

	[tounge_images.append(glimpse.Image(path=imagepath,cam=tounge_images[0].cam.copy())) for imagepath in tounge_imagepaths[1:]]
	tounge_images.sort(key= lambda img: img.datetime ) # sort by datetime
	tounge_matcher = glimpse.optimize.KeypointMatcher(tounge_images)
	tounge_matcher.build_keypoints(contrastThreshold=0.02, overwrite=False,clear_images=True, clear_keypoints=True)
	tounge_matcher.build_matches(
        maxdt=MAXDT, seq=MATCH_SEQ,
        path=MATCHES_PATH, overwrite=True, max_ratio=0.75,
        max_distance=None, parallel=4, weights=True,
        clear_keypoints=True, clear_matches=True)
	tounge_observer = glimpse.Observer(tounge.images)
	tounge_observercameras = glimpse.optimize.ObserverCameras(tounge_observer,matches=tounge_matcher.matches,anchors=tounge.images[0])
	tounge_observercameras.set_cameras(tounge_observercameras.fit())
	save_observercams(tounge_observer,"/home/dunbar/Research/wolverine/data/cam_tounge/images_json")
