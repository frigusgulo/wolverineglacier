import matplotlib
matplotlib.use('agg')
import glimpse
from glimpse.imports import (sys, datetime, matplotlib, np, os)
import glob


# ------------------------------
SNAP = datetime.timedelta(hours=1)
MAXDT = datetime.timedelta(days=0.5)
MATCH_SEQ = (np.arange(4) + 1)
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
TOUNGE_MATCHES_PATH = os.path.join(root,'subdata/matches/tounge.pkl')
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
    base_model = '/home/dunbar/Research/wolverine/wolverineglacier/scripts/intrinsicmodel.json'
    base_cam = glimpse.Camera.read(base_model)




    cliff_camdict =   dict(sensorsz=(35.9,24),xyz=(393506.713,6695855.64, 961.3370), viewdir=(6.55401513e+01, -1.95787507e+01,  7.90589866e+00))
    cliff_camdict = glimpse.helpers.merge_dicts(base_cam.as_dict(),cliff_camdict)
    cliff_imgpts = np.array([[5193, 549],[3101, 642],[6153.0, 2297.0]])
    cliff_worldpts = np.array([[408245.86,6695847.03,1560 ],[416067.22,6707259.97,988],[394569.509, 6695550.678, 621.075]])
    cliff_imagepaths = glob.glob(os.path.join("/home/dunbar/Research/wolverine/data/cam_cliff/images",'*.JPG'),recursive=True)
    cliff_images  = [glimpse.Image(path=cliff_imagepaths[0],exif=glimpse.Exif(cliff_imagepaths[0]),cam=cliff_camdict.copy())]
    cliff_images[0].anchor=True
    cliff_points = glimpse.optimize.Points(cliff_images[0].cam,cliff_imgpts,cliff_worldpts)

    #cliff_matcher = glimpse.optimize.KeypointMatcher(cliff.images)


    Cameras = glimpse.optimize.Cameras(cams=[cliff_images[0].cam],controls=[cliff_points],cam_params=dict(viewdir=True,f=True,p=True,c=True))
    Cameras.set_cameras(Cameras.fit())

    #[tounge_images.append(glimpse.Image(path=imagepath,cam=tounge_images[0].cam.copy())) for imagepath in tounge_imagepaths[1:]]
    #tounge_images.sort(key= lambda img: img.datetime ) # sort by datetime
    #tounge_matcher = glimpse.optimize.KeypointMatcher(tounge_images)
    [cliff_images.append(glimpse.Image(path=imagepath,cam=cliff_images[0].cam.copy())) for imagepath in cliff_imagepaths[1:]]
    cliff_images.sort(key= lambda img: img.datetime ) # sort by datetime

    split = int(len(cliff_images)/3)
    cliff_imagesA = cliff_images[:split]
    cliff_imagesB = cliff_images[split:2*split]
    cliff_imagesC = cliff_images[2*split:]

##########################################################
    cliff_matcher = glimpse.optimize.KeypointMatcher(cliff_imagesA)
    print("\n Building Keypoints \n")
    cliff_matcher.build_keypoints(
    	clear_images=True,
        clear_keypoints=True)
    print("\nBuilding Matches\n")
    cliff_matcher.build_matches(
        maxdt=MAXDT, seq=MATCH_SEQ,
        path=TOUNGE_MATCHES_PATH, max_ratio=0.75,
        max_distance=None, parallel=4, weights=True)
    #tounge_matches = np.load(TOUNGE_MATCHES_PATH)
    #group_indices = {key: 0 for _, key in enumerate(np.arange(len(tounge_images)).tolist()) }
    cam_params = [dict() if img.anchor else dict(viewdir=True) for img in cliff_imagesA]
    #group_params = [dict(c=True,p=True) for img in tounge_images]
    Cameras = glimpse.optimize.Cameras(
    	cams = [image.cam for image in cliff_imagesA], 
    	controls=list(cliff_matcher.matches.data),
    	cam_params=cam_params
    	#group_params = group_params
    	)
    print("\nFitting Cameras\n")
    fit = Cameras.fit(ftol=1, full=True, loss='soft_l1')
    Cameras.set_cameras(fit.params)

    CamerasA = Cameras.cams

    Cameras=None
    cliff_matcher = None
##########################################################################
    cliff_matcher = glimpse.optimize.KeypointMatcher(cliff_imagesB)
    print("\n Building Keypoints \n")
    cliff_matcher.build_keypoints(
        clear_images=True,
        clear_keypoints=True)
    print("\nBuilding Matches\n")
    cliff_matcher.build_matches(
        maxdt=MAXDT, seq=MATCH_SEQ,
        path=TOUNGE_MATCHES_PATH, max_ratio=0.75,
        max_distance=None, parallel=4, weights=True)
    #tounge_matches = np.load(TOUNGE_MATCHES_PATH)
    #group_indices = {key: 0 for _, key in enumerate(np.arange(len(tounge_images)).tolist()) }
    cam_params = [dict() if img.anchor else dict(viewdir=True) for img in cliff_imagesB]
    #group_params = [dict(c=True,p=True) for img in tounge_images]
    Cameras = glimpse.optimize.Cameras(
        cams = [image.cam for image in cliff_imagesB], 
        controls=list(cliff_matcher.matches.data),
        cam_params=cam_params
        #group_params = group_params
        )
    print("\nFitting Cameras\n")
    fit = Cameras.fit(ftol=1, full=True, loss='soft_l1')
    Cameras.set_cameras(fit.params)
    CamerasB = Cameras.cams

    Cameras=None
    cliff_matcher = None
####################################################################
    cliff_matcher = glimpse.optimize.KeypointMatcher(cliff_imagesC)
    print("\n Building Keypoints \n")
    cliff_matcher.build_keypoints(
        clear_images=True,
        clear_keypoints=True)
    print("\nBuilding Matches\n")
    cliff_matcher.build_matches(
        maxdt=MAXDT, seq=MATCH_SEQ,
        path=TOUNGE_MATCHES_PATH, max_ratio=0.75,
        max_distance=None, parallel=4, weights=True)
    #tounge_matches = np.load(TOUNGE_MATCHES_PATH)
    #group_indices = {key: 0 for _, key in enumerate(np.arange(len(tounge_images)).tolist()) }
    cam_params = [dict() if img.anchor else dict(viewdir=True) for img in cliff_imagesC]
    #group_params = [dict(c=True,p=True) for img in tounge_images]
    Cameras = glimpse.optimize.Cameras(
        cams = [image.cam for image in cliff_imagesC], 
        controls=list(cliff_matcher.matches.data),
        cam_params=cam_params
        #group_params = group_params
        )
    print("\nFitting Cameras\n")
    fit = Cameras.fit(ftol=1, full=True, loss='soft_l1')
    Cameras.set_cameras(fit.params)
    CamerasC = Cameras.cams

    Cameras=None
    cliff_matcher = None
    Cameras_end = CamerasA + CamerasB + CamerasC


    directory = "/home/dunbar/Research/wolverine/data/cam_cliff/images_json/"
    for images,newcam in  zip(cliff_images,Cameras_end):
    	filename = images.path
    	path = os.path.join(directory,chext(filename,"JSON"))
    	print("\n",path,"\n")
    	newcam.write(path=path,attributes=("viewdir","xyz","sensorsz","fmm","cmm","p","k","imgsz","f"))
