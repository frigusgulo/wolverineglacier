import glimpse
import glob
import numpy as np
from glimpse.imports import datetime,np,os

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

if __name__ == '__main__':

    cliff_img = '/home/dunbar/Research/wolverine/data/cam_cliff/images/cam_cliff_0493.JPG'
    tounge_img = '/home/dunbar/Research/wolverine/data/cam_tounge/images/cam_tounge_0521.JPG'
    #weather_img = '/home/dunbar/Research/wolverine/data/cam_weather/images/cam_weather_0056.JPG'

    cliff_cam =   dict(sensorsz=(35.9,24),xyz=(393506.713,6695855.64, 961.3370), viewdir=(6.55401513e+01, -1.95787507e+01,  7.90589866e+00))
    tounge_cam =  dict(sensorsz=(35.9,24),xyz=(393797.3785,6694756.62, 767.029), viewdir=(2.85064110e-01,2.54395619e-02, 6.17540651e-03))
    #wx_cam =      dict(sensorsz=(35.9,24),xyz=(392875.681,6696842.62,  1403.860),viewdir=(6.98964088e-02, -1.63792403e-01,  1.84747297e+00))

    cliff = glimpse.Image(cliff_img,exif=glimpse.Exif(cliff_img),cam=cliff_cam)
    tounge = glimpse.Image(tounge_img,exif=glimpse.Exif(tounge_img),cam=tounge_cam)
    #weather = glimpse.Image(weather_img,exif=glimpse.Exif(weather_img),cam=wx_cam)

    cliff_imgpts = np.array([[5193, 549],[3101, 642],[6153.0, 2297.0]])
    cliff_worldpts = np.array([[408245.86,6695847.03,1560 ],[416067.22,6707259.97,988],[394569.509, 6695550.678, 621.075]])
    tounge_worldpts = np.array([[393610.609, 6695578.333, 782.287],[393506.713, 6695855.641, 961.337],[393868.946, 6695316.571,644.398]])
    tounge_imgpts = np.array([[479, 2448],[164, 1398],[2813, 3853]])
    #weather_worldpts = np.array([[408230.99, 6695826.46, 1576.73],[403888.03, 6690411.89,1675],[394097.434, 6695625.150, 683.678]])
    #weather_imgpts = np.array([[2361, 1596],[5088,1288],[6750.0, 3889.0]])
    cliff_points = glimpse.optimize.Points(cliff.cam,cliff_imgpts,cliff_worldpts)
    tounge_points = glimpse.optimize.Points(tounge.cam,tounge_imgpts,tounge_worldpts)
    #weather_points = glimpse.optimize.Points(weather.cam,weather_imgpts,weather_worldpts)

    cliffparams = glimpse.optimize.Cameras([cliff.cam],[cliff_points],dict(viewdir=True,p=True,k=True,f=True))
    cliffparams = cliffparams.fit()
    toungeparams = glimpse.optimize.Cameras([tounge.cam],[tounge_points],dict(viewdir=True,p=True,k=True,f=True))
    toungeparams = toungeparams.fit()

    cliff_optdict = dict(viewdir=cliffparams[:3],p=cliffparams[3:5],k=cliffparams[5:11],f=cliffparams[11:])
    tounge_optdict = dict(viewdir=toungeparams[:3],p=toungeparams[3:5],k=toungeparams[5:],f=cliffparams[11:])
    cliff_cam = cliff.cam.as_dict(attributes=("xyz","viewdir","fmm","cmm","sensorsz","imgsz","f","c","k","p"))
    tounge_cam = tounge.cam.as_dict(attributes=("xyz","viewdir","fmm","cmm","sensorsz","imgsz","f","c","k","p"))
    cliff_cam = glimpse.helpers.merge_dicts(cliff_cam,cliff_optdict)
    tounge_cam = glimpse.helpers.merge_dicts(tounge_cam,tounge_optdict)

    cliff_kyp = "/home/dunbar/Research/wolverine/wolverine/subdata/cliff"
    tounge_kyp = "/home/dunbar/Research/wolverine/wolverine/subdata/tounge"

    time_unit = datetime.timedelta(days=1)

    cliff_imgs = '/home/dunbar/Research/wolverine/data/cam_cliff/images'
    cliffimages = glob.glob(os.path.join(cliff_imgs,'*.JPG'),recursive=True)
    cliffimages = [glimpse.Image(path=imagepath,cam=cliff_cam.copy(),keypoints_path=os.path.join(cliff_kyp,chext(imagepath,"JSON") )) for imagepath in cliffimages]
    cliffimages.sort(key= lambda img: img.datetime ) # sort by datetime
    cliff_viewopt = glimpse.optimize.KeypointMatcher(cliffimages)
    print("\nBuilding Cliff Keypoints\n")
    cliff_viewopt.build_keypoints(clear_images=True,overwrite=True,clear_keypoints=True)
    cliff_viewopt.build_matches(maxdt=time_unit,overwrite=True,clear_matches=True,clear_keypoints=True,path=cliff_kyp)
    cliffObserver = glimpse.Observer(cliffimages)
    cliffOpt = glimpse.optimize.ObserverCameras(cliffObserver,matches=cliff_viewopt).fit()
    save_observercams(cliffObserver,"/home/dunbar/Research/wolverine/data/cam_cliff/images_json")


    tounge_imgs = '/home/dunbar/Research/wolverine/data/cam_tounge/images'
    toungeimages = glob.glob(os.path.join(tounge_imgs,'*.JPG'),recursive=True)
    toungeimages = [glimpse.Image(path=imagepath,cam=tounge_cam.copy(),keypoints_path=os.path.join(tounge_kyp,chext(imagepath,"JSON"))) for imagepath in toungeimages]
    toungeimages.sort(key= lambda img: img.datetime ) # sort by datetime
    tounge_viewopt = glimpse.optimize.KeypointMatcher(toungeimages)
    tounge_viewopt.build_keypoints(clear_images=True,overwrite=True,clear_keypoints=True)
    tounge_viewopt.build_matches(maxdt=time_unit,overwrite=True,clear_matches=True,clear_keypoints=True,path=tounge_kyp)
    print("\nBuilding Tounge Keypoints\n")
    toungeObserver = glimpse.Observer(toungeimages)
    toungeOpt = glimpse.optimize.ObserverCameras(toungeObserver,matches=tounge_viewopt).fit()
    save_observercams(toungeObserver,"/home/dunbar/Research/wolverine/data/cam_tounge/images_json")
