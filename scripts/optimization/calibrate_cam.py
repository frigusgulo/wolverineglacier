import numpy as np
import cv2 as cv
import glob
import glimpse
import  glimpse.convert as gc
from glimpse.helpers import merge_dicts
import matplotlib.pyplot as plt
import sys
import os
def parse_camera_matrix(x):
    """
    Return fx, fy, cx, and cy from camera matrix.
    Arguments:
        x (array-like): Camera matrix [[fx 0 cx], [0 fy cy], [0 0 1]]
    
    Returns:
        dict: fx, fy, cx, and cy
    """
    x = np.asarray(x)
    return {'fx': x[0, 0], 'fy': x[1, 1], 'cx': x[0, 2], 'cy': x[1, 2]}


def parse_distortion_coefficients(x):

    x = np.asarray(x)
    labels = ('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6')
    return {key: x[i] if i < len(x) else 0 for i, key in enumerate(labels)}

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
dimx = 7
dimy = 6
edgesizeMeters = 1
objp = np.zeros((dimx*dimy,3), np.float32)
objp[:,:2] = np.mgrid[0:dimx,0:dimy].T.reshape(-1,2)*edgesizeMeters
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
dir_ = sys.argv[1]
images = glob.glob(os.path.join(dir_,'*.JPG'))
print("Found ",len(images)," Images")
print("Dims: {} {}".format(dimx,dimy))
for fname in images:
    print(fname,"\n")
    img = cv.imread(fname)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (dimx,dimy))
    # If found, add object points, image points (after refining them)
    print("Corner status: ",ret)
    keep = False
    if ret == True:
        objpoints.append(objp)
        corners = np.squeeze(corners)
        cv.cornerSubPix(gray,corners, (3,3), (-1,-1), criteria)
        imgpoints.append(corners)
        
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("Camera Matrix: ",mtx)
print("Distortion Coefficients: ",dist)
cam = glimpse.Image(images[0],exif=glimpse.Exif(images[0]),cam=dict(sensorsz=(35.9,24))).cam
cam_matdict = parse_camera_matrix(mtx)
cam_distcoefs = parse_distortion_coefficients(np.squeeze(dist))
cammodel = merge_dicts(cam_matdict,cam_distcoefs)
cammodel = merge_dicts(cam.as_dict(),cammodel)
cammodel["f"][0] = cammodel["fx"]
cammodel["f"][1] = cammodel["fy"]
cammodel.pop("fx")
cammodel.pop("fy")
cammodel["c"][0] = cammodel["cx"] 
cammodel["c"][1] = cammodel["cy"] 
cammodel["p"][0] = cammodel["p1"]
cammodel["p"][1] = cammodel["p2"]
cammodel["k"][0] = cammodel["k1"]
cammodel["k"][1] = cammodel["k2"] 
cammodel["k"][2] = cammodel["k3"]
cammodel.pop("cy")
cammodel.pop("cx")
cammodel.pop("p1")
cammodel.pop("p2")
cammodel.pop("k1")
cammodel.pop("k2")
cammodel.pop("k3")
cammodel.pop("k4")
cammodel.pop("k5")
cammodel.pop("k6")
camera = glimpse.Camera(**cammodel)
camera.write("intrinsicmodel.json")

