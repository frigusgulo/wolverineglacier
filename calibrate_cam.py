import numpy as np
import cv2 as cv
import glob
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
dimx = 7
dimy = 6
objp = np.zeros((dimx*dimy,3), np.float32)
objp[:,:2] = np.mgrid[0:dimx,0:dimy].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('/home/dunbar/Research/wolverine/104_0101/*.JPG')
print("Found ",len(images)," Images")
for fname in images:
    print(fname,"\n")
    img = cv.imread(fname)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (dimx,dimy))
    # If found, add object points, image points (after refining them)
    print("Corner status: ",ret)
    if ret == True:
        print("Found Corners")
        objpoints.append(objp)
        #corners2 = cv.cornerSubPix(img,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (gray.shape[1],gray.shape[0]), None, None)
print(ret)
print("Camera Matrix: ",mtx)
print("Distortion Coefficients: ",dist)
print(rvecs)
print(tvecs)