import cv2
import numpy as np
import sys
from PIL import Image
from PIL.ExifTags import TAGS
import os
import matplotlib.pyplot as plt
import math

class Camera():
	def __init__(self, image=None, pose=None,worldGCP=None,imgGCP=None):
		self.image = Image.open(image,"r")
		self.imagew, self.imageh = self.image.size
		self.rvec = pose[:3]
		self.tvec = pose[3:]
		self.worldGCP = worldGCP
		self.imgGCP = imgGCP
		self.focal_length = None
		self.cameraMatrix = None
		self.world2img = None

	def extract_metadata(self):
		self.metaData = {}
		exif_info = self.image._getexif()
		if exif_info:
			print("Found Meta Data!","\n")
			for (tag, value) in exif_info.items():
				tagname = TAGS.get(tag,tag)
				self.metaData[tagname] = value
			self.focal_length = (55 / 22.2) * 4272#int(self.metaData['FocalLength'][0]/self.metaData['FocalLength'][1])*self.imagew/36
			self.cameraMatrix = np.array([[self.focal_length,0,self.imageh/2],[0,self.focal_length,self.imagew/2],[0,0,1]])
	
	def rotationMatrixToEulerAngles(self, R) :
	
		sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
		singular = sy < 1e-6
		if  not singular :
			x = math.atan2(R[2,1] , R[2,2])
			y = math.atan2(-R[2,0], sy)
			z = math.atan2(R[1,0], R[0,0])
		else :
			x = math.atan2(-R[1,2], R[1,1])
			y = math.atan2(-R[2,0], sy)
			z = 0
		return np.array([x, y, z])

	def estimatePose(self):
		_ ,self.rvec,self.tvec = cv2.solvePnP(self.worldGCP,self.imgGCP,self.cameraMatrix,distCoeffs=None,rvec=self.rvec,tvec=self.tvec,useExtrinsicGuess=1)
		self.R = np.zeros((3,3))
		cv2.Rodrigues(self.rvec,self.R)
		self.rvec = self.rotationMatrixToEulerAngles(self.R)
		anglez = np.degrees(self.rvec)
		#self.tvec = self.tvec[:,np.newaxis]
		#self.R = np.append(self.R,self.tvec,1)

		print("Pose Estimate: ", " Easting,Northing,Elevation ", -1*self.tvec, "\n\n", " Roll, Pitch, Yaw: ", anglez)

	def rotational_transform(self, X):
		'''Expects non-homogeneous coordinates.'''
		# ASSUMES RADIANS
	
		s = np.sin
		c = np.cos

		X_h = np.zeros((X.shape[0], X.shape[1]+1))
		
		X_h[:, :X.shape[1]] = X
		X_h[:, -1] = np.ones((X.shape[0]))

		location = self.tvec
		orientation = self.rvec
		roll = orientation[0]
		pitch = orientation[1]
		yaw = orientation[2]

		trans = np.mat(([1, 0, 0, -location[0]], [0, 1, 0, -location[1]], 
		                  [0, 0, 1, -location[2]], [0, 0, 0, 1]))
		r_yaw = np.mat(([c(yaw), -s(yaw), 0, 0], [s(yaw), c(yaw), 0, 0], [0, 0, 1, 0]))
		r_pitch = np.mat(([1, 0, 0], [0, c(pitch), s(pitch)], [0, -s(pitch), c(pitch)]))
		r_roll = np.mat(([c(roll), 0, -s(roll)], [0, 1, 0], [s(roll), 0, c(roll)]))
		r_axis = np.mat(([1, 0, 0], [0, 0, -1], [0, 1, 0]))
		C = r_axis @ r_roll @ r_pitch @ r_yaw @ trans
		Xt = C @ X_h.T
		return Xt.T


	def world_to_ims(self,X):
		print(" W 2 image : ", X)
		X = np.reshape(X,(1,3))
		X = np.squeeze(self.rotational_transform(X))
		uv = np.zeros(2)
		print(X)
		x = X[:,0] / X[:,2]
		y = X[:,1] / X[:,2]
		print(x,y)
		u = (x*self.focal_length + (self.imagew/2)) #sensor location
		v = (y*self.focal_length + (self.imageh/2))
		uv[0] = u
		uv[1] = v
		print(uv)
		return uv
	

if __name__ == '__main__':
	coords = np.array([[3917,1574,273012.94,5195299.31,990],
					[2310,1464,273854.87,5195177.47,1233],
					[175,959,274060.24,5195364.16,1343],
					[639,1648,273790.64,5195311.87,1199],
					[1061,2137,273014.38,5195303.62,990]])
	print(coords.shape)
	u_gcp = coords[:, :2].astype('float32')
	X_gcp = coords[:, 2:].astype('float32')
	im = 'IMG_0433.jpeg'
	#def __init__(self, image=None, pose=None):
	pose = np.array([272008, 5133938, 900, 0,np.radians(20),np.radians(120)])
	cam = Camera(image=im,pose=pose,worldGCP=X_gcp,imgGCP=u_gcp)
	cam.extract_metadata()
	cam.estimatePose()
	points = []
	for point in X_gcp:
		#print(point)
		#print(cam.world_to_ims(point))
		points.append(cam.world_to_ims(point))
	
	points = np.array(points)
	#plt.imshow(cam.image)
	#print(u_gcp)
	print(points)
	#plt.plot(u_gcp[:,0],u_gcp[:,1],'r.',ms=10)
	#plt.plot(points[:,0],points[:,1],'b.',ms=10,label='prediction')
	#plt.legend()
	#plt.show()
