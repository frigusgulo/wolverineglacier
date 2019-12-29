# Post Estimation and gcp for gcp annotation
from scipy.optimize import leastsq, fmin_l_bfgs_b
import subprocess
import sys
from PIL import Image
from PIL.ExifTags import TAGS
import os
import scipy
import numpy as np
'''
This class takes in the cameras focal length, pose estimate (Easting, Northing, Elevation (m),roll, pitch, yaw) and sensor dimensions upon instantiation.

Pose Estimation: To estimate pose, call "estimate_pose(X,U)", where X is a list of real_world_coordinates and U is their corresponding points on the given image.

** The general purpose of this is too:
		1) Input and Image, camera pose estimate, 


'''
class Camera():
	def __init__(self, image=None, pose=None, bounds=None ,instance = None):
		self.bounds = bounds
		self.p = pose #[3:] # Pose: x, y, z, phi, theta, psi,
		self.image = Image.open(image)
		self.location = pose[:3]
		self.imagew, self.imageh = self.image.size
		self.instance = instance # Keeps track of which station the script is performing on for file writing purposes
    
	def projective_transform(self,X_world):
		imgCords = []
		for X_point in X_world:
			x = np.ravel(X_point[:,0]/X_point[:,2])
			y = np.ravel(X_point[:,1]/X_point[:,2])
			u = x*self.focal_length + self.sensor_x/2 #sensor location
			v = y*self.focal_length + self.sensor_y/2

			imgCords.append((u,v))
		return imgCords
    
	def rotational_transform(self, X, p=None):
		'''Expects non-homogeneous coordinates.'''
		if p is None:
			p = self.p
		p = np.hstack(p)
		if len(X.shape) < 2:
			n = len(X)
			j = int(n/3)
			X = np.reshape(X, (j,3))

		s = np.sin
		c = np.cos

		X_h = np.zeros((X.shape[0], X.shape[1]+1))
		
		X_h[:, :X.shape[1]] = X
		X_h[:, -1] = np.ones((X.shape[0]))
		location = self.location
		orientation = np.rad2deg(p[3:])
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
		#print("******** C",C.shape)
		#print("******** X_h",X_h.shape)
		Xt = C @ X_h.T
		return Xt.T

	def _func(self, x, _p):
		
		X = self.rotational_transform(x,p=_p)
		uv = self.projective_transform(X)
		return uv
    
	def _errfunc(self, p, x, y, func):
		xx = func(x, p)
		temp = y.ravel().astype('float64')
		ss = temp - xx
		return np.sum(ss**2)

	def estimate_pose(self):
		# estimate_pose -> _errfunc -> rotational_transform -> projective_transform
		#err = np.ones(self.world_gcp) The fuck is this for?
		print("Estimating Pose","\n")
		out = fmin_l_bfgs_b(self._errfunc, x0=self.p, args=(self.world_gcp, self.img_gcp, self._func),bounds=self.bounds,approx_grad=True,epsilon=1e-7)
		self.p = out[0]
		print("Pose: " , self.p , "\n")
		

	def extract_metadata(self):
		# extracts meta-data from .jpeg image used for camera pose
		# assigns [sensor dimensions, image dimensions, focal length]
		#get focal length
		self.metaData = {}
		exif_info = self.image._getexif()
		if exif_info:
			print ("Found Meta Data!","\n")
			for (tag, value) in exif_info.items():
				tagname = TAGS.get(tag,tag)
				self.metaData[tagname] = value
			print(self.metaData['FocalLength'])

		#print(self.metaData.keys())    #DEBUG
		self.focal_length = int(self.metaData['FocalLength'][0]/self.metaData['FocalLength'][1])*10  # THIS _getexif() method is a bunch of bullshit. need to find a way to extract meta data from images...
		self.sensor_x = int(self.metaData['XResolution'][0])
		self.sensor_y = int(self.metaData['YResolution'][0])
		print("Meta Data: ", "focal: ", self.focal_length," | sensor (x,y): ", self.sensor_x, self.sensor_y, " image (height,width): ", self.imageh, self.imagew)
		#self.focal_length*= self.sensor_x*self.sensor_y

	def world_to_ims(self,X):
		# For a given ground control point (gcp), this function converts the gcp's real world coordinates to approxiate image coordinates,
		# so as too assist in finding them in the images
		X = X[:,np.newaxis]
	
		#X = np.squeeze(X)
		X = self.rotational_transform(X.T,p=self.p)
	
		uv = self.projective_transform(X)
		uv = np.squeeze(uv)

		return uv

	def choosen_gcp_assign(self,file):
		#takes in a text file of real world coords and their corresponding image coordinates and assigns the class objects (World_gcp, image_gcp)
		print("Processing Handpicked GCP Points","\n")
		worldcords = []
		imgcords = []
		with open(file) as file:
			next(file) #skip description line in text file
			next(file)
			for line in file:	
				line[line=='\t'] == ' '		
				line = line.split()
				world = line[:3]
				img = line[3:]
				#line = line.rstrip('\n').split(" ")
				worldcords.append(world)
				imgcords.append(img)
		file.close()
		self.world_gcp = np.array(worldcords).astype('float64')
		self.img_gcp = np.array(imgcords).astype('float64')

	def gcp_imgcords_predict(self,file):
		'''
		(Assuming the cameras pose has been estimated)
		Given a list of gps measured gcp's, this method will predict their location in the image. 
		This could allow the user to find the gcp in the image and "Ground truth" its image coordinate. 
		'''
		predictions = []
		with open(file,'r') as file:
			next(file) #skip description line in text file
			for line in file:
				if "\t" in line:
					line = line.split("\t")
				else: 
					line = line.split()
				try: 
					line.remove("\n")
					line.remove("")
				except:
					pass
				name = line[0]
				x,y,z = line[1],line[2],line[3]
				X = np.array([x,y,z]).astype('float64')
				#print("gcp_imgcords_predict Debug: ", X,"\n\n")
				uv = self.world_to_ims(X)
				#if (uv[0] <= self.imagew and uv[0] >= 0 ) and (uv[1] <= self.imageh and uv[1] >=0) :
				predictions.append((name, "| World Cords (Easting,Northing,Elev) : ",x,y,z, " | Predicted Image Cords (U,V) "+self.instance + ": ", np.round(uv[0]), np.round(uv[1])))
		file.close()
		with open('imcords_predictions_' + str(self.instance) + '.txt', 'w') as filehandle:
			filehandle.write("Camera Pose (Easting, Northing, Elevation, Roll, Pitch, Yaw : " + str(self.p)+ "\n")
			for listitem in predictions:
				for item in listitem:
					filehandle.write(str(item) + " ")
					
				filehandle.write("\n")
		filehandle.close()
		print( "============ Done ============","\n\n")

if __name__ == '__main__':	
	check_gcp_path = '/home/fdunbar/Research/wolverine/DF_TLCs/tlcameras.txt'
	path = '/home/fdunbar/Research/wolverine'
	#cliff = 'ref_cliff.JPG' #reference images to extract meta-data and predict gcp location
	tounge = 'ref_tounge.JPG'
	#weather = 'ref_wx.JPG'
	#cliff_gcp = 'cliff_cam_gcp.txt' #txt files for camera gcp's
	tounge_gcp = 'tounge_cam_gcp.txt'
	#weather_gcp = 'wx_cam_gcp.txt'
	#cliff_pose = (393506.713,6695855.641,961.337,np.radians(0),np.radians(-5),np.radians(80)) # easting, northing, elevation (m), roll, pitch, yaw
	tounge_pose = (393797.378, 6694756.620, 767.029,np.radians(0),np.radians(-0.5),np.radians(-1)) # easting, northing, elevation (m), roll, pitch, yaw
	tounge_bounds = ((393797.37,393797.37), (6694756.620,6694756.62 ),(767.029,767.029),(np.radians(-5),np.radians(5)),(np.radians(-5),np.radians(5)),(np.radians(-5),np.radians(10)))
	#weather_pose = (392875.681,6696842.618,1403.860,np.radians(0),np.radians(-15),np.radians(105)) # easting, northing, elevation (m), roll, pitch, yaw
	'''
	print("Processing Cliff \n")
	cliff_cam = Camera(image= os.path.join(path,cliff), pose=cliff_pose, instance="cliff_cam")
	cliff_cam.extract_metadata()
	cliff_cam.choosen_gcp_assign(os.path.join(path,cliff_gcp))
	cliff_cam.estimate_pose()
	cliff_cam.gcp_imgcords_predict(check_gcp_path)
	'''
	print("Processing Tounge \n")
	tounge_cam = Camera(image= os.path.join(path,tounge), pose=tounge_pose,bounds= tounge_bounds, instance="Tounge_cam")
	tounge_cam.extract_metadata()
	tounge_cam.choosen_gcp_assign(os.path.join(path,tounge_gcp))
	tounge_cam.estimate_pose()
	tounge_cam.gcp_imgcords_predict(check_gcp_path)
	'''
	print("Processing Weather \n")
	weather_cam = Camera(image= os.path.join(path,weather), pose=weather_pose, instance="Weather_cam")
	weather_cam.extract_metadata()
	weather_cam.choosen_gcp_assign(os.path.join(path,weather_gcp))
	weather_cam.estimate_pose()
	weather_cam.gcp_imgcords_predict(check_gcp_path)
	'''
