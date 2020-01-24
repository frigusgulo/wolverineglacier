import cv2
import numpy as np
import sys
from PIL import Image
from PIL.ExifTags import TAGS
import os

class Camera():
	def __init__(self, image=None, rvec=None,tvec=None, instance=None):
		self.image = Image.open(image)
		self.imagew, self.imageh = self.image.size
		self.instance = instance
		self.rvec = rvec
		self.tvec = tvec
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
			self.focal_length = int(self.metaData['FocalLength'][0]/self.metaData['FocalLength'][1])
			self.cameraMatrix = np.array([[self.focal_length,0,self.imagew/2],[0,self.focal_length,self.imageh/2],[0,0,1]])

	def procGCP(self,_file):
		worldcords = []
		imgcords = []
		with open(_file) as file:
			print("Processing Handpicked GCP Points from ",_file,"\n")
			next(file)
			next(file)
			for line in file:
				line[line=='\t'] == ' '
				line = line.split()
				world = line[:3]
				img = line[3:]
				worldcords.append(world)
				imgcords.append(img)
				print("World: ",world, " Image: ",img,"\n")
		file.close()
		self.worldGCP = np.array(worldcords).astype('float64')
		self.imgGCP = np.array(imgcords).astype('float64')

	def estimatePose(self):
		print("Estimating Pose for ", str(self.instance),"\n")
		self.rvel ,self.rvec,self.tvec = cv2.solvePnP(self.worldGCP,self.imgGCP,self.cameraMatrix,distCoeffs=None,rvec=self.rvec,tvec=self.tvec,useExtrinsicGuess=1)
		print(self.rvel)
		print("Pose Estimate: ", " Easting,Northing,Elevation ", self.tvec, " Roll, Pitch, Yaw ", self.rvec)


	def transMat(self):
		if self.cameraMatrix is not None:
			'''
			s = np.sin
			c = np.cos
			location = self.tvec
			roll = np.radians(self.rvec[0])
			pitch = np.radians(self.rvec[1])
			yaw = np.radians(self.rvec[2])
			r_yaw = np.mat(([c(yaw), -s(yaw), 0], [s(yaw), c(yaw), 0], [0, 0, 1])).astype('float64')
			r_pitch = np.mat(([1, 0, 0], [0, c(pitch), s(pitch)], [0, -s(pitch), c(pitch)])).astype('float64')
			r_roll = np.mat(([c(roll), 0, -s(roll)], [0, 1, 0], [s(roll), 0, c(roll)])).astype('float64')
			r_axis = np.mat(([1, 0, 0], [0, 0, -1], [0, 1, 0])).astype('float64')
			R =  r_axis@r_roll@r_pitch@r_yaw 
			'''
			R = np.zeros((3,3))
			cv2.Rodrigues(self.tvec,R)
			R = np.append(R,self.tvec,1)
			self.world2img = self.cameraMatrix@R

	def _world2img(self,X):
		if self.world2img is None:
		 	self.transMat()

		uv = self.world2img@X
		return uv



	def gcp_imgcords_predict(self,file):
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
				X = np.array([x,y,z,1]).astype('float64')
				#print("gcp_imgcords_predict Debug: ", X,"\n\n")
				uv = self._world2img(X)
				print(uv)
				#if (uv[0] <= self.imagew and uv[0] >= 0 ) and (uv[1] <= self.imageh and uv[1] >=0) :
				predictions.append((name, "| World Cords (Easting,Northing,Elev) : ",x,y,z, " | Predicted Image Cords (U,V) "+self.instance + ": ", np.round(uv[0]), np.round(uv[1])))
		file.close()
		with open('imcords_predictions_' + str(self.instance) + '.txt', 'w') as filehandle:
			filehandle.write("Camera Pose (Easting, Northing, Elevation, Roll, Pitch, Yaw : " + str(self.tvec) + "\n")
			for listitem in predictions:
				for item in listitem:
					filehandle.write(str(item) + " ")
					
				filehandle.write("\n")
		filehandle.close()
		print( "============ Done ============","\n\n")

if __name__ == '__main__':	
	check_gcp_path = '/home/fdunbar/Research/wolverine/wolverineglacier/DF_TLCs/tlcameras.txt'
	path = '/home/fdunbar/Research/wolverine/'
	cliff = 'ref_cliff.JPG' #reference images to extract meta-data and predict gcp location
	tounge = 'ref_tounge.JPG'
	weather = 'ref_wx.JPG'
	cliff_gcp = 'wolverineglacier/cliff_cam_gcp.txt' #txt files for camera gcp's
	tounge_gcp = 'wolverineglacier/tounge_cam_gcp.txt'
	weather_gcp = 'wolverineglacier/wx_cam_gcp.txt'
	cliff_pose = (393506.713,6695855.641,961.337,0,-5,80) # easting, northing, elevation (m), roll, pitch, yaw
	tounge_pose = (393797.378, 6694756.620, 767.029,0,0,0) # easting, northing, elevation (m), roll, pitch, yaw
	weather_pose = (392875.681,6696842.618,1403.860,0,15,100) # easting, northing, elevation (m), roll, pitch, yaw
	
	print("Processing Cliff \n")
	cliff_cam = Camera(image= os.path.join(path,cliff), rvec=cliff_pose[:3],tvec=cliff_pose[3:], instance="cliff_cam")
	cliff_cam.extract_metadata()
	cliff_cam.procGCP(os.path.join(path,cliff_gcp))
	cliff_cam.estimatePose()
	cliff_cam.gcp_imgcords_predict(check_gcp_path)
	
	'''
	print("Processing Tounge \n")
	tounge_cam = Camera(image= os.path.join(path,tounge), pose=tounge_pose,bounds= tounge_bounds, instance="Tounge_cam")
	tounge_cam.extract_metadata()
	tounge_cam.choosen_gcp_assign(os.path.join(path,tounge_gcp))
	tounge_cam.estimate_pose()
	tounge_cam.gcp_imgcords_predict(check_gcp_path)
	
	print("Processing Weather \n")
	weather_cam = Camera(image= os.path.join(path,weather), pose=weather_pose, instance="Weather_cam")
	weather_cam.extract_metadata()
	weather_cam.choosen_gcp_assign(os.path.join(path,weather_gcp))
	weather_cam.estimate_pose()
	weather_cam.gcp_imgcords_predict(check_gcp_path)


    '''        
