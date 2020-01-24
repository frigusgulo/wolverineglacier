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
			self.focal_length = int(self.metaData['FocalLength'][0]/self.metaData['FocalLength'][1])*self.imagew/36
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
		self.worldGCP = np.squeeze(np.array(worldcords).astype('float64'))
		self.imgGCP = np.squeeze(np.array(imgcords).astype('float64'))

	def estimatePose(self):
		print("Estimating Pose for ", str(self.instance),"\n")
		_ ,self.rvec,self.tvec = cv2.solvePnP(self.worldGCP,self.imgGCP,self.cameraMatrix,distCoeffs=None,rvec=self.rvec,tvec=self.tvec,useExtrinsicGuess=1)
		self.R = np.zeros((3,3))
		cv2.Rodrigues(self.rvec,self.R)
		angle = self.R@np.ones((3,1))
		self.R = np.append(self.R,self.tvec,1)
		self.world2img = self.cameraMatrix@self.R

		print("Pose Estimate: ", " Easting,Northing,Elevation ", -1*self.tvec, "\n\n", " Roll, Pitch, Yaw: ", angle)


	def _world2img(self,X):
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
	print("Processing Tounge ")
	cliff_cam = Camera(image= os.path.join(path,tounge), rvec=tounge_pose[:3],tvec=tounge_pose[3:], instance="tounge_cam")
	cliff_cam.extract_metadata()
	cliff_cam.procGCP(os.path.join(path,tounge_gcp))
	cliff_cam.estimatePose()
	cliff_cam.gcp_imgcords_predict(check_gcp_path)

    print("Processing Weather")
	cliff_cam = Camera(image= os.path.join(path,weather), rvec=weather_pose[:3],tvec=weather_pose[3:], instance="weather_cam")
	cliff_cam.extract_metadata()
	cliff_cam.procGCP(os.path.join(path,weather_gcp))
	cliff_cam.estimatePose()
	cliff_cam.gcp_imgcords_predict(check_gcp_path)
'''