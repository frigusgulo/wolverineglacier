import glob
import os
import shutil

dir = '/home/dunbar/Research/wolverine/data/cam_weather/images/'

for file in glob.glob(os.path.join(dir,"*")):
	if os.path.isdir(file):
		
		imgs = glob.glob(os.path.join(file,"*.JPG"),recursive=True)
		for image in imgs:
			print(image)
			shutil.move(image,dir)