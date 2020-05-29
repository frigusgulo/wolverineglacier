
import cv2
import glob
import matplotlib.pyplot as plt
import os

dir = "/home/dunbar/Research/wolverine/data/new_wx/ftpext.usgs.gov/pub/wr/ak/anchorage/Sass/ForFrankie/DCIM/**/*.JPG"
#dir = os.path.join(dir,".JPG")
imageset = glob.glob(dir,recursive=True)

for imagepath in imageset:
	print(imagepath)
	image = cv2.imread(imagepath)
	plt.imshow(image)
	plt.show()
	keep = int(input("Keep Image?"))
	keep = bool(keep)
	if not keep:
	    print("Deleting: {}".format(imagepath))
	    os.remove(imagepath)
	print(keep)