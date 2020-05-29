import matplotlib
matplotlib.use('agg')
import glimpse
from glimpse.imports import (sys, datetime, matplotlib, np, os)
import glob

SNAP = datetime.timedelta(hours=1)
MAXDT = datetime.timedelta(days=0.5)
MATCH_SEQ = (np.arange(3) + 1)
MAX_RATIO = 0.6
MAX_ERROR = 0.03 # fraction of image width
N_MATCHES = 50
MIN_MATCHES = 50
root = '~/Research/wolverine/data'
TOUNGE_MATCHES_PATH = os.path.join(root,'subdata/matches/tounge.pkl')

class optView():
	def __init__(self,imagedir,basemodelpath,basedict,imgpoints,worldpoints):
		self.imagepaths =  glob.glob(os.path.join(imagedir,'*.JPG'),recursive=True)
		self.basecam = glimpse.helpers.merge_dicts(glimpse.Camera.read(basemodelpath).as_dict(),basedict)
		self.imagepoints = imgpoints
		self.worldpoints = worldpoints
		self.refimg = glimpse.Image(path=self.imagepaths[0],exif=glimpse.Exif(self.imagepaths[0]),cam=self.basecam.copy())
		#self.refimg.anchor=True
		
	


	def modelRef(self):
		points = glimpse.optimize.Points(self.refimg.cam,self.imagepoints,self.worldpoints)
		camera = glimpse.optimize.Cameras(cams=[self.refimg.cam],controls=[points],cam_params=dict(viewdir=True,f=True,p=True,c=True))
		camera.set_cameras(camera.fit())


	def getImages(self):
		self.images = []
		[self.images.append( glimpse.Image(path=imagepath,exif=glimpse.Exif(imagepath),cam=self.refimg.cam.copy()) ) for imagepath in self.imagepaths[1:]]
		self.images.sort(key= lambda img: img.datetime)
		print("Found {} Images \n".format(len(self.images)))


	def iterMatch(self,setSize=50):
		subSet = []
		anchorimage = None
		for i,image in enumerate(self.images):
			subSet.append(image)
			if i > 0 and i % setSize == 0 or i == len(self.images)-1:
				print("\nSubset: {} Images: {}\n".format((i+1),len(subSet)))
				if anchorimage is not None and i > setSize:
					subSet.insert(0,anchorimage)
				matcher = glimpse.optimize.KeypointMatcher(subSet)

				matcher.build_keypoints(
    				clear_images=True,
        			clear_keypoints=True,
        			overwrite=False)

				matcher.build_matches(
        			maxdt=MAXDT, 
        			seq=MATCH_SEQ,
        			path=TOUNGE_MATCHES_PATH,
			        max_ratio=0.75,
			        weights=True,
			        max_distance=None,
			        parallel=4)

				matcher.filter_matches(clear_weights=True)
				matcher.convert_matches(glimpse.optimize.RotationMatchesXY, clear_uvs=True)

				camParams = [dict() if img.anchor else dict(viewdir=True) for img in matcher.images]
				cams = [image.cam for image in matcher.images]
				controls = list(matcher.matches.data)

				Cameras = glimpse.optimize.Cameras(
			    	cams = cams, 
			    	controls=controls,
			    	cam_params=camParams
			    	)

				fit = Cameras.fit(ftol=1)
				Cameras.set_cameras(fit)
				anchorimage = subSet[-1]
				anchorimage.anchor=True
				subSet = None
				matcher = None
				Cameras = None
				fit = None
				subSet = []
	def run(self):
		self.modelRef()
		self.getImages()
		self.iterMatch()

	def saveCams(self):
		directory = self.savedir
		for images,newcam in  zip(self.images,[img.cam for img in self.images]):
			filename = images.path
			path = os.path.join(directory,chext(filename,"JSON"))
			print("\n",path,"\n")
			newcam.write(path=path,attributes=("viewdir","xyz","sensorsz","fmm","cmm","p","k","imgsz","f"))
