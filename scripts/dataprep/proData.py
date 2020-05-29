import numpy
import shutil
import os
import sys
import glob

if __name__ == "__main__" :

    directory = str(sys.argv[1])
    dirname = directory.split("/")[1].strip("/")
    images = glob.glob(os.path.join(directory,"images","*.JPG"))
    images = [image for image in images if os.stat(image).st_size > 0]
    json = glob.glob(os.path.join(directory,"images_json","*.json"))
    json = json[0]
    for image in images:
        newimage = image.replace("IMGP",dirname+"_")
        os.rename(image,newimage)
        newjson = newimage.split("/")[-1].replace(".JPG",".json")
        newjson = os.path.join(directory,"images_json",newjson)
        try:
            shutil.copyfile(json, newjson)
        except:
            pass
