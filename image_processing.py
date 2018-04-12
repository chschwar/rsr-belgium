import os
from PIL import Image
import numpy as np

def loadImageAsNpArray(pathToImage):    
    img = Image.open(pathToImage).convert('L')
    return np.array(img, dtype=np.float32)

def loadImageAsNpArrayRGB(pathToImage):
    img = Image.open(pathToImage).convert('RGB')
    return np.array(img, dtype=np.float32)

def load_data(path_to_imag_dir, extension='ppm'):
	"""
	This function loads the training data
	"""
	labels = list([])
	images = list([])

	for path, subdirs, files in os.walk(path_to_imag_dir):
		for name in files:
			if (-1 != name.find(extension)):
				filname = os.path.join(path, name)				
				labels.append(int(filname.split('\\')[-2]))
				images.append(loadImageAsNpArrayRGB(filname)/255.0)

	return images, labels


