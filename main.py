# training a simple net based on https://www.datacamp.com/community/tutorials/tensorflow-tutorial

from image_processing import load_data
from simple_net_with_tensorflow import first_test, simple_net
import numpy as np

# Import the 'pyplot' module
import matplotlib.pyplot as plt


def exploring_data(images, labels):
	# Determine the (random) indexes of the images that you want to see 
	traffic_signs = [2, 3, 6, 55, 99, 212, 300, 400]

	# Fill out the subplots with the random images that you defined 
	for i in range(len(traffic_signs)):
	    plt.subplot(2, 4, i+1)
	    plt.axis('off')
	    plt.imshow(images[traffic_signs[i]])
	    plt.subplots_adjust(wspace=0.5)
	    print("shape: {0}, min: {1}, max: {2}".format(images[traffic_signs[i]].shape, 
                                                  images[traffic_signs[i]].min(), 
                                                  images[traffic_signs[i]].max()))
	plt.show()

	# Get the unique labels 
	unique_labels = set(labels)

	# Initialize the figure
	plt.figure(figsize=(15, 15))

	# Set a counter
	i = 1

	# For each unique label,
	for label in unique_labels:
	    # You pick the first image for each label
	    image = images[labels.index(label)]
	    # Define 64 subplots 
	    plt.subplot(8, 8, i)
	    # Don't include axes
	    plt.axis('off')
	    # Add a title to each subplot 
	    plt.title("Label {0} ({1})".format(label, labels.count(label)))
	    # Add 1 to the counter
	    i += 1
	    # And you plot this first image 
	    plt.imshow(image)
	    
	# Show the plot
	plt.show()

def prepare_data(images):
	# Import the `transform` module from `skimage`
	from skimage import transform
	from skimage.color import rgb2gray

	# Rescale the images in the 'images'' array
	images_28x28 = [transform.resize(image, (28, 28)) for image in images]
	images_28x28 = np.array(images_28x28)
	images_28x28 = rgb2gray(images_28x28)

	return images_28x28


if __name__ == "__main__":
	import os

	ROOT_PATH = "D:/workspace/dataset/road-signs/BelgiumTSC/"
	train_data_directory = os.path.join(ROOT_PATH, "Training")
	test_data_directory = os.path.join(ROOT_PATH, "Testing")


	train_x, train_y = load_data(train_data_directory)
	test_x, test_y = load_data(test_data_directory)



	train_images_28x28 = prepare_data(train_x)
	test_images_28x28 = prepare_data(test_x)
	#exploring_data(train_x_28x28, train_y)

	simple_net(train_images_28x28, np.array(train_y), test_images_28x28, np.array(test_y))
