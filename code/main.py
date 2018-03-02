#########################################################################################
#Description:
# Read the training data and testing data and perform the classifications
#
#########################################################################################
import numpy as np
from libs import *
from sklearn import preprocessing
print('Preprocessing the Mnist data and storing it into npz files...\n')
print('Training Data...\n')
try:#training data
	data = np.load("trainData.npz")
	trains_images = data['trains_images']
	train_images_label = data['train_images_label']
except FileNotFoundError:#read from mnist file if presaved data not found, supported nly by python3 or above
	images = gzip.open('MNIST_Data/train-images-idx3-ubyte.gz', 'rb')
	labels = gzip.open('MNIST_Data/train-labels-idx1-ubyte.gz', 'rb')
	(trains_images,train_images_label) =read_gz(images, labels);
	np.savez("trainData.npz", trains_images=trains_images, train_images_label=train_images_label)
print('Test Data...\n')
try:
	data = np.load("testData.npz")
	test_images = data['test_images']
	test_images_label = data['test_images_label']
except FileNotFoundError:
	images = gzip.open('MNIST_Data/t10k-images-idx3-ubyte.gz', 'rb')
	labels = gzip.open('MNIST_Data/t10k-labels-idx1-ubyte.gz', 'rb')
	(test_images,test_images_label) =read_gz(images,labels);
	np.savez("testData.npz", test_images=test_images, test_images_label=test_images_label)

############################################### PART 1 Solution###########################################################
# print(trains_images.shape)
#finding and saving mean images of every digits
mean_train_images = np.zeros((10, 28, 28), dtype=float32) #for storing mean images of the mnist digits, 10 digits in total
for i in range(10):
	mean_train_images[i,:,:] = np.mean(trains_images[np.where(train_images_label == i)[0],:,:], axis=0)#find the indices of images of particular label and take average of them
	# view_image(mean_train_images[i,:,:], str(i)+"-mean") #uncomment this line to save new images
# print(mean_train_images.shape)
# view_image(mean_train_images[1,:,:], 1)

#finding and saving standard deviation images of every digits
std_train_images = np.zeros((10, 28, 28), dtype=float32) #for storing mean images of the mnist digits, 10 digits in total
for i in range(10):
	std_train_images[i,:,:] = np.std(trains_images[np.where(train_images_label == i)[0],:,:], axis=0)#find the indices of images of particular label and take average of them
	# view_image(std_train_images[i,:,:], str(i)+"-std") ##uncomment this line to save new images
############################################### PART 1 Solution Ends #####################################################