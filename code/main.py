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
# print(trains_images.shape)
# print(train_images_label.shape)
view_image(trains_images[9999,:,:], train_images_label[9999])
# trains_images = trains_images.reshape([60000,784])#flattening the input array
# test_images = test_images.reshape([10000,784])
# trains_images = preprocessing.scale(trains_images)#standardising the image data set with zero mean and unit standard deviation
# test_images = preprocessing.scale(test_images)
# trains_images = (trains_images - trains_images.min())/(trains_images.max()-trains_images.min())
# test_images = (test_images - test_images.min())/(test_images.max()-test_images.min())
# print(trains_images.shape)
# print(train_images_label.shape)
# print(test_images.shape)
# print(test_images_label.shape)
