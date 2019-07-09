import argparse
import h5py
import numpy as np
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import provider
import transforms3d.euler as t3d

parser = argparse.ArgumentParser()
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--class_file', type=str, default='classes.txt', help='Specify file which contains the classes.')
FLAGS = parser.parse_args()


NUM_POINT = 2048
NUM_CLASSES = 40

class Provider:
	def __init__(self):
		BASE_DIR = os.path.dirname(os.path.abspath(__file__))
		sys.path.append(BASE_DIR)

	def getDataFiles(self, list_filename):
		return [line.rstrip() for line in open(list_filename)]

	def load_h5(self, h5_filename):
		f = h5py.File(h5_filename)
		data = f['data'][:]
		label = f['label'][:]
		return (data, label)

	def loadDataFile(self, filename):
		return load_h5(filename)

	def load_h5_data_label_seg(self, h5_filename):
		f = h5py.File(h5_filename)
		data = f['data'][:]
		label = f['label'][:]
		seg = f['pid'][:]
		return (data, label, seg)

	def loadDataFile_with_seg(self, filename):
		return load_h5_data_label_seg(filename)

	def loadShapeNames(self, filename):
		with open(filename) as file:
			shapes = file.readlines()
			shapes = [x.split()[0] for x in shapes]
		return shapes

	def loadClasses(self, filename):
		with open(filename) as file:
			classes = file.readlines()
			classes = [x.split()[0] for x in classes]
			classes = [x.split(',') for x in classes]
			models = [int(x[1]) for x in classes]
			classes = [x[0] for x in classes]
		return classes, models


provider = Provider()

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles('data/modelnet40_ply_hdf5_2048/train_files.txt')
TEST_FILES = provider.getDataFiles('data/modelnet40_ply_hdf5_2048/test_files.txt')
shapes = provider.loadShapeNames('data/modelnet40_ply_hdf5_2048/shape_names.txt')


def apply_random_rotation(templates):
	# templates:		Array of templates or point clouds (BxNx3)
	templates = np.array(templates)
	for i in range(templates.shape[0]):
		# Random rotation in range [-45, 45] degrees.
		rot = t3d.euler2mat((np.pi/2)*np.random.random_sample()-np.pi/4, (np.pi/2)*np.random.random_sample()-np.pi/4, (np.pi/2)*np.random.random_sample()-np.pi/4, 'szyx')
		templates[i,:,:] = np.dot(rot, templates[i,:,:].T).T
	return templates

def find_models(category, model, templates, case):
	# model:		No of models to be stored for a particular category.
	# category: 	Name of the category to be stored.
	# templates:	Array having templates (BxNx3)
	# case:			Which files to be used? (test/train)

	if case == 'test':
		FILES = TEST_FILES
	if case == 'train':
		FILES = TRAIN_FILES
	print(FILES)
	count = 0														# Counter to find number of models.
	for train_idx in range(len(FILES)):						# Loop over all the training files from ModelNet40 data.
		current_data, current_label = provider.loadDataFile(FILES[train_idx])		# Load data of from a file.
		for i in range(current_data.shape[0]):
			if count<model and shapes.index(category)==current_label[i]:
				# import transforms3d.euler as t3d 
				# rot = t3d.euler2mat(0*np.pi/1	80, 0*np.pi/180, 90*np.pi/180, 'szyx')
				# templates.append((np.dot(rot, current_data[i].T).T))
				templates.append(current_data[i]/2.0)				# Append data if it belongs to the category and less than given number of models.
				count += 1
	return templates

def store_h5(templates, dict_name):
	# templates:	Array of templates (BxNx3)
	# dict_name:	Dictionary to store data.
	if not os.path.exists(os.path.join('data',dict_name)): os.mkdir(os.path.join('data',dict_name))

	file_names_txt = open(os.path.join('data',dict_name,'files.txt'),'w')			# Store names of files in txt file to read data.
	file_name = os.path.join('data',dict_name,'templates.h5')
	file_names_txt.write(file_name)
	f = h5py.File(file_name,'w')
	f.create_dataset('templates',data=templates)
	f.close()
	file_names_txt.close()

def remove_points(templates_ip):
	# templates_ip:		Array of point cloud (Nx3)
	templates = []
	for i in range(templates_ip.shape[0]):
		mean = np.mean(templates_ip, axis=0)-0.05
		if templates_ip[i,1]>mean[0]: #and templates_ip[i,1]<mean[1] and templates_ip[i,2]<mean[2]:
			templates.append(templates_ip[i])
	return np.asarray(templates[0:1024])

def create_partial_data(templates_ip):
	import helper
	templates = np.copy(templates_ip)
	partial_templates = np.zeros((templates.shape[0], 1024, templates.shape[2]))
	for i in range(templates.shape[0]):
		templates[i] = templates[i]-np.array([1,1,1])
		temp = remove_points(templates[i])
		print(temp.shape)
		helper.display_three_clouds(temp, templates[i], templates_ip[i], "")
		# helper.display_clouds_data(templates[i])

# Fit the data in a unit cube.
def normalize_data(templates):
	# Find min, max along x,y,z-axes
	for i in range(templates.shape[0]):
		min_x, min_y, min_z = min(templates[i,:,0]), min(templates[i,:,1]), min(templates[i,:,2])
		max_x, max_y, max_z = max(templates[i,:,0]), max(templates[i,:,1]), max(templates[i,:,2])

		# Subtract the min value and then divide by range and after that subtract 0.5 to shift the cube at origin.
		templates[i,:,0] = (templates[i,:,0] - min_x)/(max_x - min_x)
		templates[i,:,1] = (templates[i,:,1] - min_y)/(max_y - min_y)
		templates[i,:,2] = (templates[i,:,2] - min_z)/(max_z - min_z)

		templates[i] = templates[i] - np.mean(templates[i], axis=0)

	return templates

	
def store_datasets(dict_name, text_file, case):
	# dict_name:		Name of directory to store the data.
	# text_file:		Name of .h5 file to store the data.
	# case:				Train or test.
	templates = []

	if case == 'train':
		classes, models = provider.loadClasses(text_file)
	if case == 'test':
		classes, models = provider.loadClasses(text_file)

	for idx, val in enumerate(classes):
		if case == 'train':
			templates = find_models(val, models[idx], templates, case)
		if case == 'test':
			templates = find_models(val, models[idx], templates, case)
		print(shapes.index(val), ' ',len(templates))

	# create_partial_data(templates)
	# templates = normalize_data(np.array(templates))

	store_h5(templates, dict_name)

if __name__ == "__main__":
	store_datasets('noise_level_tests', 'noise_test.txt', 'train')