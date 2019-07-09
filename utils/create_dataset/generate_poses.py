import numpy as np 
import csv
import os

def generate_poses(file_path):
	with open(file_path, 'a') as csvfile:
		csvwriter = csv.writer(csvfile)
		for idx in range(10000):
			orientation_x = np.round(np.random.uniform()*2*45*(np.pi/180) - 45*(np.pi/180),4)
			orientation_y = np.round(np.random.uniform()*2*45*(np.pi/180) - 45*(np.pi/180),4)
			orientation_z = np.round(np.random.uniform()*2*45*(np.pi/180) - 45*(np.pi/180),4)
			x = np.round(2*np.random.uniform()-1,4)
			y = np.round(2*np.random.uniform()-1,4)
			z = np.round(2*np.random.uniform()-1,4)
			pose = [x,y,z,orientation_x,orientation_y,orientation_z]
			csvwriter.writerow(pose)

if __name__= '__main__':
	generate_poses(file_path='data/car_data/itr_net_eval_data180.csv')