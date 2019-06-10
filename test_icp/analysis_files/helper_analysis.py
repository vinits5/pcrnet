import numpy as np
import transforms3d.euler as t3d 
import tensorflow as tf 
import time
import matplotlib.pyplot as plt 
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..'))
import helper

MAX_NUM_POINT = 2048
BATCH_SIZE = 1

class NetworkAnalysis:
	def __init__(self, templates, NUM_POINT, mode, sess = None, ops19 = None, ops_L = None, saver = None, model_path = None):
		self.sess = sess
		self.ops_L = ops_L
		self.ops19 = ops19
		self.templates = templates
		self.is_training = False
		if not mode=='icp_test':
			saver.restore(sess,model_path)
		self.NUM_POINT = NUM_POINT
	
	def set_template_idx(self, template_idx):
		self.template_idx = template_idx

	def set_MAX_LOOPS(self, MAX_LOOPS):
		self.MAX_LOOPS = MAX_LOOPS

	def set_ftol(self, ftol):
		self.ftol = ftol

	# Run network for one source and template
	def test_one_case(self, template_data, source_data):
		# template_data: 				Input Template Data for network
		# source_data:					Input Source Data for network

		display_poses_in_itr = False
		display_ptClouds_in_itr = False
		template = np.copy(template_data)
		source = np.copy(source_data)
		template_check = np.copy(template_data)

		TRANSFORMATIONS = np.identity(4)		# Initialize identity transformation matrix.
		TRANSFORMATIONS = np.matlib.repmat(TRANSFORMATIONS,BATCH_SIZE,1).reshape(BATCH_SIZE,4,4)
		TRANSFORMATIONS_check = np.copy(TRANSFORMATIONS)

		start = time.time()
		# helper.display_three_clouds(template_data[0],source_data[0],source_data[0],"Iteration: 0")
		for loop_idx in range(self.MAX_LOOPS-1):
			# Feed the placeholders of Network19 with template data and source data.
			feed_dict = {self.ops19['source_pointclouds_pl']: source,
						 self.ops19['template_pointclouds_pl']: template,
						 self.ops19['is_training_pl']: self.is_training}
			predicted_transformation = self.sess.run([self.ops19['predicted_transformation']], feed_dict=feed_dict) 		# Ask the network to predict the pose.

			# Apply the transformation on the template data and multiply it to transformation matrix obtained in previous iteration.
			TRANSFORMATIONS, template = helper.transformation_quat2mat(predicted_transformation,TRANSFORMATIONS,template)

			if np.sum(np.abs(template[:,0:100,:] - template_check[:,0:100,:])) < self.ftol:
			# if np.sum(np.dot(np.linalg.inv(TRANSFORMATIONS_check), TRANSFORMATIONS)) < ftol:
				break
			else:
				# TRANSFORMATIONS_check = np.copy(TRANSFORMATIONS)
				template_check = np.copy(template)

			# Display Results after each iteration.
			if display_poses_in_itr:
				print(predicted_transformation[0,0:3])
				print(predicted_transformation[0,3:7]*(180/np.pi))
			if display_ptClouds_in_itr:
				helper.display_clouds_data(template[0])
				# transformed_source_data = np.dot(np.linalg.inv(TRANSFORMATIONS[0])[0:3,0:3], source[0].T).T + np.linalg.inv(TRANSFORMATIONS[0])[0:3,3]
				# helper.display_three_clouds(template_data[0], source[0], transformed_source_data, "Iteration: "+str(loop_idx+1))

		# Feed the placeholders of Network_L with source data and template data obtained from N-Iterations.
		feed_dict = {self.ops_L['source_pointclouds_pl']: source,
					 self.ops_L['template_pointclouds_pl']: template,
					 self.ops_L['is_training_pl']: self.is_training}

		# Ask the network to predict transformation, calculate loss using distance between actual points.
		step, predicted_transformation = self.sess.run([self.ops_L['step'], self.ops_L['predicted_transformation']], feed_dict=feed_dict)

		end = time.time()
		loss_val = 0

		# Apply the final transformation on the template data and multiply it with the transformation matrix obtained from N-Iterations.
		TRANSFORMATIONS, template = helper.transformation_quat2mat(predicted_transformation, TRANSFORMATIONS, template)
		final_pose = helper.find_final_pose(TRANSFORMATIONS)
		transformed_source_data = np.dot(np.linalg.inv(TRANSFORMATIONS[0])[0:3,0:3], source[0].T).T + np.linalg.inv(TRANSFORMATIONS[0])[0:3,3]

		return final_pose, TRANSFORMATIONS, loss_val, template, transformed_source_data, end-start, (loop_idx+1)

	# Function used to create loss vs rotation and translation plots.
	def generate_loss_2Dplots(self, axis , x_axis_param):
		# Parameters to deal with:
		# axis 					# This will decide the rotation or translation of point cloud about a particular axis. 'x' or 'y' or 'z'
		# x_axis_param			# This will decide either to rotate or translate the point cloud 'rotation' or 'translation'.

		template_data = self.templates[self.template_idx,:,:].reshape((1,MAX_NUM_POINT,3))		# Extract the template and reshape it.
		template_data = template_data[:,0:self.NUM_POINT,:]

		loss = []																			# Store the losses.
		if x_axis_param == 'rotation':
			angles = []						# Store the angles.
			# Loop to find loss for various angles from -90 to 90.
			for i in range(-90,91):
				if axis == 'X':
					gt_pose = np.array([[0.0, 0.0, 0.0, i*(np.pi/180), 0.0, 0.0]])			# New poses as per each index.
				if axis == 'Y':
					gt_pose = np.array([[0.0, 0.0, 0.0, 0.0, i*(np.pi/180), 0.0]])			# New poses as per each index.
				if axis == 'Z':
					gt_pose = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, i*(np.pi/180)]])			# New poses as per each index.

				source_data = helper.apply_transformation(template_data,gt_pose)		# Generate Source Data.
				final_pose, TRANSFORMATIONS, loss_i, predicted_data, transformed_source_data, _, _ = self.test_one_case(template_data, source_data)	# Find final transformation by network.
				loss.append(loss_i)
				angles.append(i)

				# helper.display_three_clouds(template_data[0],source_data[0],transformed_source_data,"Results")

			plt.plot(angles, loss, linewidth=6)
			plt.xlabel('Rotation about '+axis+',Y,Z-axes (in degrees)', fontsize=40)
			plt.ylabel('Error in Pose (L2 Norm)', fontsize=40)
			plt.ylim((-0.5,2))
			plt.tick_params(labelsize=30)
			plt.show()

		if x_axis_param == 'translation':
			position = []						# Store the angles.
			# Loop to find loss for various angles from -90 to 90.
			for i in range(-10,11):
				if axis == 'X':
					gt_pose = np.array([[i/10, 0.0, 0.0, 0.0, 0.0, 0.0]])			# New poses as per each index.
				if axis == 'Y':
					gt_pose = np.array([[0.0, i/10, 0.0, 0.0, 0.0, 0.0]])			# New poses as per each index.
				if axis == 'Z':
					gt_pose = np.array([[0.0, 0.0, i/10, 0.0, 0.0, 0.0]])			# New poses as per each index.

				source_data = helper.apply_transformation(template_data,gt_pose)		# Generate Source Data.
				final_pose, TRANSFORMATIONS, loss_i, predicted_data, transformed_source_data, _, _ = self.test_one_case(template_data,source_data,MAX_LOOPS)	# Find final transformation by network.
				loss.append(np.sum(np.square(final_pose[0]-gt_pose[0]))/6)				# Calculate L2 Norm between gt pose and predicted pose.
				position.append(i/10.0)

				# helper.display_three_clouds(template_data[0],source_data[0],transformed_source_data,"Results")

			plt.plot(position, loss, linewidth=6)
			plt.ylim((-0.5,2))
			plt.xlabel('Translations about '+axis+'-axis', fontsize=40)
			plt.ylabel('Error in Poses (L2 Norm)', fontsize=40)
			plt.tick_params(labelsize=30)
			plt.show()

	# Function used to create loss vs rotation and translation plots.
	def generate_loss_3Dplots(self, axis, x_axis_param):
		# Parameters to deal with:
		# axis					This will decide the rotation or translation of point cloud about a particular axis. 'x' or 'y' or 'z'
		# x_axis_param			This will decide either to rotate or translate the point cloud 'rotation' or 'translation'.

		template_data = self.templates[self.template_idx,:,:].reshape((1,MAX_NUM_POINT,3))		# Extract the template and reshape it.
		template_data = template_data[:,0:self.NUM_POINT,:]

		loss = []
		angles_x = []
		angles_y = []														# Store the losses.
		if x_axis_param == 'rotation':
			# Loop to find loss for various angles from -90 to 90.
			for i in range(-90,91):
				print('I: {}'.format(i))
				for j in range(-90,91):
					if axis == 'XY':
						gt_pose = np.array([[0.0, 0.0, 0.0, i*(np.pi/180), j*(np.pi/180), 0.0]])			# New poses as per each index.
					if axis == 'YZ':
						gt_pose = np.array([[0.0, 0.0, 0.0, 0.0, i*(np.pi/180), j*(np.pi/180)]])			# New poses as per each index.
					if axis == 'XZ':
						gt_pose = np.array([[0.0, 0.0, 0.0, i*(np.pi/180), 0.0, j*(np.pi/180)]])			# New poses as per each index.

					source_data = helper.apply_transformation(template_data,gt_pose)		# Generate Source Data.
					final_pose, TRANSFORMATIONS, loss_i, predicted_data, transformed_source_data, _, _ = self.test_one_case(template_data, source_data)	# Find final transformation by network.
					loss.append(loss_i)
					angles_x.append(i)
					angles_y.append(j)
					# helper.display_three_clouds(template_data[0],source_data[0],transformed_source_data,"Results")

			fig = plt.figure()

			ax = fig.add_subplot(111,projection='3d')
			ax.scatter(angles_x,angles_y,loss)
			ax.set_xlabel('Rotation Angle about '+axis[0]+'-axis', fontsize=25, labelpad=25)
			ax.set_ylabel('Rotation Angle about '+axis[1]+'-axis', fontsize=25, labelpad=25)
			ax.set_zlabel('Error in Poses (L2 Norm)', fontsize=25, labelpad=25)
			ax.tick_params(labelsize=25)
			plt.show()

	def generate_loss_vs_itr(self, axis, x_axis_param):
		# axis					This will decide the rotation or translation of point cloud about a particular axis. 'x' or 'y' or 'z'
		# x_axis_param			This will decide either to rotate or translate the point cloud 'rotation' or 'translation'.

		template_data = self.templates[self.template_idx,:,:].reshape((1,MAX_NUM_POINT,3))		# Extract the template and reshape it.
		template_data = template_data[:,0:self.NUM_POINT,:]

		loss = []
		angles_x = []
		angles_y = []														# Store the losses.
		if x_axis_param == 'rotation':
			# Loop to find loss for various angles from -90 to 90.
			for i in [4,8,12,16,20,24,28,32]:
				self.set_MAX_LOOPS(i)
				print('I: {}'.format(i))
				for j in range(-90,91):
					if axis == 'X':
						gt_pose = np.array([[0.0, 0.0, 0.0, j*(np.pi/180), 0.0, 0.0]])			# New poses as per each index.
					if axis == 'Y':
						gt_pose = np.array([[0.0, 0.0, 0.0, 0.0, j*(np.pi/180), 0.0]])			# New poses as per each index.
					if axis == 'Z':
						gt_pose = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, j*(np.pi/180)]])			# New poses as per each index.

					source_data = helper.apply_transformation(template_data,gt_pose)		# Generate Source Data.
					final_pose, TRANSFORMATIONS, loss_i, predicted_data, transformed_source_data, _ = self.test_one_case(template_data,source_data)	# Find final transformation by network.
					loss.append(np.sum(np.square(final_pose[0]-gt_pose[0]))/6)
					angles_x.append(i)
					angles_y.append(j)

			fig = plt.figure()

			ax = fig.add_subplot(111,projection='3d')
			ax.scatter(angles_x,angles_y,loss)
			ax.set_xlabel('No of Iterations', fontsize=25, labelpad=25)
			ax.set_ylabel('Rotation Angle about '+axis[0]+'-axis', fontsize=25, labelpad=25)
			ax.set_zlabel('Error in Poses (L2 Norm)', fontsize=25, labelpad=25)
			ax.tick_params(labelsize=25)
			plt.show()

	# Generates & Stores Time, Rotation Error, Translation Error & No. of iterations to a .csv file.
	# Also stores mean and variance of all parameters in a .txt file.
	def generate_stat_data(self, filename):
		eval_poses = helper.read_poses(FLAGS.data_dict, FLAGS.eval_poses)
		template_data = self.templates[self.template_idx,:,:].reshape((1,MAX_NUM_POINT,3))
		TIME, ITR, Trans_Err, Rot_Err = [], [], [], []
		for pose in eval_poses:
			source_data = helper.apply_transformation(self.templates[self.template_idx,:,:],pose.reshape((-1,6)))
			final_pose, _, _, _, _, elapsed_time, itr = self.test_one_case(source_data,template_data)
			translation_error, rotational_error = self.find_errors(pose.reshape((-1,6)), final_pose)
			TIME.append(elapsed_time)
			ITR.append(itr)
			Trans_Err.append(translation_error)
			Rot_Err.append(rotational_error)
		
		TIME_mean, ITR_mean, Trans_Err_mean, Rot_Err_mean = sum(TIME)/len(TIME), sum(ITR)/len(ITR), sum(Trans_Err)/len(Trans_Err), sum(Rot_Err)/len(Rot_Err)
		TIME_var, ITR_var, Trans_Err_var, Rot_Err_var = np.var(np.array(TIME)), np.var(np.array(ITR)), np.var(np.array(Trans_Err)), np.var(np.array(Rot_Err))
		import csv
		with open(filename + '.csv','w') as csvfile:
			csvwriter = csv.writer(csvfile)
			for i in range(len(TIME)):
				csvwriter.writerow([i, TIME[i], ITR[i], Trans_Err[i], Rot_Err[i]])
		with open(filename+'.txt','w') as file:
			file.write("Mean of Time: {}".format(TIME_mean))
			file.write("Mean of Iterations: {}".format(ITR_mean))
			file.write("Mean of Translation Error: {}".format(Trans_Err_mean))
			file.write("Mean of Rotation Error: {}".format(Rot_Err_mean))

			file.write("Variance in Time: {}".format(TIME_var))
			file.write("Variance in Iterations: {}".format(ITR_var))
			file.write("Variance in Translation Error: {}".format(Trans_Err_var))
			file.write("Variance in Rotation Error: {}".format(Rot_Err_var))


	def generate_results(self, ftol, gt_pose, swap_case):
		template_data = self.templates[self.template_idx,:,:].reshape((1,MAX_NUM_POINT,3))		# Extract the template and reshape it.
		template_data = template_data[:,0:self.NUM_POINT,:]

		source_data = helper.apply_transformation(template_data,gt_pose)		# Generate Source Data.
		# source_data = source_data + np.random.normal(0,0.001,source_data.shape)	# Noisy Data

		if swap_case:
			final_pose, TRANSFORMATIONS, loss_i, predicted_data, transformed_source_data, elapsed_time, itr = self.test_one_case(source_data, template_data)	# Find final transformation by network.
		else:
			final_pose, TRANSFORMATIONS, loss_i, predicted_data, transformed_source_data, elapsed_time, itr = self.test_one_case(template_data, source_data)	# Find final transformation by network.

		if not swap_case:
			title = "Actual T (Red->Green): "
			for i in range(len(gt_pose[0])):
				if i>2:
					title += str(round(gt_pose[0][i]*(180/np.pi),2))
				else:
					title += str(gt_pose[0][i])
				title += ', '
			title += "\nPredicted T (Red->Blue): "
			for i in range(len(final_pose[0])):
				if i>2:
					title += str(round(final_pose[0,i]*(180/np.pi),3))
				else:
					title += str(round(final_pose[0,i],3))
				title += ', '	
		else:
			title = "Predicted Transformation: "
			for i in range(len(final_pose[0])):
				if i>2:
					title += str(round(final_pose[0,i]*(180/np.pi),3))
				else:
					title += str(round(final_pose[0,i],3))
				title += ', '	

		title += '\nElapsed Time: '+str(np.round(elapsed_time*1000,3))+' ms'+' & Iterations: '+str(itr)
		title += ' & Iterative Network'

		self.find_errors(gt_pose, final_pose)
		if swap_case:
			helper.display_three_clouds(source_data[0], template_data[0], transformed_source_data, title)
		else:
			helper.display_three_clouds(template_data[0], source_data[0], transformed_source_data, title)

	def generate_icp_results(self, gt_pose):
		from icp import icp_test
		from scipy.spatial import KDTree
		M_given = self.templates[self.template_idx,:,:]

		S_given = helper.apply_transformation(M_given.reshape((1,-1,3)), gt_pose)[0]
		# S_given = S_given + np.random.normal(0,1,S_given.shape)						# Noisy Data

		M_given = M_given[0:self.NUM_POINT,:]				# template data
		S_given = S_given[0:self.NUM_POINT,:]				# source data

		tree_M = KDTree(M_given)
		tree_M_sampled = KDTree(M_given[0:100,:])

		final_pose, model_data, sensor_data, predicted_data, title, _, _ = icp_test(S_given[0:100,:], M_given, tree_M, M_given[0:100,:], tree_M_sampled, S_given, gt_pose.reshape((1,6)), self.MAX_LOOPS, self.ftol)		
		self.find_errors(gt_pose, final_pose)
		helper.display_three_clouds(model_data, sensor_data, predicted_data, title)

	def find_errors(self, gt_pose, final_pose):
		import transforms3d
		gt_position = gt_pose[0,0:3]
		predicted_position = final_pose[0,0:3]

		translation_error = np.sqrt(np.sum(np.square(gt_position - predicted_position)))
		print("Translation Error: {}".format(translation_error))

		gt_euler = gt_pose[0,3:6]
		pt_euler = final_pose[0,3:6]

		gt_mat = t3d.euler2mat(gt_euler[2],gt_euler[1],gt_euler[0],'szyx')
		pt_mat = t3d.euler2mat(pt_euler[2],pt_euler[1],pt_euler[0],'szyx')

		error_mat = np.dot(pt_mat,np.linalg.inv(gt_mat))
		_,angle = transforms3d.axangles.mat2axangle(error_mat)
		print("Rotation Error: {}".format(abs(angle*(180/np.pi))))
		return translation_error, angle*(180/np.pi)
