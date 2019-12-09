import argparse
import math
import h5py
import numpy as np
from numpy import matlib as npm
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import tf_util
import helper
import transforms3d.euler as t3d
import transforms3d
import time
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
# Implementation parameters
parser.add_argument('-log', '--log_dir', default='results_itrPCRNet', help='Store the results. test_network_log_data : network: itr/siamese/icp, log: multi_catg/airplane_multi_models, data: test_data1/unseen_data1')
parser.add_argument('-weights', '--model_path', type=str, default='log_itrPCRNet/best_model.ckpt', help='Path of the weights (.ckpt file) to be used for test')
parser.add_argument('-noise', '--use_noise_data', required=True, type=bool, default=False, help='Use Noisy Data for Source')

parser.add_argument('--iterations', type=int, default=5, help='No of Iterations for pose estimation')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--filename', type=str, default='test', help='Name of files')

# Data parameters
parser.add_argument('--template_idx', type=int, default=0, help='Index of template to be used for evaluation of network')
parser.add_argument('--data_dict', type=str, default='train_data',help='Data used to train templates or multi_model_templates')
parser.add_argument('--eval_poses', type=str, default='itr_net_test_data45.csv', help='Poses for evaluation')
parser.add_argument('--threshold', type=float, default=1e-07, help='threshold for convergence criteria')

# Useful and default parameters.
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='ipcr_model', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--decay_step', type=int, default=400000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
parser.add_argument('--centroid_sub', type=bool, default=True, help='Centroid Subtraction from Source and Template before Pose Prediction.')
FLAGS = parser.parse_args()

# Parameters for data
NUM_POINT = FLAGS.num_point
MAX_NUM_POINT = 2048
NUM_CLASSES = 40
centroid_subtraction_switch = FLAGS.centroid_sub
BATCH_SIZE = FLAGS.batch_size

# Network hyperparameters
MAX_LOOPS = FLAGS.iterations
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

# Calculate Learning Rate during training.
def get_learning_rate(batch):
	learning_rate = tf.train.exponential_decay(
						BASE_LEARNING_RATE,  # Base learning rate.
						batch * BATCH_SIZE,  # Current index into the dataset.
						DECAY_STEP,          # Decay step.
						DECAY_RATE,          # Decay rate.
						staircase=True)
	learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
	return learning_rate        

# Get Batch Normalization decay.
def get_bn_decay(batch):
	bn_momentum = tf.train.exponential_decay(
					  BN_INIT_DECAY,
					  batch*BATCH_SIZE,
					  BN_DECAY_DECAY_STEP,
					  BN_DECAY_DECAY_RATE,
					  staircase=True)
	bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
	return bn_decay

def find_errors(gt_pose, final_pose):
	# Simple euler distand between translation part.
	gt_position = gt_pose[0:3]				
	predicted_position = final_pose[0:3]
	translation_error = np.sqrt(np.sum(np.square(gt_position - predicted_position)))

	# Convert euler angles rotation matrix.
	gt_euler = gt_pose[3:6]
	pt_euler = final_pose[3:6]
	gt_mat = t3d.euler2mat(gt_euler[2],gt_euler[1],gt_euler[0],'szyx')
	pt_mat = t3d.euler2mat(pt_euler[2],pt_euler[1],pt_euler[0],'szyx')

	# Multiply inverse of one rotation matrix with another rotation matrix.
	error_mat = np.dot(pt_mat,np.linalg.inv(gt_mat))
	_,angle = transforms3d.axangles.mat2axangle(error_mat)			# Convert matrix to axis angle representation and that angle is error.
	return translation_error, abs(angle*(180/np.pi))

def check_convergence(previous_pose, predicted_pose):
	prevT, predT, identityT = np.zeros((4,4)), np.zeros((4,4)), np.zeros((4,4))
	prevT[3,3], predT[3,3] = 1, 1
	identityT[0,0], identityT[1,1], identityT[2,2], identityT[3,3] = 1,1,1,1
	prevT[0:3,3] = previous_pose[0:3]
	predT[0:3,3] = predicted_pose[0:3]

	prevT[0:3,0:3] = t3d.quat2mat([previous_pose[3],previous_pose[4],previous_pose[5],previous_pose[6]])
	predT[0:3,0:3] = t3d.quat2mat([predicted_pose[3],predicted_pose[4],predicted_pose[5],predicted_pose[6]])

	errorT = np.dot(predT, np.linalg.inv(prevT))
	errorT = errorT - identityT
	errorT = errorT*errorT
	error = np.sum(errorT)

	converged = False
	if error < FLAGS.threshold:
		converged = True
	return converged

def check_convergenceT(previous_T, predicted_T):
	identityT = np.zeros((4,4))
	identityT[0,0], identityT[1,1], identityT[2,2], identityT[3,3] = 1,1,1,1

	errorT = np.dot(predicted_T, np.linalg.inv(previous_T))
	errorT = errorT - identityT
	errorT = errorT*errorT
	error = np.sum(errorT)

	converged = False
	if error < FLAGS.threshold:
		converged = True
	return converged

def train():
	with tf.Graph().as_default():
		with tf.device('/cpu:0'):
			batch = tf.Variable(0)										# That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
			
		with tf.device('/gpu:'+str(GPU_INDEX)):
			is_training_pl = tf.placeholder(tf.bool, shape=())			# Flag for dropouts.
			bn_decay = get_bn_decay(batch)								# Calculate BN decay.
			learning_rate = get_learning_rate(batch)					# Calculate Learning Rate at each step.

			# Define a network to backpropagate the using final pose prediction.
			with tf.variable_scope('Network') as _:
				# Get the placeholders.
				source_pointclouds_pl, template_pointclouds_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
				# Extract Features.
				source_global_feature, template_global_feature = MODEL.get_model(source_pointclouds_pl, template_pointclouds_pl, is_training_pl, bn_decay=bn_decay)
				# Find the predicted transformation.
				predicted_transformation = MODEL.get_pose(source_global_feature,template_global_feature,is_training_pl, bn_decay=bn_decay)
				# Find the loss using source and transformed template point cloud.
				# loss = MODEL.get_loss(predicted_transformation, BATCH_SIZE, template_pointclouds_pl, source_pointclouds_pl)
				loss = 0

		with tf.device('/cpu:0'):
			# Add ops to save and restore all the variables.
			saver = tf.train.Saver()

			
		# Create a session
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.allow_soft_placement = True
		config.log_device_placement = False
		sess = tf.Session(config=config)

		# Init variables
		init = tf.global_variables_initializer()
		sess.run(init, {is_training_pl: True})

		saver.restore(sess, FLAGS.model_path)

		# Create a dictionary to pass the tensors and placeholders in train and eval function for Network.
		ops = {'source_pointclouds_pl': source_pointclouds_pl,
			   'template_pointclouds_pl': template_pointclouds_pl,
			   'is_training_pl': is_training_pl,
			   'predicted_transformation': predicted_transformation,
			   'loss': loss,
			   'step': batch}

		templates = helper.loadData(FLAGS.data_dict)
		eval_poses = helper.read_poses(FLAGS.data_dict, FLAGS.eval_poses)			# Read all the poses data for evaluation.
		eval_network(sess, ops, templates, eval_poses)

def eval_network(sess, ops, templates, poses):
	# Arguments:
	# sess: 		Tensorflow session to handle tensors.
	# ops:			Dictionary for tensors of Network
	# templates:	Training Point Cloud data.
	# poses: 		Training pose data.

	is_training = False
	display_ptClouds = False
	display_poses = False
	display_poses_in_itr = False
	display_ptClouds_in_itr = False

	loss_sum = 0											# Total Loss in each batch.
	num_batches = int(poses.shape[0]/BATCH_SIZE) 				# Number of batches in an epoch.
	print('Number of batches to be executed: {}'.format(num_batches))

	# Store time taken, no of iterations, translation error and rotation error for registration.
	TIME, ITR, Trans_Err, Rot_Err = [], [], [], []
	idxs_5_5, idxs_10_1, idxs_20_2 = [], [], []

	if FLAGS.use_noise_data:
		print(FLAGS.data_dict)
		templates, sources = helper.read_noise_data(FLAGS.data_dict)
		print(templates.shape, sources.shape)
	
	for fn in range(num_batches):
		start_idx = fn*BATCH_SIZE 			# Start index of poses.
		end_idx = (fn+1)*BATCH_SIZE 		# End index of poses.
		
		if FLAGS.use_noise_data:
			template_data = np.copy(templates[fn,:,:]).reshape(1,-1,3)				# As template_data is changing.
			source_data = np.copy(sources[fn,:,:]).reshape(1,-1,3)
			batch_euler_poses = poses[start_idx:end_idx]			# Extract poses for batch training.
		else:
			template_data = np.copy(templates[fn,:,:]).reshape(1,-1,3)				# As template_data is changing.

			batch_euler_poses = poses[start_idx:end_idx]			# Extract poses for batch training.
			source_data = helper.apply_transformation(template_data, batch_euler_poses)		# Apply the poses on the templates to get source data.

		template_data = template_data[:,0:NUM_POINT,:]
		source_data = source_data[:,0:NUM_POINT,:]

		# Just to visualize the data.
		TEMPLATE_DATA = np.copy(template_data)				# Store the initial template to visualize results.
		SOURCE_DATA = np.copy(source_data)					# Store the initial source to visualize results.

		# Subtract the Centroids from the Point Clouds.
		if centroid_subtraction_switch:
			source_data = source_data - np.mean(source_data, axis=1, keepdims=True)
			template_data = template_data - np.mean(template_data, axis=1, keepdims=True)

		# To visualize the source and point clouds:
		if display_ptClouds:
			helper.display_clouds_data(source_data[0])
			helper.display_clouds_data(template_data[0])

		TRANSFORMATIONS = np.identity(4)				# Initialize identity transformation matrix.
		TRANSFORMATIONS = npm.repmat(TRANSFORMATIONS,BATCH_SIZE,1).reshape(BATCH_SIZE,4,4)		# Intialize identity matrices of size equal to batch_size

		# previous_pose = np.array([0,0,0,1,0,0,0])
		previous_T = np.eye(4)

		start = time.time()												# Log start time.
		# Iterations for pose refinement.
		for loop_idx in range(MAX_LOOPS):
			for network_itr in range(7):
				# Feed the placeholders of Network19 with template data and source data.
				feed_dict = {ops['source_pointclouds_pl']: source_data,
							 ops['template_pointclouds_pl']: template_data,
							 ops['is_training_pl']: is_training}
				predicted_transformation = sess.run([ops['predicted_transformation']], feed_dict=feed_dict)		# Ask the network to predict the pose.

				# Apply the transformation on the source data and multiply it to transformation matrix obtained in previous iteration.
				TRANSFORMATIONS, source_data = helper.transformation_quat2mat(predicted_transformation, TRANSFORMATIONS, source_data)

				# Display Results after each iteration.
				if display_poses_in_itr:
					print(predicted_transformation[0,0:3])
					print(predicted_transformation[0,3:7]*(180/np.pi))
				if display_ptClouds_in_itr:
					helper.display_clouds_data(template_data[0])

			# Feed the placeholders of Network_L with source data and template data obtained from N-Iterations.
			feed_dict = {ops['source_pointclouds_pl']: source_data,
						 ops['template_pointclouds_pl']: template_data,
						 ops['is_training_pl']: is_training}

			# Ask the network to predict transformation, calculate loss using distance between actual points.
			predicted_transformation = sess.run([ops['predicted_transformation']], feed_dict=feed_dict)

			# Apply the final transformation on the source data and multiply it with the transformation matrix obtained from N-Iterations.
			TRANSFORMATIONS, source_data = helper.transformation_quat2mat(predicted_transformation, TRANSFORMATIONS, source_data)

			if check_convergenceT(previous_T, TRANSFORMATIONS[0]):
				break
			else:
				previous_T = np.copy(TRANSFORMATIONS[0])
		end = time.time()													# Log end time.


		final_pose = helper.find_final_pose_inv(TRANSFORMATIONS)		# Find the final pose (translation, orientation (euler angles in degrees)) from transformation matrix.
		final_pose[0,0:3] = final_pose[0,0:3] + np.mean(SOURCE_DATA, axis=1)[0]

		translation_error, rotational_error = find_errors(batch_euler_poses[0], final_pose[0])

		TIME.append(end-start)
		ITR.append(loop_idx+1)
		Trans_Err.append(translation_error)
		Rot_Err.append(rotational_error)

		if rotational_error<20 and translation_error<0.2:
			if rotational_error<10 and translation_error<0.1:
				if rotational_error<5 and translation_error<0.05:
					idxs_5_5.append(fn)
				idxs_10_1.append(fn)
			idxs_20_2.append(fn)

		# Display the ground truth pose and predicted pose for first Point Cloud in batch 
		if display_poses:
			print('Ground Truth Position: {}'.format(batch_euler_poses[0,0:3].tolist()))
			print('Predicted Position: {}'.format(final_pose[0,0:3].tolist()))
			print('Ground Truth Orientation: {}'.format((batch_euler_poses[0,3:6]*(180/np.pi)).tolist()))
			print('Predicted Orientation: {}'.format((final_pose[0,3:6]*(180/np.pi)).tolist()))

		# Display Loss Value.
		# helper.display_three_clouds(TEMPLATE_DATA[0],SOURCE_DATA[0],template_data[0],"")
		print("Batch: {} & time: {}, iteration: {}".format(fn, end-start, loop_idx+1))

	log = {'TIME': TIME, 'ITR':ITR, 'Trans_Err': Trans_Err, 'Rot_Err': Rot_Err, 'idxs_5_5': idxs_5_5, 'idxs_10_1': idxs_10_1, 'idxs_20_2': idxs_20_2, 'num_batches': num_batches}

	helper.log_test_results(FLAGS.log_dir, FLAGS.filename, log)

def store_params(FLAGS):
	with open(os.path.join(FLAGS.log_dir,'params.txt'),'w') as file:
		file.write('Model:\t\t\t\t\t\t{}\n'.format(FLAGS.model))
		file.write('Model Path:\t\t\t\t\t{}\n'.format(FLAGS.model_path))
		file.write('Data Dict:\t\t\t\t\t{}\n'.format(FLAGS.data_dict))
		file.write('Log Dir:\t\t\t\t\t{}\n'.format(FLAGS.log_dir))
		file.write('Evaluation Pose:\t\t\t{}\n'.format(FLAGS.eval_poses))
		file.write('Max Allowed Iterations:\t\t{}\n'.format(FLAGS.iterations))
		file.write('Threshold for convergence:\t{}\n'.format(FLAGS.threshold))
	return True

def set_params(model, data_dict, model_path, log_dir, eval_poses):
	set_p = False
	if not set_p:
		if not os.path.exists(log_dir): os.mkdir(log_dir)
		FLAGS.model = model
		FLAGS.data_dict = data_dict
		FLAGS.model_path = model_path
		FLAGS.log_dir = log_dir
		FLAGS.eval_poses = eval_poses
		set_p = store_params(FLAGS)
	return set_p

if __name__=='__main__':
	if set_params(FLAGS.model, FLAGS.data_dict, FLAGS.model_path, FLAGS.log_dir, FLAGS.eval_poses):
		MODEL = importlib.import_module(FLAGS.model) # import network module
		helper.download_data(FLAGS.data_dict)
		train()
	
	# Used the following code to test multiple datas and multiple logs.

	# model_paths2test = ['log_multi_catg/model250.ckpt']#,['log_multi_catg_noise/model300.ckpt']
	# # model_paths2test = ['log_car_multi_models/model200.ckpt']#, 'log_car_multi_models_noise/model350.ckpt']
	# data_dicts2test = ['train_data','unseen_data']#, 'car_data']#, 'unseen_data']
	# eval_poses2test = ['itr_net_test_data45.csv']

	# FLAGS.use_noise_data = False

	# for mpt in model_paths2test:
	# 	for ddt in data_dicts2test:
	# 		for ept in eval_poses2test:
	# 			log_dir = 'test_itr_'+str(mpt[4:len(mpt)-14])+'_'+str(ddt)+'_'+str(ept[len(ept)-6:len(ept)-4])
	# 			if set_params(FLAGS.model, ddt, mpt, log_dir, ept):
	# 				MODEL = importlib.import_module(FLAGS.model)
	# 				train()