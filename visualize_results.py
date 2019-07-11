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

parser = argparse.ArgumentParser()
parser.add_argument('-weights','--model_path', type=str, default='log_multi_catg_noise/model300.ckpt', help='Path of the weights (.ckpt file) to be used for test')
parser.add_argument('-idx','--template_idx', type=int, default='log_multi_catg_noise/model300.ckpt', help='Template Idx')


parser.add_argument('--iterations', type=int, default=8, help='No of Iterations for Registration')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='ipcr_model', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log_test', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Number of Points in a Point Cloud [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=501, help='Epoch to run [default: 250]')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=3000000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
parser.add_argument('--centroid_sub', type=bool, default=True, help='Centroid Subtraction from Source and Template before Registration.')
parser.add_argument('--use_pretrained_model', type=bool, default=False, help='Use a pretrained model of airplane to initialize the training.')
parser.add_argument('--use_random_poses', type=bool, default=False, help='Use of random poses to train the model in each batch')
parser.add_argument('--data_dict', type=str, default='train_data',help='Templates data dictionary used for training')
parser.add_argument('--train_poses', type=str, default='itr_net_train_data45.csv', help='Poses for training')
parser.add_argument('--eval_poses', type=str, default='itr_net_eval_data45.csv', help='Poses for evaluation')
FLAGS = parser.parse_args()

TRAIN_POSES = FLAGS.train_poses
EVAL_POSES = FLAGS.eval_poses

BATCH_SIZE = 1

# Parameters for data
NUM_POINT = FLAGS.num_point
MAX_NUM_POINT = 2048
NUM_CLASSES = 40
centroid_subtraction_switch = FLAGS.centroid_sub

# Network hyperparameters
MAX_EPOCH = FLAGS.max_epoch
MAX_LOOPS = FLAGS.iterations
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

# Model Import
MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
 
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
			   'step': batch}

		templates = helper.loadData(FLAGS.data_dict)
		eval_poses = helper.read_poses(FLAGS.data_dict, EVAL_POSES)			# Read all the poses data for evaluation.

		# Just to test the results
		test_one_epoch(sess, ops, templates, eval_poses, saver, FLAGS.model_path)

def test_one_epoch(sess, ops, templates, poses, saver, model_path):
	# Arguments:
	# sess: 		Tensorflow session to handle tensors.
	# ops:			Dictionary for tensors of Network
	# templates:	Training Point Cloud data.
	# poses: 		Training pose data.
	# saver: 		To restore the weights.
	# model_path: 	Path of log directory.

	saver.restore(sess, model_path)			# Restore the weights of trained network.

	is_training = False
	display_ptClouds = False
	display_poses = False
	display_poses_in_itr = False
	display_ptClouds_in_itr = False
	swap_case = False
	MAX_LOOPS = 4
	
	template_data = np.zeros((BATCH_SIZE,MAX_NUM_POINT,3))		# Extract Templates for batch training.
	template_data[0]=np.copy(templates[FLAGS.template_idx,:,:])

	batch_euler_poses = poses[0].reshape((1,6))		# Extract poses for batch training.

	# Define test case.
	batch_euler_poses[0]=[0.4,0.5,0.1,10*(np.pi/180),20*(np.pi/180),20*(np.pi/180)]
	source_data = helper.apply_transformation(template_data,batch_euler_poses)		# Apply the poses on the templates to get source data.

	# Chose Random Points from point clouds for training.
	if np.random.random_sample()<0:
		source_data = helper.select_random_points(source_data, NUM_POINT)						# probability that source data has different points than template
	else:
		source_data = source_data[:,0:NUM_POINT,:]
	# Add noise to source point cloud.
	if np.random.random_sample()<1.0:
		source_data = helper.add_noise(source_data)

	# Only choose limited number of points from the source and template data.
	source_data = source_data[:,0:NUM_POINT,:]
	template_data = template_data[:,0:NUM_POINT,:]

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

	TRANSFORMATIONS = np.identity(4)		# Initialize identity transformation matrix.
	TRANSFORMATIONS = npm.repmat(TRANSFORMATIONS,BATCH_SIZE,1).reshape(BATCH_SIZE,4,4)			# Intialize identity matrices of size equal to batch_size

	# Store the transformed point clouds after each iteration.
	ITR = np.zeros((MAX_LOOPS,template_data.shape[0],template_data.shape[1],template_data.shape[2]))

	# Iterations for pose refinement.
	for loop_idx in range(MAX_LOOPS-1):
		# 4a
		# Feed the placeholders of Network with template data and source data.
		feed_dict = {ops['source_pointclouds_pl']: source_data,
					 ops['template_pointclouds_pl']: template_data,
					 ops['is_training_pl']: is_training}
		predicted_transformation = sess.run([ops['predicted_transformation']], feed_dict=feed_dict) 		# Ask the network to predict the pose.
		#print (predicted_transformation[0])

		# 4b,4c
		# Apply the transformation on the template data and multiply it to transformation matrix obtained in previous iteration.
		TRANSFORMATIONS, source_data = helper.transformation_quat2mat(predicted_transformation, TRANSFORMATIONS, source_data)

		# Display Results after each iteration.
		if display_poses_in_itr:
			print(predicted_transformation[0,0:3])
			print(predicted_transformation[0,3:7]*(180/np.pi))
		if display_ptClouds_in_itr:
			helper.display_clouds_data(source_data[0])
		ITR[loop_idx,:,:,:]=source_data
	
	# Feed the placeholders of Network with source data and template data obtained from N-Iterations.
	feed_dict = {ops['source_pointclouds_pl']: source_data,
				 ops['template_pointclouds_pl']: template_data,
				 ops['is_training_pl']: is_training}

	# Ask the network to predict transformation, calculate loss using distance between actual points.
	step, predicted_transformation = sess.run([ops['step'], ops['predicted_transformation']], feed_dict=feed_dict)

	# Apply the final transformation on the template data and multiply it with the transformation matrix obtained from N-Iterations.
	TRANSFORMATIONS, source_data = helper.transformation_quat2mat(predicted_transformation, TRANSFORMATIONS, source_data)

	final_pose = helper.find_final_pose_inv(TRANSFORMATIONS)
	final_pose[0,0:3] = final_pose[0,0:3] + np.mean(SOURCE_DATA, axis=1)[0]
	
	title = "Actual T (Red->Green): "
	for i in range(len(batch_euler_poses[0])):
		if i>2:
			title += str(round(batch_euler_poses[0][i]*(180/np.pi),2))
		else:
			title += str(batch_euler_poses[0][i])
		title += ', '
	title += "\nPredicted T (Red->Blue): "
	for i in range(len(final_pose[0])):
		if i>2:
			title += str(round(final_pose[0,i]*(180/np.pi),3))
		else:
			title += str(round(final_pose[0,i],3))
		title += ', '

	# Display the ground truth pose and predicted pose for first Point Cloud in batch 
	if display_poses:
		print('Ground Truth Position: {}'.format(batch_euler_poses[0,0:3].tolist()))
		print('Predicted Position: {}'.format(final_pose[0,0:3].tolist()))
		print('Ground Truth Orientation: {}'.format((batch_euler_poses[0,3:6]*(180/np.pi)).tolist()))
		print('Predicted Orientation: {}'.format((final_pose[0,3:6]*(180/np.pi)).tolist()))

	helper.display_three_clouds(TEMPLATE_DATA[0],SOURCE_DATA[0],source_data[0],title)



if __name__ == "__main__":
	helper.download_data(FLAGS.data_dict)
	train()