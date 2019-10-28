import tensorflow as tf
import numpy as np
import math
import sys
import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import helper
import tf_util
import tf_util_loss

def placeholder_inputs(batch_size, num_point):
	source_pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
	template_pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
	return source_pointclouds_pl,template_pointclouds_pl

def get_model(source_point_cloud, template_point_cloud, is_training, bn_decay=None):
	point_cloud = tf.concat([source_point_cloud, template_point_cloud],0)
	batch_size = point_cloud.get_shape()[0].value
	num_point = point_cloud.get_shape()[1].value
	end_points = {}

	input_image = tf.expand_dims(point_cloud, -1)

	net = tf_util.conv2d(input_image, 64, [1,3],
						 padding='VALID', stride=[1,1],
						 bn=False, is_training=is_training,
						 scope='conv1', bn_decay=bn_decay)
	net = tf_util.conv2d(net, 64, [1,1],
						 padding='VALID', stride=[1,1],
						 bn=False, is_training=is_training,
						 scope='conv2', bn_decay=bn_decay)
	net = tf_util.conv2d(net, 64, [1,1],
						 padding='VALID', stride=[1,1],
						 bn=False, is_training=is_training,
						 scope='conv3', bn_decay=bn_decay)
	net = tf_util.conv2d(net, 128, [1,1],
						 padding='VALID', stride=[1,1],
						 bn=False, is_training=is_training,
						 scope='conv4', bn_decay=bn_decay)
	net = tf_util.conv2d(net, 1024, [1,1],
						 padding='VALID', stride=[1,1],
						 bn=False, is_training=is_training,
						 scope='conv5', bn_decay=bn_decay)

	# Symmetric function: max pooling
	net = tf_util.max_pool2d(net, [num_point,1],
							 padding='VALID', scope='maxpool')
	net = tf.reshape(net, [batch_size, -1])
	source_global_feature = tf.slice(net, [0,0], [int(batch_size/2),1024])
	template_global_feature = tf.slice(net, [int(batch_size/2),0], [int(batch_size/2),1024])
	return source_global_feature, template_global_feature

def get_pose(source_global_feature, template_global_feature, is_training, bn_decay=None):
	net = tf.concat([source_global_feature,template_global_feature],1)
	net = tf_util.fully_connected(net, 1024, bn=False, is_training=is_training,scope='fc1', bn_decay=bn_decay)
	net = tf_util.fully_connected(net, 512, bn=False, is_training=is_training,scope='fc2', bn_decay=bn_decay)
	net = tf_util.fully_connected(net, 256, bn=False, is_training=is_training,scope='fc3', bn_decay=bn_decay)
	net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,scope='dp4')
	predicted_transformation = tf_util.fully_connected(net, 7, activation_fn=None, scope='fc4')
	return predicted_transformation

def get_loss(predicted_transformation, batch_size, template_pointclouds_pl, source_pointclouds_pl):
	with tf.variable_scope('loss') as LossEvaluation:
		predicted_position = tf.slice(predicted_transformation,[0,0],[batch_size,3])
		predicted_quat = tf.slice(predicted_transformation,[0,3],[batch_size,4])

		# with tf.variable_scope('quat_normalization') as norm:
		norm_predicted_quat = tf.reduce_sum(tf.square(predicted_quat),1)
		norm_predicted_quat = tf.sqrt(norm_predicted_quat)
		norm_predicted_quat = tf.reshape(norm_predicted_quat,(batch_size,1))
		const = tf.constant(0.0000001,shape=(batch_size,1),dtype=tf.float32)
		norm_predicted_quat = tf.add(norm_predicted_quat,const)
		predicted_norm_quat = tf.divide(predicted_quat,norm_predicted_quat)

		transformed_predicted_point_cloud = helper.transformation_quat_tensor(source_pointclouds_pl, predicted_norm_quat,predicted_position)

		#loss = tf_util_loss.earth_mover(template_pointclouds_pl, transformed_predicted_point_cloud)
		loss = tf_util_loss.chamfer(template_pointclouds_pl, transformed_predicted_point_cloud)
	return loss

if __name__=='__main__':
	with tf.Graph().as_default():
		inputs = tf.zeros((32,1024,3))
		outputs = get_model(inputs, inputs, tf.constant(True))
		print(outputs)
