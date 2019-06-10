import helper
from analysis_files import icp
from scipy.spatial import KDTree
import argparse
import os
import numpy as np 
import transforms3d.euler as t3d 
import transforms3d
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='test_icp_unseen_data', help='Store the results')
parser.add_argument('--filename', type=str, default='test', help='Name of files')

parser.add_argument('--data_dict', type=str, default='train_data',help='Data used to train templates or multi_model_templates')
parser.add_argument('--eval_poses', type=str, default='itr_net_test_data45.csv', help='Poses for evaluation')
parser.add_argument('--pairs_file', type=str, default='itr_net_test_data_pairs.csv', help='Pairs of templates and poses')
parser.add_argument('--threshold', type=float, default=1e-07, help='threshold for convergence criteria')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
FLAGS = parser.parse_args()

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

# Store all the results.
# if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
DATA_DICT = FLAGS.data_dict
EVAL_POSES = FLAGS.eval_poses

def run():
	NUM_POINT = FLAGS.num_point

	if not use_noise_data:
		templates = helper.loadData(FLAGS.data_dict)
		pairs = helper.read_pairs(FLAGS.data_dict, FLAGS.pairs_file)
	else:
		templates, sources = helper.read_noise_data(FLAGS.data_dict)
		# templates = helper.loadData(FLAGS.data_dict)
	eval_poses = helper.read_poses(FLAGS.data_dict, FLAGS.eval_poses)			# Read all the poses data for evaluation.
	eval_poses = eval_poses[0:1,:]
	num_batches = eval_poses.shape[0]

	TIME, ITR, Trans_Err, Rot_Err = [], [], [], []
	idxs_5_5, idxs_10_1, idxs_20_2 = [], [], []

	counter = 0
	for fn,gt_pose in enumerate(eval_poses):
		if fn >0:
			break
		if not use_noise_data:
			# template_idx = pairs[fn,1]
			template_idx = 0
			M_given = templates[template_idx,:,:]
			S_given = helper.apply_transformation(M_given.reshape((1,-1,3)), gt_pose.reshape((1,6)))[0]
		else:
			M_given = templates[fn,:,:]
			S_given = sources[fn,:,:]

		# M_given = np.loadtxt('template_car_itr.txt')
		# S_given = np.loadtxt('source_car_itr.txt')
		# helper.display_clouds_data(M_given)
		# helper.display_clouds_data(S_given)

		# To generate point cloud for Xueqian: For CAD model figures
		# gt_pose = np.array([[0.5,0.2,0.4,40*(np.pi/180),20*(np.pi/180),30*(np.pi/180)]])
		# templates = helper.loadData('unseen_data')
		# gt_pose = np.array([[-0.3,-0.7,0.4,-34*(np.pi/180),31*(np.pi/180),-27*(np.pi/180)]])
		# gt_pose = np.array([[0.5929,-0.0643,-0.961,0.4638,-0.3767,-0.6253]])
		# M_given = templates[48,:,:]
		# S_given = helper.apply_transformation(M_given.reshape(1,-1,3),gt_pose)
		# S_given = helper.add_noise(S_given)
		# S_given = S_given[0]

		M_given = M_given[0:NUM_POINT,:]				# template data
		S_given = S_given[0:NUM_POINT,:]				# source data

		tree_M = KDTree(M_given)
		tree_M_sampled = KDTree(M_given[0:100,:])

		final_pose, model_data, sensor_data, predicted_data, _, time_elapsed, itr = icp.icp_test(S_given[0:100,:], M_given, tree_M, M_given[0:100,:], tree_M_sampled, S_given, gt_pose.reshape((1,6)), 100, FLAGS.threshold)
		translation_error, rotational_error = find_errors(gt_pose[0], final_pose[0])
		print(translation_error, rotational_error)

		TIME.append(time_elapsed)
		ITR.append(itr)
		Trans_Err.append(translation_error)
		Rot_Err.append(rotational_error)

		if rotational_error<20 and translation_error<0.2:
			if rotational_error<10 and translation_error<0.1:
				if rotational_error<5 and translation_error<0.05:
					idxs_5_5.append(fn)
				idxs_10_1.append(fn)
			idxs_20_2.append(fn)

		print('Batch: {}, Iterations: {}, Time: {}'.format(counter, itr, time_elapsed))
		# counter += 1

		# helper.display_three_clouds(M_given, S_given, predicted_data, "")
		# np.savetxt('template_piano.txt',M_given)
		# np.savetxt('source_piano.txt',S_given)
		# np.savetxt('predicted_piano.txt',predicted_data)

	log = {'TIME': TIME, 'ITR':ITR, 'Trans_Err': Trans_Err, 'Rot_Err': Rot_Err, 'idxs_5_5': idxs_5_5, 'idxs_10_1': idxs_10_1, 'idxs_20_2': idxs_20_2, 'num_batches': num_batches}

	helper.log_test_results(FLAGS.log_dir, FLAGS.filename, log)


def set_params(data_dict, log_dir, pairs_file):
	if not os.path.exists(log_dir): os.mkdir(log_dir)
	FLAGS.data_dict = data_dict
	FLAGS.log_dir = log_dir
	FLAGS.pairs_file = pairs_file

if __name__=='__main__':
	data_dicts2test = ['unseen_data']

	use_noise_data = True
	
	for ddt in data_dicts2test:
		log_dir = 'test_icp_'+str(ddt)+'_45_maxItr100'
		set_params(ddt, log_dir, FLAGS.pairs_file)
		run()