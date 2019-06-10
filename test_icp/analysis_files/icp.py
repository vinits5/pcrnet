""" ICP algorithm

	References:
	(ICP)
	[1] Paul J. Besl and Neil D. McKay,
		"A method for registration of 3-D shapes",
		PAMI Vol. 14, Issue 2, pp. 239-256, 1992.
	(SVD)
	[2] K. S. Arun, T. S. Huang and S. D. Blostein,
		"Least-Squares Fitting of Two 3-D Point Sets",
		PAMI Vol. 9, Issue 5, pp.698--700, 1987
"""
import numpy as np
from scipy.spatial import KDTree
import math
import time
import transforms3d
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..'))
import helper


def _icp_find_rigid_transform(p_from, p_target):
	######### flag = 1 for SVD 
	######### flag = 2 for Gaussian
	A, B = np.copy(p_from), np.copy(p_target)

	centroid_A = np.mean(A, axis=0)
	centroid_B = np.mean(B, axis=0)

	A -= centroid_A						# 500x3
	B -= centroid_B						# 500x3

	H = np.dot(A.T, B)					# 3x3
	# print(H)
	U, S, Vt = np.linalg.svd(H)
	# H_new = U*np.matrix(np.diag(S))*Vt
	# print(H_new)
	R = np.dot(Vt.T, U.T)

	# special reflection case
	if np.linalg.det(R) < 0:
		Vt[2,:] *= -1
		R = np.dot(Vt.T, U.T)
	t = np.dot(-R, centroid_A) + centroid_B
	return R, t


def _icp_Rt_to_matrix(R, t):
	# matrix M = [R, t; 0, 1]
	Rt = np.concatenate((R, np.expand_dims(t.T, axis=-1)), axis=1)
	a = np.concatenate((np.zeros_like(t), np.ones(1)))
	M = np.concatenate((Rt, np.expand_dims(a, axis=0)), axis=0)
	return M

def check_convergence(M_prev, M, ftol):
	identityT = np.eye(4)
	errorT = np.dot(M, np.linalg.inv(M_prev))
	errorT = errorT - identityT
	errorT = errorT*errorT
	error = np.sum(errorT)

	converged = False
	if error < ftol:
		converged = True
	return converged

# def find_errors(final_rot, final_trans):
# 	import transforms3d
# 	import transforms3d.euler as t3d
# 	# Simple euler distand between translation part.
# 	gt_pose = [0.5, 0.2, 0.4, 40*(np.pi/180), 20*(np.pi/180), 30*(np.pi/180)]
# 	gt_position = gt_pose[0:3]				
# 	predicted_position = final_trans
# 	translation_error = np.sqrt(np.sum(np.square(gt_position - predicted_position)))
	

# 	# Convert euler angles rotation matrix.
# 	gt_euler = gt_pose[3:6]
# 	gt_mat = t3d.euler2mat(gt_euler[2],gt_euler[1],gt_euler[0],'szyx')

# 	# Multiply inverse of one rotation matrix with another rotation matrix.

# 	error_mat = np.dot(final_rot,np.linalg.inv(gt_mat))
# 	_,angle = transforms3d.axangles.mat2axangle(error_mat)			# Convert matrix to axis angle representation and that angle is error.
	
# 	return translation_error, abs(angle*(180/np.pi))


class ICP:
	""" Estimate a rigid-body transform g such that:
		p0 = g.p1
	"""
	def __init__(self, p0, p1, tree_M_sampled):
		self.p0 = p0		# M_sampled (model sampled pts)
		self.p1 = p1		# S (sensor pts)
		# self.nearest = KDTree_M_sampled(self.p0)
		self.nearest = tree_M_sampled

		self.g_series = None

	def compute(self, max_iter, ftol):
		dim_k = self.p0.shape[1]
		g = np.eye(dim_k + 1, dtype=self.p0.dtype)
		p = np.copy(self.p1)						# S (sensor pts)
		
		self.g_series = np.zeros((max_iter + 1, dim_k + 1, dim_k + 1), dtype=g.dtype)
		self.g_series[0, :, :] = g

		M_prev = np.eye(4)

		itr = -1
		TERR, RERR, TRANSFORMATION_store = [], [], []
		for itr in range(max_iter):
			neighbor_idx = self.nearest.query(p)[1]		# KNN search from Model points for sensor points as target.
			
			targets = self.p0[neighbor_idx]
			R, t = _icp_find_rigid_transform(p, targets)
		   
			new_p = np.dot(R, p.T).T + t
	 
			# if np.sum(np.abs(p - new_p)) < ftol:
			# 	break
	 
			p = np.copy(new_p)
			dg = _icp_Rt_to_matrix(R, t)
			M = np.copy(dg)
			new_g = np.dot(dg, g)
			g = np.copy(new_g)
			self.g_series[itr + 1, :, :] = g

			MAT = np.linalg.inv(new_g)
			print(MAT)
			MAT = MAT.flatten()
			TRANSFORMATION_store.append(MAT)
			# terr, rerr = find_errors(MAT[0:3,0:3], MAT[0:3,3])
			# TERR.append(terr)
			# RERR.append(rerr)

			if check_convergence(M_prev, M, ftol):
				# pass
				break
			else:
				M_prev = np.copy(M)
		
			show_plot_switch = 0
			if show_plot_switch == 1:
				print(targets.shape, "targets.shape")
	
				import matplotlib.pyplot as plt
				from mpl_toolkits.mplot3d import Axes3D

				fig = plt.figure()
				ax = Axes3D(fig)
				ax = fig.add_subplot(111, projection='3d')
				ax.set_label("x - axis")
				ax.set_label("y - axis")
				ax.set_label("z - axis")
				ax.plot(self.p0[:,0], self.p0[:,1], self.p0[:,2], "o", color="red", ms=4, mew=0.5)
				ax.plot(p[:,0], p[:,1], p[:,2], "o", color="blue", ms=8, mew=0)   
				ax.plot(targets[:,0], targets[:,1], targets[:,2], "o", color="green", ms=4, mew=0)
				plt.show()        

		# np.savetxt('rotation_error.txt',RERR)
		# np.savetxt('translation_error.txt',TERR)
		np.savetxt('TRANSFORMATION_store.txt',TRANSFORMATION_store)
		self.g_series[(itr+1):, :, :] = g
		return g, p, (itr + 1), neighbor_idx


#### please note that the S over here, is not total niumber of sensor points
#### it is the number sensor points sampled for ICP 
def icp_test(S, M, tree_M, M_sampled, tree_M_sampled, S_given, gt_pose, MAX_LOOPS, ftol):
	from math import sin, cos
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
 
	icp = ICP(M_sampled, S, tree_M_sampled)
	start = time.time()

	matrix, points, itr, neighbor_idx = icp.compute(MAX_LOOPS, ftol)		# Perform the iterations.

	end = time.time()

	final_pose = helper.find_final_pose((np.linalg.inv(matrix)).reshape((1,4,4)))

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

	title += '\nElapsed Time: '+str(np.round((end-start)*1000,3))+' ms'+' & Iterations: '+str(itr)
	title += ' & Method: ICP'

	rot = matrix[0:3,0:3]
	tra = np.reshape(np.transpose(matrix[0:3,3]), (3,1))
	transformed = np.matmul(rot, np.transpose(S_given)) + tra
	return final_pose, M, S_given, transformed.T, title, end-start, itr

if __name__ == '__main__':
	import helper
	templates = helper.process_templates('multi_model_templates')

	template_idx = 201
	NUM_POINT = 1024
	MAX_LOOPS = 500

	M_given = templates[template_idx,:,:]

	x_trans = 0.5
	y_trans = 0.5
	z_trans = 0.5
	x_rot = 45*(np.pi/180)
	y_rot = 45*(np.pi/180)
	z_rot = 45*(np.pi/180)
	gt_pose = np.array([[x_trans, y_trans, z_trans, x_rot, y_rot, z_rot]])			# Pose: Source to Template

	S_given = helper.apply_transformation(M_given.reshape((1,-1,3)), gt_pose)[0]
	S_given = S_given + np.random.normal(0,1,S_given.shape)

	M_given = M_given[0:NUM_POINT,:]				# template data
	S_given = S_given[0:NUM_POINT,:]				# source data

	tree_M = KDTree(M_given)
	tree_M_sampled = KDTree(M_given[0:100,:])

	final_pose, model_data, sensor_data, predicted_data, title = icp_test(S_given[0:100,:], M_given, tree_M, M_given[0:100,:], tree_M_sampled, S_given, gt_pose.reshape((1,6)), MAX_LOOPS, 1e-05)
	helper.display_three_clouds(model_data, sensor_data, predicted_data, title)