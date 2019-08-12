import csv
import numpy as np 
import os
import matplotlib.pyplot as plt 

# This function is used to read the folders mentioned in folders.txt file.
def read_folders(file_name):
	# Open folders.txt file and read all the lines in it.
	files = []
	with open(file_name) as file:
		files = file.readlines()
		files = [x.split()[0] for x in files]
	return files

# This function is useful to read labels for each folder. You can change the label from the labels.txt file.
def readlabels(label_file):
	# Open labels.txt file and read all the lines in it.
	labels = []
	with open(label_file) as file:
		labels = file.readlines()
		labels = [x.split('\n')[0] for x in labels]
	return labels

# Read rot_err, trans_err from csv files.
def read_csv(folder_name):
	data = []
	# Each folder having results contain test.csv file with all the log.
	# Read all data from the csv file.
	with open(os.path.join(folder_name, 'test.csv')) as csvfile:
		csvreader = csv.reader(csvfile)
		for row in csvreader:
			row = [float(x) for x in row]
			data.append(row)

	rot_err, trans_err = [], []

	# Log stored is as per following sequence in csv files:
	# Sr. No. [0], time taken [1], number of iterations [2], translation error [3], rotation error [4].
	if folder_name[5:9]=='PNLK':
		for data_i in data:
			rot_err.append(data_i[2])
			trans_err.append(data_i[1])
	else:	
		for data_i in data:
			rot_err.append(data_i[4])
			trans_err.append(data_i[3])
	return rot_err, trans_err

# It will count the total number of test cases having rotation error below certain threshold.
def count_success_rot(rot_err):
	# A dictionary to store:
	# key: rotation error for success criteria ([0, 180, 0.5] degree)
	# value: total number of cases that passed this criteria
	success_dict = {}

	# Update dictionary with zero values for all keys.
	for i in range(0,1805,5):
		success_dict.update({i/10.0:0})

	# Find the values for each key value in dictionary.
	for rot in rot_err:
		idx = -1
		for i in range(1800,-5,-5):
			if rot>i/10.0:
				idx = i+5
				break
		for j in range(idx, 1805, 5):
			success_dict[j/10.0] = success_dict[j/10.0]+1
	return success_dict

def make_plot(files, labels):
	plt.figure()
	AUC = []		# Calculate Area under curve for each test folder. (quantification of success)

	for file_idx in range(len(files)):
		rot_err, trans_err = read_csv(files[file_idx])
		success_dict = count_success_rot(rot_err)

		x_range = list(success_dict.keys())
		x_range.sort()
		success = []
		for i in x_range:
			success.append(success_dict[i])

		# Ratio of successful cases to total test cases.
		success = np.array(success)/total_cases

		area = np.trapz(success, dx=0.5)
		AUC.append(area)

		plt.plot(x_range, success, linewidth=6, label=labels[file_idx])
	
	plt.xlabel('Rotation Error for Success Criteria', fontsize=40)
	plt.ylabel('Success Ratio', fontsize=40)
	
	plt.tick_params(labelsize=40, width=3, length=10)
	
	plt.xticks(np.arange(0,180.5,30))
	plt.yticks(np.arange(0,1.1,0.2))
		
	plt.xlim(-0.5,180)
	plt.ylim(0,1.01)
	
	plt.grid(True)
	plt.legend(fontsize=30, loc=4)

	AUC = np.array(AUC)/180.0
	print('Area Under Curve values: {}'.format(AUC))
	np.savetxt('auc.txt',AUC)

if __name__=='__main__':
	total_cases = 5070.0			# Total number of test cases used.

	colors = [[102,102,255], [215,85,85], [128,128,128], [255,153,51], [153,51,255], [102,255,102]]
	# colors = [[102,102,255], [215,85,85], [215,85,85], [215,85,85]]
	colors = np.array(colors)/255.0

	files = read_folders('folders.txt')
	labels = readlabels('labels.txt')
	
	make_plot(files, labels)
	plt.show()
