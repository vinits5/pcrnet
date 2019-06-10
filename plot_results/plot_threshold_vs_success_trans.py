import csv
import numpy as np 
import os
import matplotlib.pyplot as plt 

# This function is used to read the folders mentioned in folders.txt file.
def read_folders(file_name):
	files = []
	with open(file_name) as file:
		files = file.readlines()
		files = [x.split()[0] for x in files]
	return files

# This function is useful to read labels for each folder. You can change the label from the labels.txt file.
def readlabels(label_file):
	labels = []
	with open(label_file) as file:
		labels = file.readlines()
		labels = [x.split('\n')[0] for x in labels]
	return labels

# Read rot_err, trans_err from csv files.
def read_csv(folder_name):
	data = []
	with open(os.path.join(folder_name, 'test.csv')) as csvfile:
		csvreader = csv.reader(csvfile)
		for row in csvreader:
			row = [float(x) for x in row]
			data.append(row)

	rot_err, trans_err = [], []
	for data_i in data:
		rot_err.append(data_i[4])
		trans_err.append(data_i[3])
	return rot_err, trans_err

def count_success(rot_err):
	success_dict = {}
	for i in range(0,210):
		success_dict.update({i/100.0:0})

	for rot in rot_err:
		idx = -1
		for i in range(200,-1,-1):
			if rot>i/100.0:
				idx = i+1
				break

		for j in range(idx, 210, 1):
			success_dict[j/100.0] = success_dict[j/100.0]+1
	return success_dict

def make_plot(files, labels):
	plt.figure()
	for file_idx in range(len(files)):
		rot_err, trans_err = read_csv(files[file_idx])
		success_dict = count_success(trans_err)

		x_range = success_dict.keys()
		x_range.sort()
		success = []
		for i in x_range:
			success.append(success_dict[i])
		success = np.array(success)/total_cases

		plt.plot(x_range, success, linewidth=3, label=labels[file_idx])
		# plt.scatter(x_range, success, s=50)
	plt.ylabel('Success Ratio', fontsize=40)
	plt.xlabel('Threshold for Translation Error', fontsize=40)
	plt.tick_params(labelsize=40, width=3, length=10)
	plt.grid(True)
	plt.ylim(0,1.005)
	plt.yticks(np.arange(0,1.2,0.2))
	plt.xticks(np.arange(0,2.1,0.2))
	plt.xlim(0,2)
	plt.legend(fontsize=30, loc=4)

if __name__=='__main__':
	total_cases = 5000.0

	colors = [[102,102,255], [215,85,85], [128,128,128], [255,153,51], [153,51,255], [102,255,102]]
	colors = np.array(colors)/255.0

	files = read_folders('folders.txt')
	labels = readlabels('labels.txt')
	
	make_plot(files, labels)
	plt.show()