#!/bin/bash

cd $PWD/utils/pc_distance/
make -f makefile_10.0 clean
make -f makefile_10.0		# Change name to makefile_8.0 if you have CUDA-8.0 and Ubuntu-14.04
cd $PWD/../..

PY="python3"
# Prefer python3
# For python2.7, follow steps mentioned below:
# Open train_iPCRNet.py, test_iPCRNet.py, statistical_analysis.py
# 	a. Remove "from numpy import matlib as npm"
# 	b. Replace "npm" with "np.matlib"


LOG_DIR="log_itrPCRNet"		# Folder name to store log.4
MODE="train"				# Either train or test.
RESULTS="best_model"		# Keep saving network model after each epoch.
NOISE=False					# To train network with noise in source data. (False/True)

# Train iterative PCRNet
$PY iterative_PCRNet.py -log $LOG_DIR -mode $MODE -results $RESULTS -noise $NOISE

WEIGHTS=$PWD/$LOG_DIR/$RESULTS".ckpt"
LOG_RESULTS="results_itrPCRNet"

# Test iterative PCRNet
$PY results_itrPCRNet.py -weights $WEIGHTS -log $LOG_RESULTS -noise $NOISE


# Visualize the results for various templates
for i in 0 700 1900 3000 4000 5000
do
	$PY visualize_results.py -weights $WEIGHTS -idx $i
done
