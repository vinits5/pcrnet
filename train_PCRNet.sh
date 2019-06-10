#!/bin/bash

cd $PWD/utils/pc_distance/
make -f makefile_10.0		# Change name to makefile_8.0 if you have CUDA-8.0 and Ubuntu-14.04
cd $PWD/../..

PY="python3"
# Prefer python3
# For python2.7, follow steps mentioned below:
# Open train_iPCRNet.py, test_iPCRNet.py, statistical_analysis.py
# 	a. Remove "from numpy import matlib as npm"
# 	b. Replace "npm" with "np.matlib"


LOG_DIR="log_PCRNet"		# Folder name to store log.4
MODE="train"			# Either train or test.
RESULTS="best_model"	# Keep saving network model after each epoch.

# Train iterative PCRNet
$PY PCRNet.py -log $LOG_DIR -mode $MODE -results $RESULTS

WEIGHTS=$PWD/$LOG_DIR/$RESULTS".ckpt"
LOG_RESULTS="results_PCRNet"

# Test iterative PCRNet
$PY results_PCRNet.py -weights $WEIGHTS -log $LOG_RESULTS -noise $NOISE