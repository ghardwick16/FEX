import os
import time
import random

# Scripts here uses 3 loops, one for varying dimension, one for varying variance, and one for the number of trials of
# each set up.  Note that 'dim' refers to total problem dimensions, i.e. 1 time dimension + space dimensions.  So 'dim'
# is always dimension of x + 1 (since t always is 1-d).  In practice, we refer to the dimensions of x when talking
# about a 'n' dimensional problem i.e. if 'dim' is 11, that is the 10-d problem.

# NOTE: plotting and timing are mutually exclusive.  If you uncomment the lines in controller to make plots, timing
# results will be inaccurate.

gpus = [0]*200
idx = 0
for dim in [3, 5, 7, 9, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101]:
    for var in [.0001]:
        for _ in range(10):
            gpu = gpus[idx]
            idx += 1
            thresh = 1/(dim**2)
            epochs = 50
            os.system('screen python controller_PIDESv2.py --epoch ' + str(epochs) + ' --bs 10 --greedy 0.1 --gpu ' + str(
            gpu) + ' --ckpt t_range_0_1_2ksearch_int20k_bd4kDim' + str(
            dim) + ' --tree depth2_sub --random_step 3 --lr 0.001 --dim ' + str(
            dim) + ' --base 1000 --left 0 --right 1 --clustering_thresh ' + str(thresh) + ' --var ' + str(var))

