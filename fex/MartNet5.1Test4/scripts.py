import os
import time
import random

# Scripts here uses 3 loops, one for varying dimension, one for varying variance, and one for the number of trials of
# each set up.  Note that 'dim' refers to total problem dimensions, i.e. 1 time dimension + space dimensions.  So 'dim'
# is always dimension of x + 1 (since t always is 1-d).  In practice, we refer to the dimensions of x when talking
# about a 'n' dimensional problem i.e. if 'dim' is 11, that is the 10-d problem.

# NOTE: plotting and timing are mutually exclusive.  If you uncomment the lines in controller to make plots, timing
# results will be inaccurate.

gpus = [1]*200
idx = 0
for dim in [10]:
    for _ in range(1):
        gpu = gpus[idx]
        idx += 1
        thresh = 1/(dim**2)
        epochs = 100
        os.system('screen python controller_PIDESv2.py --epoch ' + str(epochs) + ' --bs 10 --greedy 0.1 --gpu ' + str(
        gpu) + ' --ckpt Dim' + str(dim) + ' --tree depth2_sub --random_step 3 --lr 0.001 --dim ' + str(dim) + ' --base 1000 --clustering_thresh ' + str(thresh))

