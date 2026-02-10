import os
import time
import random

# Scripts here uses 2 loops, one for varying dimension, and one for the number of trials of
# each set up.

# NOTE: plotting and timing are mutually exclusive.  If you uncomment the lines in controller to make plots, timing
# results will be inaccurate.

gpus = [0]*200
idx = 0
for _ in range(1):
    for dim in [100]:
        gpu = gpus[idx]
        idx += 1
        thresh = 1/dim
        epochs = 75
        os.system('screen python controller_PIDESv2.py --epoch ' + str(epochs) + ' --bs 10 --greedy 0.1 --gpu ' + str(
        gpu) + ' --ckpt Dim' + str(dim) + ' --tree depth2_sub --random_step 3 --lr 0.001 --dim ' + str(dim) + ' --base 1000 --clustering_thresh ' + str(thresh))

