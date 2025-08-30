import os
import time
import random

# NOTE: plotting and timing are mutually exclusive.  If you uncomment the lines in controller to make plots, timing
# results will be inaccurate.

gpus = [0]*200
idx = 0
for dim in [10]:
    for _ in range(1):
        gpu = gpus[idx]
        idx += 1
        thresh = 10/(dim**2)
        epochs = 100
        os.system('screen python controller_PIDESv2.py --epoch ' + str(epochs) + ' --bs 10 --greedy 0.1 --gpu ' + str(
        gpu) + ' --ckpt Dim' + str(dim) + ' --tree depth1 --random_step 3 --lr 0.001 --dim ' + str(dim) + ' --base 1000 --clustering_thresh ' + str(thresh))

