import os
import time
import random

gpus = [0]*200
idx = 0
for dim in [101]:
    for var in [.0001]:
        for _ in range(1):
            gpu = gpus[idx]
            idx += 1
            thresh = 1/(dim**2)
            #epochs = min(50, int(1/2*(dim-1) + 20))
            epochs = 50
            os.system('python controller_PIDESv2.py --epoch ' + str(epochs) + ' --bs 10 --greedy 0.1 --gpu ' + str(
            gpu) + ' --ckpt t_range_0_1_2ksearch_int20k_bd4kDim' + str(
            dim) + ' --tree depth2_sub --random_step 3 --lr 0.001 --dim ' + str(
            dim) + ' --base 1000 --left 0 --right 1 --clustering_thresh ' + str(thresh) + ' --var ' + str(var))

