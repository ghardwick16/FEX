import os
import time
import random

gpus = [1,7,8,9,2,3,4,5,6]*5
idx = 0

for dim in [6]:
    gpu = gpus[idx]
    idx += 1
    os.system('python controller_PIDESv2.py --epoch 200 --bs 10 --greedy 0.1 --gpu ' + str(
            gpu) + ' --ckpt t_range_0_1_2ksearch_int20k_bd4kDim' + str(
            dim) + ' --tree depth2_sub --random_step 3 --lr 0.001 --dim ' + str(
            dim) + ' --base 1000 --left 0 --right 1')
    time.sleep(500)