import os
import time
import random

gpus = [1,7,8,9,2,3,4,5,6]*5
idx = 0

for i in range(5):
    for dim in [1]:
        gpu = gpus[idx]
        idx += 1
        os.system('python controller_PIDES.py --epoch 200 --bs 10 --greedy 0.1 --gpu '+str(gpu)+' --ckpt t_range_0_1_2ksearch_int20k_bd4kDim'+str(dim)+' --tree depth2_sub --random_step 3 --lr 0.002 --dim '+str(dim)+' --base 200000 --left -1 --right 1 --domainbs 20000 --bdbs 4000')
        time.sleep(500)