import os
import time
import random

gpus = [1]*10
idx = 0
dim = 2
for _ in range(5):
    gpu = gpus[idx]
    idx += 1
    os.system('screen python controller_PIDES.py --epoch 50 --bs 10 --greedy 0.1 --gpu '+str(gpu)+' --ckpt t_range_0_1_2ksearch_int20k_bd4kDim'+str(dim)+' --tree depth2_sub --random_step 3 --lr 0.002 --dim '+str(dim)+' --base 1000 --left 0 --right 1 --domainbs 1000 --bdbs 1000')