import os

gpus = [1]*5
idx = 0
dim = 2
for _ in range(5):
    gpu = gpus[idx]
    idx += 1
    os.system('screen python controller_PIDESv3.py --epoch 50 --bs 10 --greedy 0.1 --gpu '+str(gpu)+' --ckpt t_range_0_1_2ksearch_int20k_bd4kDim'+str(dim)+' --tree depth2_sub --random_step 3 --lr 0.001 --dim '+str(dim)+' --base 1000 --left 0 --right 1')