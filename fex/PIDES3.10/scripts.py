import os
gpus = [1]*200
idx = 0
for dim in [101]:
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
