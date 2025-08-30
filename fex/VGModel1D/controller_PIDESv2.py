"""A module with NAS controller-related code."""
import torch
import torch.nn.functional as F
import numpy as np
import tools
import scipy
from utils import Logger, mkdir_p
import os
from computational_tree import BinaryTree
import function as func
import argparse
import random
import math
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hcluster
from torch.nn.parameter import Parameter, UninitializedParameter
from torch import Tensor
import torch.nn as nn
import itertools
import time

torch.set_default_dtype(torch.double)

parser = argparse.ArgumentParser(description='NAS')

parser.add_argument('--epoch', default=2000, type=int)
parser.add_argument('--bs', default=1, type=int)
parser.add_argument('--greedy', default=0, type=float)
parser.add_argument('--random_step', default=0, type=float)
parser.add_argument('--ckpt', default='', type=str)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--dim', default=5, type=int)
parser.add_argument('--tree', default='depth2', type=str)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--percentile', default=0.5, type=float)
parser.add_argument('--base', default=100, type=int)
parser.add_argument('--clustering_thresh', default=None, type=float)
parser.add_argument('--var', default=1e-4, type=float)
# parser.add_argument('--domainbs', default=1000, type=int)
# parser.add_argument('--bdbs', default=1000, type=int)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

unary = func.unary_functions
binary = func.binary_functions
unary_functions_str = func.unary_functions_str
unary_functions_str_leaf = func.unary_functions_str_leaf
binary_functions_str = func.binary_functions_str
thresh = args.clustering_thresh

dim = args.dim
num_samples = 5000
theta = -.4
sigma = .4
nu = .4

### Pre-compute a grid of FFT solutions to train FEX off of to see if FEX can produce a decent approximation to
### the true solution if given the true soluton exactly.  Essentially boils down to function-fitting

#FFT_sol = func.generate_FFT_sol(t_points=250, x_pts=250)
#Rescaling the solution to make for better learning
#FFT_sol[:, 1] *= 1/400
#FFT_sol[:, 2] *= 1/100
#torch.save(FFT_sol, 'FFT_data.pt')

FFT_data = torch.load('/export/users/ghardwic/RL_projects/FEX/fex/fex/VGModel1D/FFT_data.pt')
print(torch.max(FFT_data[:,0]), torch.max(FFT_data[:,1]), torch.max(FFT_data[:,2]))



lam_p, lam_n = func.precomputed_lams(theta, sigma, nu)
c = func.precomputed_expect(lam_p, lam_n, nu)

leaf_modes = []

if args.clustering_thresh:
    thresh = args.clustering_thresh
    clustering = True
else:
    clustering = False

class candidate(object):
    def __init__(self, action, leaf_action, expression, error, clusters, params):
        self.action = action
        self.leaf_action = leaf_action
        self.expression = expression
        self.error = error
        self.clusters = clusters
        self.params = params

class SaveBuffer(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.candidates = []

    def num_candidates(self):
        return len(self.candidates)

    def add_new(self, candidate):
        flag = 1
        action_idx = None
        for idx, old_candidate in enumerate(self.candidates):
            if candidate.action == old_candidate.action and candidate.error < old_candidate.error:  # 如果判断出来和之前的action一样的话，就不去做
                flag = 1
                action_idx = idx
                break
            elif candidate.action == old_candidate.action:
                flag = 0

        if flag == 1:
            if action_idx is not None:
                print(action_idx)
                self.candidates.pop(action_idx)
            self.candidates.append(candidate)
            self.candidates = sorted(self.candidates, key=lambda x: x.error)  # from small to large

        if len(self.candidates) > self.max_size:
            self.candidates.pop(-1)  # remove the last one


if args.tree == 'depth2':
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.insertRight('', True)

        tree.insertRight('', True)
        tree.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.insertRight('', True)
        return tree

elif args.tree == 'depth0':
    def basic_tree():
        tree = BinaryTree('', True)

        return tree


elif args.tree == 'depth1':
    def basic_tree():

        tree = BinaryTree('', False)
        tree.insertLeft('', True)
        tree.insertRight('', True)

        return tree

elif args.tree == 'depth2_rml':
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', True)

        tree.insertRight('', True)
        tree.rightChild.insertLeft('', True)

        return tree

elif args.tree == 'depth2_rmu':
    print('**************************rmu**************************')


    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.insertRight('', True)

        tree.insertRight('', False)
        tree.rightChild.insertLeft('', True)
        tree.rightChild.insertRight('', True)

        return tree

elif args.tree == 'depth2_rmu2':
    print('**************************rmu2**************************')


    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.insertRight('', True)

        tree.insertRight('', True)
        # tree.rightChild.insertLeft('', True)
        # tree.rightChild.insertRight('', True)

        return tree

elif args.tree == 'depth3':
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.leftChild.insertRight('', True)

        tree.leftChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.rightChild.leftChild.insertRight('', True)

        tree.insertRight('', True)
        tree.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.leftChild.insertRight('', True)

        tree.rightChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.rightChild.leftChild.insertRight('', True)
        return tree

elif args.tree == 'depth2_sub':
    print('**************************sub**************************')


    def basic_tree():
        tree = BinaryTree('', True)

        tree.insertLeft('', False)
        tree.leftChild.insertLeft('', True)
        tree.leftChild.insertRight('', True)

        # tree.rightChild.insertLeft('', True)
        # tree.rightChild.insertRight('', True)

        return tree

elif args.tree == 'depth2_sub_test':
    print('**************************sub**************************')


    def basic_tree():
        tree = BinaryTree('', True)

        tree.insertLeft('', False)
        tree.leftChild.insertLeft('', True)
        tree.leftChild.insertRight('', True)

        tree.leftChild.rightChild.insertRight('', True)
        tree.leftChild.leftChild.insertRight('', True)
        # tree.rightChild.insertLeft('', True)
        # tree.rightChild.insertRight('', True)

        return tree

structure = []


def inorder_structure(tree):
    global structure
    if tree:
        inorder_structure(tree.leftChild)
        structure.append(tree.is_unary)
        inorder_structure(tree.rightChild)

'''
inorder_structure(basic_tree())
print('tree structure', structure)

structure_choice = []
for is_unary in structure:
    if is_unary == True:
        structure_choice.append(len(unary))
    else:
        structure_choice.append(len(binary))
print('tree structure choices', structure_choice)
'''
if args.tree == 'depth1':
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.insertRight('', True)

        return tree

elif args.tree == 'depth2':
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.insertRight('', True)

        tree.insertRight('', True)
        tree.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.insertRight('', True)
        return tree

elif args.tree == 'depth3':
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.leftChild.insertRight('', True)

        tree.leftChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.rightChild.leftChild.insertRight('', True)

        tree.insertRight('', True)
        tree.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.leftChild.insertRight('', True)

        tree.rightChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.rightChild.leftChild.insertRight('', True)
        return tree

elif args.tree == 'depth4':
    def basic_tree():
        tree = BinaryTree('', False)

        # Left main branch
        # L L L L/R
        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertRight('', True)
        # L L L R L/R
        tree.leftChild.leftChild.leftChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertRight('', True)
        # L R L L/R
        tree.leftChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertRight('', True)
        # L R R L/R
        tree.leftChild.leftChild.rightChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertRight('', True)

        # Right main branch
        # Left main branch
        # R L L L/R
        tree.insertRight('', True)
        tree.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertRight('', True)
        # R L R L/R
        tree.rightChild.leftChild.leftChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertRight('', True)
        # R R L L/R
        tree.rightChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertRight('', True)
        # R R R L/R
        tree.rightChild.leftChild.rightChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertRight('', True)
        return tree

elif args.tree == 'depth5':
    def basic_tree():
        tree = BinaryTree('', False)

        # Left main branch

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', True) #LLLLL
        tree.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertRight('', True) #LLLLR
        tree.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertLeft('', True) #LLLRL
        tree.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertRight('', True) #LLLRR
        tree.leftChild.leftChild.leftChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertLeft('', True)  #LLRLL
        tree.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertRight('',True)  #LLRLR
        tree.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertLeft('',True)  #LLRRL
        tree.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertRight('', True)  #LLRRR
        tree.leftChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertLeft('',True)  #LRLLL
        tree.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertRight('',True)  #LRLLR
        tree.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertLeft('',True)  #LRLRL
        tree.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertRight('',True)  #LRLRR
        tree.leftChild.leftChild.rightChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertLeft('', True)  #LRRLL
        tree.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertRight('',True)  #LRRLR
        tree.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertLeft('',True)  #LRRRL
        tree.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertRight('',True)  #LRRRR

        #Right Main branch
        tree.insertRight('', True)
        tree.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertLeft('',True)  #RLLLL
        tree.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertRight('', True)  #RLLLR
        tree.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertLeft('',True)  #RLLRL
        tree.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertRight('',True)  #RLLRR
        tree.rightChild.leftChild.leftChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertLeft('',True)  #RLRLL
        tree.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertRight('',True)  #RLRLR
        tree.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertLeft('',True)  #RLRRL
        tree.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertRight('',True)  #RLRRR
        tree.rightChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', True)  #RRLLL
        tree.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertRight('',True)  #RRLLR
        tree.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertLeft('',True)  #RRLRL
        tree.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertRight('',True)  #RRLRR
        tree.rightChild.leftChild.rightChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertLeft('',True)  #RRRLL
        tree.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertRight('', True)  #RRRLR
        tree.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertLeft('',True)  #RRRRL
        tree.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertRight('',True)  #RRRRR
        return tree

elif args.tree == 'depth6':
    def basic_tree():
        tree = BinaryTree('', False)

        # Left main branch
        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', True) #LLLLL
        tree.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertRight('', True) #LLLLR
        tree.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertLeft('', True) #LLLRL
        tree.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertRight('', True) #LLLRR
        tree.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertLeft('', True)  # LLRLL
        tree.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertRight('',True)  # LLRLR
        tree.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertLeft('',True)  # LLRRL
        tree.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertRight('', True)  # LLRRR
        tree.leftChild.leftChild.leftChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertLeft('',True)  # LRLLL
        tree.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertRight('',True)  # LRLLR
        tree.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertLeft('',True)  # LRLRL
        tree.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertRight('',True)  # LRLRR
        tree.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertLeft('', True)  # LRRLL
        tree.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertRight('',True)  # LRRLR
        tree.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertLeft('',True)  # LRRRL
        tree.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertRight('',True)  # LRRRR

        #Right Main branch
        tree.leftChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertLeft('',True)  # RLLLL
        tree.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertRight('', True)  # RLLLR
        tree.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertLeft('',True)  # RLLRL
        tree.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertRight('',True)  # RLLRR
        tree.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertLeft('',True)  # RLRLL
        tree.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertRight('',True)  # RLRLR
        tree.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertLeft('',True)  # RLRRL
        tree.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertRight('',True)  # RLRRR
        tree.leftChild.leftChild.rightChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', True)  # RRLLL
        tree.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertRight('',True)  # RRLLR
        tree.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertLeft('',True)  # RRLRL
        tree.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertRight('',True)  # RRLRR
        tree.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertLeft('',True)  # RRRLL
        tree.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertRight('', True)  # RRRLR
        tree.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertLeft('',True)  # RRRRL
        tree.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertRight('',True)  # RRRRR

        # Right Branch
        tree.insertRight('', True)
        tree.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertLeft('',True)  # LLLLL
        tree.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertRight('',True)  # LLLLR
        tree.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertLeft('',True)  # LLLRL
        tree.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertRight('',True)  # LLLRR
        tree.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertLeft('',True)  # LLRLL
        tree.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertRight('',True)  # LLRLR
        tree.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertLeft('',True)  # LLRRL
        tree.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertRight('',True)  # LLRRR
        tree.rightChild.leftChild.leftChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertLeft('',True)  # LRLLL
        tree.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertRight('',True)  # LRLLR
        tree.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertLeft('',True)  # LRLRL
        tree.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertRight('',True)  # LRLRR
        tree.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertLeft('',True)  # LRRLL
        tree.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertRight('',True)  # LRRLR
        tree.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertLeft('',True)  # LRRRL
        tree.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertRight('',True)  # LRRRR

        # Right Main branch
        tree.rightChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertLeft('',True)  # RLLLL
        tree.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertRight('',True)  # RLLLR
        tree.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertLeft('',True)  # RLLRL
        tree.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertRight('',True)  # RLLRR
        tree.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertLeft('',True)  # RLRLL
        tree.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertRight('',True)  # RLRLR
        tree.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertLeft('',True)  # RLRRL
        tree.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertRight('',True)  # RLRRR
        tree.rightChild.leftChild.rightChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertLeft('',True)  # RRLLL
        tree.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.leftChild.leftChild.insertRight('',True)  # RRLLR
        tree.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertLeft('',True)  # RRLRL
        tree.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.rightChild.leftChild.insertRight('',True)  # RRLRR
        tree.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertLeft('',True)  # RRRLL
        tree.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.leftChild.leftChild.insertRight('',True)  # RRRLR
        tree.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertLeft('',True)  # RRRRL
        tree.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.rightChild.leftChild.insertRight('',True)  # RRRRR

        return tree

structure = []
leaves_index = []
leaves = 0
count = 0


def inorder_structure(tree):
    global structure, leaves, count, leaves_index
    if tree:
        inorder_structure(tree.leftChild)
        structure.append(tree.is_unary)
        if tree.leftChild is None and tree.rightChild is None:
            leaves = leaves + 1
            leaves_index.append(count)
        count = count + 1
        inorder_structure(tree.rightChild)


inorder_structure(basic_tree())

print('leaves index:', leaves_index)

print('tree structure:', structure, 'leaves num:', leaves)

structure_choice = []


for is_unary in structure:
    if is_unary == True:
        structure_choice.append(len(unary))
    else:
        structure_choice.append(len(binary))

print('tree structure choices', structure_choice)


def reset_params(trainable_tree):
    for idx, v in enumerate(trainable_tree.learnable_operator_set):
        if idx not in leaves_index:
            for modules in trainable_tree.learnable_operator_set[v]:
                for param in modules.parameters():
                    param.data.normal_(0.0, 0.1)
    for module in trainable_tree.linear:
        module.reset_parameters()
    for module in trainable_tree.input:
        module.reset_parameters()
        # v.data.fill_(0.01)

def save_parameters(trainable_tree):
    parameter_vals = [[], [], [], []]
    for module in trainable_tree.linear:
        x = torch.empty_like(module.weight.data).cuda()
        x[:] = module.weight.data[:]
        y = torch.empty_like(module.mult_weight.data).cuda()
        y[:] = module.mult_weight.data[:]
        z = torch.empty_like(module.bias.data).cuda()
        z[:] = module.bias.data[:]
        parameter_vals[0].append(x)
        parameter_vals[1].append(y)
        parameter_vals[2].append(z)
    for module in trainable_tree.input:
        x = torch.empty_like(module.a.data).cuda()
        x[:] = module.a.data[:]
        parameter_vals[3].append(x)
    return parameter_vals

def apply_parameters(trainable_tree, params):
    if params[0] is not None:
        for i in range(len(trainable_tree.linear)):
            trainable_tree.linear[i].weight.data = params[0][i]
    if params[1] is not None:
        for i in range(len(trainable_tree.linear)):
            trainable_tree.linear[i].mult_weight.data = params[1][i]
            #trainable_tree.linear[i].weight.requires_grad = True
    if params[2] is not None:
        for i in range(len(trainable_tree.linear)):
            trainable_tree.linear[i].bias.data = params[2][i]
    if params[3] is not None:
        for i in range(len(trainable_tree.input)):
            trainable_tree.input[i].a.data = params[2][i]
            #trainable_tree.input[i].a.requires_grad = True

    return trainable_tree


def inorder(tree, actions):
    global count
    if tree:
        inorder(tree.leftChild, actions)
        action = actions[count].item()
        if tree.is_unary:
            action = action
            tree.key = unary[action]
            # print(count, action, func.unary_functions_str[action])
        else:
            action = action
            tree.key = binary[action]
            # print(count, action, func.binary_functions_str[action])
        count = count + 1
        inorder(tree.rightChild, actions)


def inorder_visualize(tree, actions, trainable_tree):
    global count, leaves_cnt
    if tree:
        leftfun = inorder_visualize(tree.leftChild, actions, trainable_tree)
        action = actions[count].item()
        # print('123', tree.key)
        if tree.is_unary:  # and not tree.key.is_leave:
            if count not in leaves_index:
                midfun = unary_functions_str[action]
                a = trainable_tree.learnable_operator_set[count][action].a.item()
                b = trainable_tree.learnable_operator_set[count][action].b.item()
            else:
                midfun = unary_functions_str_leaf[action]
        else:
            midfun = binary_functions_str[action]

        count = count + 1
        rightfun = inorder_visualize(tree.rightChild, actions, trainable_tree)
        if leftfun is None and rightfun is None:
            w = []
            a = []
            if trainable_tree.clustering is not None:
                combined_a = trainable_tree.input[leaves_cnt].a[trainable_tree.input[leaves_cnt].clustering]
                combined_weight = trainable_tree.linear[leaves_cnt].weight[trainable_tree.linear[leaves_cnt].clustering]
                for i in range(dim):
                    w.append(combined_weight[0, i])
                    a.append(combined_a[0, i])
            else:
                for i in range(dim):
                    w.append(trainable_tree.linear[leaves_cnt].weight[i].item())
                    a.append(trainable_tree.input[leaves_cnt].a[i].item())
                # w2 = trainable_tree.linear[leaves_cnt].weight[0][1].item()
            bias = trainable_tree.linear[leaves_cnt].bias[0].item()
            mw = trainable_tree.linear[leaves_cnt].mult_weight[0].item()
            mode = trainable_tree.linear[leaves_cnt].mode
            leaves_cnt = leaves_cnt + 1
            ## -------------------------------------- input variable element wise  ----------------------------
            expression = ''
            for i in range(0, dim):
                # print(midfun)
                # mode is bool: True means the leaf is doing addition, False means the leaf is multiplying
                if mode:
                    x_expression = midfun.format(a[i], 'x' + str(i))
                    expression = expression + ('{:.6f}*{}' + '+').format(w[i], x_expression)
                else:
                    x_expression = midfun.format(a[i], 'x' + str(i))
                    if i != dim-1:
                        expression = expression + ('{}' + '*').format(x_expression)
                    else:
                        expression = expression + ('{}').format(x_expression)
            if mode:
                expression = expression + '{:.6f}'.format(bias)
            else:
                expression = ('{:.6f}' + '*').format(mw) + expression + ('+' + '{:.6f}').format(bias)
            expression = '(' + expression + ')'
            # print('visualize', count, leaves_cnt, action)
            return expression
        elif leftfun is not None and rightfun is None:
            if '(0)' in midfun or '(1)' in midfun:
                return midfun.format('{:.6f}'.format(a), '{:.6f}'.format(b))
            else:
                return midfun.format('{:.6f}'.format(a), leftfun, '{:.6f}'.format(b))
        elif tree.leftChild is None and tree.rightChild is not None:
            if '(0)' in midfun or '(1)' in midfun:
                return midfun.format('{:.6f}'.format(a), '{:.6f}'.format(b))
            else:
                return midfun.format('{:.6f}'.format(a), rightfun, '{:.6f}'.format(b))
        else:
            return midfun.format(leftfun, rightfun)
    else:
        return None

def get_function(actions):
    global count
    count = 0
    computation_tree = basic_tree()
    inorder(computation_tree, actions)
    count = 0  # 置零
    return computation_tree


def inorder_params(tree, actions, unary_choices):
    global count
    if tree:
        inorder_params(tree.leftChild, actions, unary_choices)
        action = actions[count].item()
        if tree.is_unary:
            action = action
            tree.key = unary_choices[count][action]
            # if tree.leftChild is None and tree.rightChild is None:
            #     print('inorder_params:', count, action)
            # print(count, action, func.unary_functions_str[action])
        else:
            action = action
            tree.key = unary_choices[count][len(unary) + action]
            # print(count, action, func.binary_functions_str[action], tree.key(torch.tensor([1]).cuda(), torch.tensor([2]).cuda()))
        count = count + 1
        inorder_params(tree.rightChild, actions, unary_choices)


def get_function_trainable_params(actions, unary_choices):
    global count
    count = 0
    computation_tree = basic_tree()
    inorder_params(computation_tree, actions, unary_choices)
    count = 0  # 置零
    return computation_tree

class leaf(nn.Module):
    ## Clustering is a list that gives numbers to each cluster of weights i.e. [0, 0, 1, 1, 1] means that
    ## x0, x1 have same weight and x2, x3, x4 have same weight

    ## There are two 'modes', addition, and multiplication.  In the first case we sum the inputs (i.e.
    ## a1*sin(x1) + a2*sin(x2) + ... + b) in the second we compute the product (i.e. a1*sin(x1)*sin(x2)* ... + b)

    def __init__(self, in_features: int, out_features: int, leaf_index: int, bias: bool = True,
                 device=None, dtype=None, clustering=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.mode = True
        self.leaf_index = leaf_index
        self.clustering = clustering
        self.in_features = in_features
        self.mult_weight = nn.Parameter(torch.empty(1).cuda())
        out_features = 1  # this is only to be used for leaves in FEX, so out features is always 1
        if clustering is not None:
            self.weight = Parameter(torch.empty(int(torch.max(clustering) + 1), **factory_kwargs))
        else:
            self.weight = Parameter(torch.empty(in_features, **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight.data.fill_(1)
        self.mult_weight.data.fill_(1)
        if self.bias:
            self.bias.data.fill_(0)

    def set_mode(self, op_flag: bool) -> None:
        # True means addition, false means multiplication
        self.mode = op_flag

    def forward(self, input: Tensor) -> Tensor:
        if self.mode:
            if self.clustering is not None:
                out = torch.sum(input * self.weight[self.clustering].repeat(input.shape[0], 1), dim=1) + self.bias
            else:
                if len(input.shape) != 2:
                    out = torch.sum(input * self.weight.repeat(input.shape[0], input.shape[1], 1), dim=-1) + self.bias
                else:
                    out = torch.sum(input * self.weight.repeat(input.shape[0], 1), dim=-1) + self.bias
            #print(f'shape of leaf output with addition: {out.shape}')
            return out
        else:
            out = self.mult_weight * torch.prod(input, dim=-1) + self.bias
            return out.cuda()
            #print(f'shape of leaf output with multiplication: {out.shape}')
            #return self.mult_weight * torch.prod(input, dim=1).cuda() + self.bias


class input_layer(nn.Module):
    def __init__(self, dim, clustering = None):
        super(input_layer, self).__init__()
        self.dim = dim
        self.clustering = clustering
        if clustering is not None:
            self.a = nn.Parameter(torch.empty(torch.max(clustering) + 1)).cuda()
        else:
            self.a = nn.Parameter(torch.empty(dim).cuda())
        self.a.data.fill_(1)
        #self.b = nn.Parameter(torch.Tensor(dim).cuda())
        #self.b.data.fill_(0)

    def forward(self, x):
        if self.clustering is not None:
            return self.a[self.clustering] * x
        else:
            return self.a * x

    def reset_parameters(self):
        self.a.data.fill_(1)
        # self.b.data.fill_(0)

class unary_operation(nn.Module):
    def __init__(self, operator, is_leave):
        super(unary_operation, self).__init__()
        self.unary = operator
        if not is_leave:
            self.a = nn.Parameter(torch.Tensor(1).cuda())
            self.a.data.fill_(1)
            self.b = nn.Parameter(torch.Tensor(1).cuda())
            self.b.data.fill_(0)
        self.is_leave = is_leave

    def forward(self, x):
        if self.is_leave:
            return self.unary(x)
        else:
            return self.a * self.unary(x) + self.b



class binary_operation(nn.Module):
    def __init__(self, operator):
        super(binary_operation, self).__init__()
        self.binary = operator
        # self.a = nn.Parameter(torch.Tensor(1).cuda())
        # # self.a.data.fill_(0)
        # nn.init.normal_(self.a.data, 0, 0.5)
        # self.b = nn.Parameter(torch.Tensor(1).cuda())
        # # self.b.data.fill_(0)
        # nn.init.normal_(self.b.data, 0, 0.5)

    def forward(self, x, y):
        # return self.binary(torch.sigmoid(self.a)*x, torch.sigmoid(self.b)*y)
        # print('unary', self.a, self.b)
        return self.binary(x, y)


leaves_cnt = 0


def compute_by_tree(tree, linear, input, x):
    # judge whether a emtpy tree, if yes, that means the leaves and call the unary operation
    if tree.leftChild == None and tree.rightChild == None:  # leaf node
        global leaves_cnt
        input = input[leaves_cnt]
        transformation = linear[leaves_cnt]
        leaves_cnt = leaves_cnt + 1
        return transformation(tree.key(input(x)))
    elif tree.leftChild is None and tree.rightChild is not None:
        return tree.key(compute_by_tree(tree.rightChild, linear, input, x))
    elif tree.leftChild is not None and tree.rightChild is None:
        return tree.key(compute_by_tree(tree.leftChild, linear, input, x))
    else:
        return tree.key(compute_by_tree(tree.leftChild, linear, input, x), compute_by_tree(tree.rightChild, linear, input, x))


## Clustering is a list of lists, the ith list corresponding to the clustering of the ith leaf.  If no clustering is
## desired for a given leaf, then that leaf's clustering list should just be range(dims) (i.e. [0,1,2,...,d])

## If no clustering is wanted at all, then pass in no argument for clustering and learnable_computation_tree will
## behave as usual
class learnable_compuatation_tree(nn.Module):
    def __init__(self, clustering=None):
        super(learnable_compuatation_tree, self).__init__()
        self.learnable_operator_set = {}
        self.clustering = clustering
        for i in range(len(structure)):
            self.learnable_operator_set[i] = []
            is_leave = i in leaves_index
            for j in range(len(unary)):
                self.learnable_operator_set[i].append(unary_operation(unary[j], is_leave))
            for j in range(len(binary)):
                self.learnable_operator_set[i].append(binary_operation(binary[j]))
        self.linear = []
        self.input = []
        for num, i in enumerate(range(leaves)):
            if clustering:
                linear_module = leaf(dim, 1, bias=True, leaf_index = i, clustering=self.clustering[0][i]).cuda()  # set only one variable
                input_module = input_layer(dim, clustering=self.clustering[1][i]).cuda()
            else:
                linear_module = leaf(dim, 1, leaf_index = i, bias=True).cuda()
                input_module = input_layer(dim, clustering=None).cuda()
            linear_module.weight.data.normal_(0, 1 / math.sqrt(dim))
            # linear_module.weight.data[0, num%2] = 1
            linear_module.bias.data.fill_(0)
            self.linear.append(linear_module)
            self.input.append(input_module)

    def forward(self, x, bs_action):
        # print(len(bs_action))
        global leaves_cnt
        leaves_cnt = 0
        function = lambda y: compute_by_tree(get_function_trainable_params(bs_action, self.learnable_operator_set),
                                             self.linear, self.input, y)
        out = function(x)
        leaves_cnt = 0
        return out


class Controller(torch.nn.Module):
    """Based on
    https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    Base the controller RNN on the GRU from:
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
    """

    def __init__(self):
        torch.nn.Module.__init__(self)

        self.softmax_temperature = 5.0
        self.tanh_c = 2.5
        self.mode = True

        self.input_size = 20
        self.hidden_size = 50
        ## we need to add 2 outputs for each leaf since now their 'mode' is learnable and there are two modes to
        ## choose from
        self.output_size = sum(structure_choice) + len(leaves_index)*2
        #self.output_size = sum(structure_choice)


        self._fc_controller = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.output_size))

    def forward(self, x):
        logits = self._fc_controller(x)

        logits /= self.softmax_temperature

        # exploration # ??
        if self.mode == 'train':
            logits = (self.tanh_c * F.tanh(logits))

        return logits

    def sample(self, batch_size=1, step=0):
        """Samples a set of `args.num_blocks` many computational nodes from the
        controller, where each node is made up of an activation function, and
        each node except the last also includes a previous node.
        """

        # [B, L, H]
        inputs = torch.zeros(batch_size, self.input_size).cuda()

        log_probs = []
        actions = []
        leaf_actions = []
        leaf_log_probs = []

        total_logits = self.forward(inputs)
        tree_logits = total_logits[:,:sum(structure_choice)]
        leaf_logits = total_logits[:,sum(structure_choice):]

        cumsum = np.cumsum([0] + structure_choice)
        for idx in range(len(leaves_index)):
            logits = leaf_logits[:, 2*idx:2*(idx+1)]

            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            # print(probs)
            if step >= args.random_step:
                leaf_action = probs.multinomial(num_samples=1).data
            else:
                leaf_action = torch.randint(0, 2, size=(batch_size, 1)).cuda()
            # print('old', action)
            if args.greedy != 0:
                for k in range(args.bs):
                    if np.random.rand(1) < args.greedy:
                        choice = random.choices(range(2), k=1)
                        leaf_action[k] = choice[0]

            selected_log_prob = log_prob.gather(
                1, tools.get_variable(leaf_action, requires_grad=False))

            leaf_log_probs.append(selected_log_prob[:, 0:1])
            leaf_actions.append(leaf_action[:, 0:1])

        for idx in range(len(structure_choice)):
            logits = tree_logits[:, cumsum[idx]:cumsum[idx + 1]]

            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            # print(probs)
            if step >= args.random_step:
                action = probs.multinomial(num_samples=1).data
            else:
                action = torch.randint(0, structure_choice[idx], size=(batch_size, 1)).cuda()
            # print('old', action)
            if args.greedy != 0:
                for k in range(args.bs):
                    if np.random.rand(1) < args.greedy:
                        choice = random.choices(range(structure_choice[idx]), k=1)
                        action[k] = choice[0]
            # print('new', action)
            selected_log_prob = log_prob.gather(
                1, tools.get_variable(action, requires_grad=False))

            log_probs.append(selected_log_prob[:, 0:1])
            actions.append(action[:, 0:1])

        log_probs = torch.cat(log_probs, dim=1)  # 3*18
        # print(actions)
        return actions, log_probs, leaf_actions, leaf_log_probs

    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.controller_hid)
        return (tools.get_variable(zeros, True, requires_grad=False),
                tools.get_variable(zeros.clone(), True, requires_grad=False))

def get_hes_flag(action, leaf_modes):
    for _, mode in enumerate(leaf_modes):
        if mode:
            return False
    idx = [i for i, b in enumerate(structure) if not b]
    for i in idx:
        if action[i].item() == 1:
            return False
    return True

def get_reward(bs, actions, leaf_actions, learnable_tree, tree_optim):
    regression_errors = []
    formulas = []
    batch_size = bs
    # bin_idxs = [i for i, b in enumerate(structure) if not b]
    params = []
    global count, leaves_cnt
    for bs_idx in range(batch_size):
        bs_action = [v[bs_idx] for v in actions]
        leaf_modes = [v[bs_idx][0].item() for v in leaf_actions]
        reset_params(learnable_tree)
        fast_hes = get_hes_flag(bs_action, leaf_modes)
        #fast_hes = False
        cand_func = (learnable_tree, bs_action)
        tree_params = []
        for idx, v in enumerate(learnable_tree.learnable_operator_set):
            if idx not in leaves_index:
                for modules in learnable_tree.learnable_operator_set[v]:
                    for param in modules.parameters():
                        tree_params.append(param)
        for linear in learnable_tree.linear:
            #for param in linear.parameters():
            #    tree_params.append(param)
            linear.set_mode(leaf_modes[linear.leaf_index])
            tree_params.append(linear.weight)
            tree_params.append(linear.mult_weight)
            tree_params.append(linear.bias)
        for input in learnable_tree.input:
            tree_params.append(input.a)
        tree_optim = torch.optim.Adam(tree_params, lr=1e-3)
        error_hist = []
        x = func.get_pts(num_samples, dim)
        for _ in range(50):
            tree_optim.zero_grad()
            loss = func.get_loss(cand_func, lam_p, lam_n, nu, c, x)
            #idxs = torch.randperm(FFT_data.shape[0])
            #data = FFT_data[idxs[:num_samples]]
            #loss = func.get_loss(cand_func, data)
            #loss = func.get_loss(cand_func, FFT_data)
            error_hist.append(loss.item())
            loss.backward(retain_graph=True)
            #torch.nn.utils.clip_grad_norm_(tree_params, 1)

            tree_optim.step()
        tree_optim = torch.optim.LBFGS(tree_params, lr=1, max_iter=30)
        print('---------------------------------- batch idx {} -------------------------------------'.format(bs_idx))
        def closure():
            tree_optim.zero_grad()
            loss = func.get_loss(cand_func, lam_p, lam_n, nu, c, x)
            #idxs = torch.randperm(FFT_data.shape[0])
            #data = FFT_data[idxs[:num_samples]]
            #loss = func.get_loss(cand_func, data)
            #loss = func.get_loss(cand_func, FFT_data)
            error_hist.append(loss.item())
            print('loss before: ', loss.item())
            loss.backward(retain_graph=True)

            return loss
        tree_optim.step(closure)
        #bw = tree_params[62].item()
        params.append(save_parameters(learnable_tree))
        #x_t = func.get_pts(num_paths, dim - 1)
        #loss = func.get_loss(cand_func, func.true_solution, x_t, sigma)
        #print(f'loss after, {loss}')
        #error_hist.append(loss.item())
        #print('min: ', min(error_hist))

        for param in tree_params:
            if torch.any(torch.isnan(param)):
                error = 10e15
                break
        else:
            error = min(error_hist)

        regression_errors.append(error)

        count = 0
        leaves_cnt = 0
        formula = inorder_visualize(basic_tree(), bs_action, learnable_tree)
        count = 0
        leaves_cnt = 0
        formulas.append(formula)


    return regression_errors, formulas, params


def discount(x, amount):
    return scipy.signal.lfilter([1], [1, -amount], x[::-1], axis=0)[::-1]


# restructure and re-tune a coarse-tuned tree
def restruct_and_tune(action, leaf_action, pre_its, tune_its, thresh, saved_params):
    clusters = [[],[]]
    new_saved_params = [[],[],[]]
    leaf_modes = [v[0].item() for v in leaf_action]
    fast_hes = get_hes_flag(action, leaf_modes)
    for i in range(len(leaves_index)):
        input_p = saved_params[3][i].unsqueeze(0)
        leaf_p = saved_params[0][i].unsqueeze(0)
        if torch.any(torch.isinf(input_p)) or torch.any(torch.isnan(input_p)):
            print('Could Not Cluster')
            return None, 1e10, None, None
        if torch.any(torch.isinf(leaf_p)) or torch.any(torch.isnan(leaf_p)):
            print('Could Not Cluster')
            return None, 1e10, None, None
        input_cluster = torch.tensor(
            [hcluster.fclusterdata(np.expand_dims(input_p[0, :].cpu().detach().numpy(), -1), thresh,
                                   criterion="distance") - 1]).int().cuda()
        leaf_cluster = torch.tensor(
            [hcluster.fclusterdata(np.expand_dims(leaf_p[0, :].cpu().detach().numpy(), -1), thresh,
                                    criterion="distance") - 1]).int().cuda()
        clusters[0].append(leaf_cluster)
        clusters[1].append(input_cluster)
        new_leaf_p = torch.empty(torch.max(leaf_cluster) + 1).cuda()
        new_input_p = torch.empty(torch.max(input_cluster) + 1).cuda()
        #new_leaf_mw = torch.empty(1).cuda()
        for j in range(torch.max(leaf_cluster) + 1):
            new_leaf_p[j] = torch.mean(leaf_p[leaf_cluster == j])
        for j in range(torch.max(input_cluster) + 1):
            new_input_p[j] = torch.mean(input_p[input_cluster == j])
        new_saved_params[0].append(new_leaf_p)
        new_saved_params[2].append(new_input_p)
        new_saved_params[1].append(saved_params[1][i].cuda())

    new_tree = learnable_compuatation_tree(clustering=clusters)

    for i, input in enumerate(new_tree.input):
        input.a = nn.Parameter(new_saved_params[2][i])
    for i, linear in enumerate(new_tree.linear):
        linear.weight = nn.Parameter(new_saved_params[0][i])
        linear.mult_weight = nn.Parameter(new_saved_params[1][i])
        linear.bias = nn.Parameter(saved_params[2][i])

    cand_func = (new_tree, action)
    params = []
    leaf_modes = [v[0].item() for v in leaf_action]
    for idx, v in enumerate(new_tree.learnable_operator_set):
        if idx not in leaves_index:
            for modules in new_tree.learnable_operator_set[v]:
                for param in modules.parameters():
                    params.append(param)
    for linear in new_tree.linear:
        linear.set_mode(leaf_modes[linear.leaf_index])
        params.append(linear.weight)
        params.append(linear.mult_weight)
        params.append(linear.bias)
    for module in new_tree.input:
        params.append(module.a)

    #Experimental Coarse tune (again):
    x = func.get_pts(num_samples, dim)

    tree_optim = torch.optim.Adam(params, lr=1e-5)
    for _ in range(50):
        tree_optim.zero_grad()
        loss = func.get_loss(cand_func, lam_p, lam_n, nu, c, x)
        #loss = func.get_loss(cand_func, FFT_data)
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(params, 1)
        tree_optim.step()
    tree_optim = torch.optim.LBFGS(params, lr=1, max_iter=50)

    def closure():
        tree_optim.zero_grad()
        loss = func.get_loss(cand_func, lam_p, lam_n, nu, c, x)
        #loss = func.get_loss(cand_func, FFT_data)
        loss.backward(retain_graph=True)
        return loss
    tree_optim.step(closure)
    '''
    #perform first steps of fine tuning:
    tree_optim = torch.optim.Adam(params, lr=1e-3)
    x = func.get_pts(num_samples, centers, radii, dim)
    bd_x = func.get_bdry_pts(int(num_samples/2), int(num_samples/250), centers, radii, dim)
    for _ in range(tune_its):
        tree_optim.zero_grad()
        loss = func.get_loss(cand_func, x, bd_x, fast_hes)
        loss.backward(retain_graph = True)
        tree_optim.step()
    '''

    final_loss = loss.item()
    if final_loss != final_loss:
        final_loss = 1e15

    new_formula = inorder_visualize(basic_tree(), action, new_tree)
    new_params = save_parameters(new_tree)

    return clusters, final_loss, new_formula, new_params


def best_error(best_action, learnable_tree):
    # t = torch.rand(args.domainbs, 1).cuda()
    # x1 = (torch.rand(args.domainbs, args.dim - 1).cuda()) * (args.right - args.left) + args.left
    # x = torch.cat((t, x1), 1)
    # x.requires_grad = True
    x = func.get_pts(num_samples, dim)
    bs_action = best_action

    cand_func = (learnable_tree, bs_action)

    # bd_pts = get_boundary(args.bdbs, dim)
    # bc_true = func.true_solution(bd_pts)
    # bd_nn = learnable_tree(bd_pts, bs_action)
    # bd_error = torch.nn.functional.mse_loss(bc_true, bd_nn)
    # function_error = torch.nn.functional.mse_loss(func.LHS_pde(lhs_func, x), func.RHS_pde(x))
    #regression_error = func.get_loss(cand_func, x, bd_x, get_hes_flag(best_action))
    #regression_error = func.get_loss(cand_func, FFT_data)
    regression_error = func.get_loss(cand_func, lam_p, lam_n, nu, c, x)

    print(f'error: {regression_error}')

    return regression_error


def train_controller(Controller, Controller_optim, trainable_tree, tree_params, hyperparams):
    thresh = args.clustering_thresh
    ### obtain a new file name ###
    file_name = os.path.join(hyperparams['checkpoint'], 'log{}.txt')
    file_idx = 0
    while os.path.isfile(file_name.format(file_idx)):
        file_idx += 1
    file_name = file_name.format(file_idx)
    logger = Logger(file_name, title='')
    logger.set_names(['iteration', 'loss', 'baseline', 'error', 'formula', 'error'])

    model = Controller
    model.train()

    baseline = None

    bs = args.bs
    smallest_error = float('inf')

    candidates = SaveBuffer(10)

    tree_optim = None
    for step in range(hyperparams['controller_max_step']):
        # sample models
        actions, log_probs, leaf_actions, leaf_log_probs = controller.sample(batch_size=bs, step=step)
        binary_code = ''
        for action in actions:
            binary_code = binary_code + str(action[0].item())
        rewards, formulas, batch_params = get_reward(bs, actions, leaf_actions, trainable_tree, tree_optim)
        rewards = torch.cuda.FloatTensor(rewards).view(-1, 1)
        rewards[rewards != rewards] = 1e15

        error = rewards
        batch_min_idx = torch.argmin(error)
        batch_min_action = [v[batch_min_idx] for v in actions]
        batch_min_leaf_action = [v[batch_min_idx] for v in leaf_actions]
        best_params = batch_params[batch_min_idx]
        batch_smallest = error.min().item()
        if thresh:
            print(
                '********************************************************************************************************')
            print(f'previous best loss: {batch_smallest}, previous best formula: {formulas[batch_min_idx]}')
            clusters, new_best_loss, new_formula, new_params = restruct_and_tune(batch_min_action,
                                                                                 batch_min_leaf_action, pre_its=100,
                                                                                 tune_its=100 + dim * 4, thresh=thresh,
                                                                                 saved_params=best_params)
            print(f'clustered loss: {new_best_loss}, clustered formula: {new_formula}')
            print(
                '********************************************************************************************************')
            if new_best_loss < batch_smallest:
                batch_smallest = new_best_loss
                formulas[batch_min_idx] = new_formula
                best_params = new_params
            else:
                clusters = None
        else:
            clusters = None
        # discount
        if 1 > hyperparams['discount'] > 0:
            rewards = discount(rewards, hyperparams['discount'])

        base = args.base
        rewards[rewards > base] = base
        rewards[rewards != rewards] = 1e15
        rewards = 1 / (1 + torch.sqrt(rewards))
        # print(f'rewards: {rewards}')
        batch_best_formula = formulas[batch_min_idx]
        if batch_smallest != batch_smallest:
            batch_smallest = 1e15
        candidates.add_new(
            candidate(action=batch_min_action, leaf_action=batch_min_leaf_action, expression=batch_best_formula,
                      error=batch_smallest, clusters=clusters, params=best_params))

        for candidate_ in candidates.candidates:
            print('error:{} action:{} formula:{}'.format(candidate_.error, [v.item() for v in candidate_.action],
                                                         candidate_.expression))

        # moving average baseline
        if baseline is None:
            baseline = (rewards).mean()
        else:
            decay = hyperparams['ema_baseline_decay']
            baseline = decay * baseline + (1 - decay) * (rewards).mean()

        argsort = torch.argsort(rewards.squeeze(1), descending=True)
        # print(error, argsort)
        # print(rewards.size(), rewards.squeeze(1), torch.argsort(rewards.squeeze(1)), rewards[argsort])
        # policy loss
        num = int(args.bs * args.percentile)
        rewards_sort = rewards[argsort]
        adv = rewards_sort - rewards_sort[num:num + 1, 0:]  # - baseline
        # print(error, argsort, rewards_sort, adv)
        log_probs_sort = log_probs[argsort]
        # print('adv', adv)
        loss = -log_probs_sort[:num] * tools.get_variable(adv[:num], True, requires_grad=False)
        loss = (loss.sum(1)).mean()

        # update
        controller_optim.zero_grad()
        loss.backward()

        if hyperparams['controller_grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(),
                                          hyperparams['controller_grad_clip'])
        Controller_optim.step()

        min_error = batch_smallest
        # print('******************** ', min_error)
        if smallest_error > min_error:
            smallest_error = min_error
            min_action = [v[batch_min_idx] for v in actions]
            best_formula = formulas[batch_min_idx]

        log = 'Step: {step}| Loss: {loss:.4f}| Action: {act} |Baseline: {base:.4f}| ' \
              'Reward {re:.4f} | {error:.8f} {formula}'.format(loss=loss.item(), base=baseline, act=binary_code,
                                                               re=(rewards).mean(), step=step, formula=best_formula,
                                                               error=smallest_error)
        print(
            '********************************************************************************************************')
        print(log)
        print(
            '********************************************************************************************************')
        if (step + 1) % 1 == 0:
            logger.append([step + 1, loss.item(), baseline, rewards.mean(), smallest_error, best_formula])
    candidates.candidates = sorted(candidates.candidates, key=lambda x: x.error)
    for candidate_ in candidates.candidates:
        print('error:{} action:{} formula:{}'.format(candidate_.error, [v.item() for v in candidate_.action],
                                                     candidate_.expression))
        #print(f'biases: {candidate_.params[2]}')
        action_string = ''
        for v in candidate_.action:
            action_string += str(v.item()) + '-'
        logger.append([666, 0, 0, action_string, candidate_.error, candidate_.expression])
        # logger.append([666, 0, 0, 0, candidate_.error.item(), candidate_.expression])
    finetune = 50000
    global count, leaves_cnt
    i = 0
    stopping = False
    cand_number = 0
    candidates.candidates = sorted(candidates.candidates, key=lambda x: x.error)
    while not stopping and cand_number < len(candidates.candidates):
        for candidate_ in candidates.candidates:
            trainable_tree = learnable_compuatation_tree(candidate_.clusters)
            leaf_modes = [v[0].item() for v in candidate_.leaf_action]
            # apply_parameters(trainable_tree, candidate_.params
            for i, input in enumerate(trainable_tree.input):
                if not torch.any(torch.isnan(candidate_.params[3][i])):
                    input.a = nn.Parameter(candidate_.params[3][i].cuda())
            for i, linear in enumerate(trainable_tree.linear):
                if not torch.any(torch.isnan(candidate_.params[0][i])):
                    linear.weight = nn.Parameter(candidate_.params[0][i].cuda())
                if not torch.any(torch.isnan(candidate_.params[1][i])):
                    linear.mult_weight = nn.Parameter(candidate_.params[1][i].cuda())
                if not torch.any(torch.isnan(candidate_.params[2][i])):
                    linear.bias = nn.Parameter(candidate_.params[2][i].cuda())
            params = []
            for idx, v in enumerate(trainable_tree.learnable_operator_set):
                if idx not in leaves_index:
                    for modules in trainable_tree.learnable_operator_set[v]:
                        for param in modules.parameters():
                            params.append(param)
            for linear in trainable_tree.linear:
                # for param in linear.parameters():
                #    tree_params.append(param)
                linear.set_mode(leaf_modes[linear.leaf_index])
                params.append(linear.weight)
                params.append(linear.mult_weight)
                params.append(linear.bias)
            for input in trainable_tree.input:
                params.append(input.a)

            #print(candidate_.params[4])
            #params[62] = nn.Parameter(candidate_.params[4][0].cuda())

            fast_hes = get_hes_flag(candidate_.action, leaf_modes)
            cand_func = (trainable_tree, candidate_.action)

            '''
            ## COARSE TUNE TO SPEED UP FINE-TUNE:

            x = func.get_pts(num_samples, centers, radii, dim)
            bd_x = func.get_bdry_pts(int(num_samples/2), int(num_samples/250), centers, radii, dim)

            tree_optim = torch.optim.Adam(params, lr=1e-3)
            for _ in range(50):
                loss = func.get_loss(cand_func, x, bd_x, fast_hes)
                tree_optim.zero_grad()
                loss.backward(retain_graph=True)
                tree_optim.step()
            
            tree_optim = torch.optim.LBFGS(params, lr=1, max_iter=50)
            def closure():
                tree_optim.zero_grad()
                loss = func.get_loss(cand_func, x, bd_x, fast_hes)
                loss.backward(retain_graph=True)
                return loss

            tree_optim.step(closure)
            ## END COARSE TUNE CODE
            '''
            relatives = []
            tree_optim = torch.optim.Adam(params, lr=1e-3, amsgrad=True)
            error_list = []
            for current_iter in range(finetune):
                #for idx, param in enumerate(params):
                #    print(idx, param)
                #print('{:.6f}'.format(params[62].item()))
                tree_optim.zero_grad()
                x = func.get_pts(num_samples, dim)
                #idxs = torch.randperm(FFT_data.shape[0])
                #data = FFT_data[idxs[:num_samples]]
                #error = func.get_loss(cand_func, data)
                error = func.get_loss(cand_func, lam_p, lam_n, nu, c, x)
                # error = best_error(action, trainable_tree)
                error_list.append(error.item())
                error.backward()
                torch.nn.utils.clip_grad_norm_(params, 1)
                tree_optim.step()
                # for para in params:
                #     if para is not None:
                #         print(para.grad)

                count = 0
                leaves_cnt = 0
                formula = inorder_visualize(basic_tree(), candidate_.action, trainable_tree)
                if current_iter == 0:
                    print('====================================================================')
                    print('Starting Point:')
                    print(formula)
                    print('====================================================================')
                    logger.append([current_iter, 0, 0, 0, error.item(), formula])
                # formula = inorder_visualize(basic_tree(), action, trainable_tree)
                leaves_cnt = 0
                count = 0
                suffix = 'Finetune-- Iter {current_iter} Error {error} Formula {formula}'.format(
                    current_iter=current_iter, error=error, formula=formula)
                if (current_iter + 1) % 100 == 0:
                    logger.append([current_iter, 0, 0, 0, error.item(), formula])
                # if smallest_error <= 1e-10:
                #     logger.append([current_iter, 0, 0, 0, error.item(), formula])
                #     return
                # cosine_lr(tree_optim, 1e-3, current_iter, finetune)
                print(suffix)
                '''
                if (current_iter + 1) % 10 == 0:
                    _, relative, _ = func.get_errors(trainable_tree, candidate_.action, args.dim)
                    relatives.append(relative.item())
                '''
                # adding a halt condition when the error is of the same order as machine epsilon (or the last 100 have been close)

                if current_iter > 100:
                    if sum(error_list[(current_iter - 5):]) / len(error_list[(current_iter - 5):]) < 1.1e-14:
                        stopping = True

                '''
                if current_iter > 1500:
                    stopping = True
                '''
                # if current_iter == finetune - 1 or stopping:
                #    relative_l2, relative, mse = func.get_errors(trainable_tree, action, args.dim - 1)
                #    logger.append([f'RL2: {relative_l2}', f'REL: {relative}', f'MSE: {mse}', f'Loss: {error.item()}', 0, 0])
                if current_iter == finetune - 1 or stopping or current_iter == 0:
                    cand_function = lambda x: trainable_tree(x, candidate_.action)
                    relative_error = 0
                    mse = 0
                    l2_rel = 0
                    for i in range(1000):
                        x = func.get_pts(num_samples, dim)
                        c_x = cand_function(x).squeeze()
                        t_x = func.true_solution(x)
                        # print(c_x.shape)
                        # print(t_x.shape)
                        relative_error += 1 / 1000 * torch.mean(abs(c_x - t_x) / abs(t_x))
                        l2_rel += 1 / 1000 * torch.mean(torch.sqrt((c_x - t_x) ** 2) / torch.sqrt(t_x ** 2))
                        mse += 1 / 1000 * torch.mean((c_x - t_x) ** 2)
                    logger.append(
                        [f'REL: {relative_error}', f'L2_REL: {l2_rel}', f'MSE: {mse}', f'Loss: {error.item()}', 0, 0])
                    print(f'REL: {relative_error}', f'L2_Rel: {l2_rel}', f'MSE: {mse}', f'Loss: {error.item()}')
                if stopping:
                    break
            '''
            plt.figure(cand_number)
            plt.plot(error_list)
            plt.yscale('log')
            plt.xlabel('Iteration', fontsize = 14)
            plt.ylabel('Loss (log scale)', fontsize = 14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            # title = 'Equation (3.9): ' + str(args.dim - 1) + ' Dimensional Problem - Finetune Loss Plot'
            # plt.title(title)
            name = 'cand' + str(cand_number) + '_dims' + str(args.dim) + '_var' + str(args.var) + '_plot.png'
            plt.savefig(name, format='png', bbox_inches='tight')
            # cand_number += 1

            plt.figure(20 + cand_number)
            plt.plot(relatives)
            plt.yscale('log')
            plt.xlabel('Iteration (in hundreds)', fontsize = 14)
            plt.ylabel('Relative Error (log scale)', fontsize = 14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            # title = 'Equation (3.9): ' + str(args.dim - 1) + ' Dimensional Problem - Relative Error Plot'
            # plt.title(title)
            name = 'cand' + str(cand_number) + '_dims' + str(args.dim) + '_var' + str(args.var) + '_relative_plot.png'
            plt.savefig(name, format='png', bbox_inches='tight')
            '''
            cand_number += 1
            if stopping:
                break


def cosine_lr(opt, base_lr, e, epochs):
    lr = 0.5 * base_lr * (math.cos(math.pi * e / epochs) + 1)
    for param_group in opt.param_groups:
        param_group["lr"] = lr
    return lr


if __name__ == '__main__':
    start = time.time()
    controller = Controller().cuda()
    hyperparams = {}
    hyperparams['controller_max_step'] = args.epoch
    hyperparams['discount'] = 1.0
    hyperparams['ema_baseline_decay'] = 0.95
    hyperparams['controller_lr'] = args.lr
    hyperparams['entropy_mode'] = 'reward'
    hyperparams['controller_grad_clip'] = 0  # 10
    hyperparams['checkpoint'] = args.ckpt
    # hyperparams['var'] = args.var
    if not os.path.isdir(hyperparams['checkpoint']):
        mkdir_p(hyperparams['checkpoint'])
    controller_optim = torch.optim.Adam(controller.parameters(), lr=hyperparams['controller_lr'])

    trainable_tree = learnable_compuatation_tree()
    trainable_tree = trainable_tree.cuda()

    params = []
    for idx, v in enumerate(trainable_tree.learnable_operator_set):
        if idx not in leaves_index:
            for modules in trainable_tree.learnable_operator_set[v]:
                for param in modules.parameters():
                    params.append(param)
    for module in trainable_tree.linear:
        for param in module.parameters():
            params.append(param)
    for module in trainable_tree.input:
        for param in module.parameters():
            params.append(param)

    train_controller(controller, controller_optim, trainable_tree, params, hyperparams)
    end = time.time()
    elapsed = end - start
    with open(f"timing{args.dim}.txt", "a") as file:
        file.write(f"{dim} Dimensional Problem Completed in {elapsed} seconds \n")
