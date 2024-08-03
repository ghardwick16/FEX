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

parser = argparse.ArgumentParser(description='NAS')

parser.add_argument('--left', default=0, type=float)
parser.add_argument('--right', default=1, type=float)
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
deriv_unary = func.unary_function_derivatives
dderiv_unary = func.unary_function_2nd_derivatives
unary_functions_str = func.unary_functions_str
unary_functions_str_leaf = func.unary_functions_str_leaf
binary_functions_str = func.binary_functions_str

left = args.left
right = args.right
dim = args.dim
sigma = args.var
num_paths = 500
if args.clustering_thresh:
    thresh = args.clustering_thresh
    clustering = True
else:
    clustering = False


def get_boundary(num_pts, dim):
    bd_pts = (torch.rand(num_pts, dim).cuda()) * (args.right - args.left) + args.left
    bd_pts[:, 0] = 1  # the 0th index would be for time, which we want to be one since the
    # given boundary condition is u(T,x) = x and T = max time  = 1 since we
    # let t be in [0,1]
    return bd_pts


class candidate(object):
    def __init__(self, action, expression, error, clusters):
        self.action = action
        self.expression = expression
        self.error = error
        self.clusters = clusters


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

structure = []


def inorder_structure(tree):
    global structure
    if tree:
        inorder_structure(tree.leftChild)
        structure.append(tree.is_unary)
        inorder_structure(tree.rightChild)


inorder_structure(basic_tree())
print('tree structure', structure)

structure_choice = []
for is_unary in structure:
    if is_unary == True:
        structure_choice.append(len(unary))
    else:
        structure_choice.append(len(binary))
print('tree structure choices', structure_choice)

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


def reset_params(tree_params):
    for v in tree_params:
        # v.data.fill_(0.01)
        v.data.normal_(0.0, 0.1)


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
            if trainable_tree.clustering:
                combined_weight = trainable_tree.linear[leaves_cnt].weight[
                    0, trainable_tree.linear[leaves_cnt].clustering]
                for i in range(dim):
                    w.append(combined_weight[0, i])
            else:
                for i in range(dim):
                    w.append(trainable_tree.linear[leaves_cnt].weight[0][i].item())
                # w2 = trainable_tree.linear[leaves_cnt].weight[0][1].item()
            bias = trainable_tree.linear[leaves_cnt].bias[0].item()
            leaves_cnt = leaves_cnt + 1
            ## -------------------------------------- input variable element wise  ----------------------------
            expression = ''
            for i in range(0, dim):
                # print(midfun)
                x_expression = midfun.format('x' + str(i))
                expression = expression + ('{:.4f}*{}' + '+').format(w[i], x_expression)
            expression = expression + '{:.4f}'.format(bias)
            expression = '(' + expression + ')'
            # print('visualize', count, leaves_cnt, action)
            return expression
        elif leftfun is not None and rightfun is None:
            if '(0)' in midfun or '(1)' in midfun:
                return midfun.format('{:.4f}'.format(a), '{:.4f}'.format(b))
            else:
                return midfun.format('{:.4f}'.format(a), leftfun, '{:.4f}'.format(b))
        elif tree.leftChild is None and tree.rightChild is not None:
            if '(0)' in midfun or '(1)' in midfun:
                return midfun.format('{:.4f}'.format(a), '{:.4f}'.format(b))
            else:
                return midfun.format('{:.4f}'.format(a), rightfun, '{:.4f}'.format(b))
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
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, clustering=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.clustering = clustering
        self.in_features = in_features
        out_features = 1  # this is only to be used for leaves in FEX, so out features is always 1
        if clustering:
            self.weight = Parameter(torch.empty((out_features, int(np.max(clustering) + 1)), **factory_kwargs))
        else:
            self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for param in self.parameters():
            param.data.normal_(0.0, 0.1)

    def forward(self, input: Tensor) -> Tensor:
        if self.clustering:
            return F.linear(input, self.weight[:, self.clustering].squeeze(0), self.bias)
        else:
            return F.linear(input, self.weight, self.bias)


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


def compute_by_tree(tree, linear, x):
    ''' judge whether a emtpy tree, if yes, that means the leaves and call the unary operation '''
    if tree.leftChild == None and tree.rightChild == None:  # leaf node
        global leaves_cnt
        transformation = linear[leaves_cnt]
        leaves_cnt = leaves_cnt + 1
        return transformation(tree.key(x))
    elif tree.leftChild is None and tree.rightChild is not None:
        return tree.key(compute_by_tree(tree.rightChild, linear, x))
    elif tree.leftChild is not None and tree.rightChild is None:
        return tree.key(compute_by_tree(tree.leftChild, linear, x))
    else:
        return tree.key(compute_by_tree(tree.leftChild, linear, x), compute_by_tree(tree.rightChild, linear, x))


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
        for num, i in enumerate(range(leaves)):
            if clustering:
                linear_module = leaf(dim, 1, bias=True, clustering=self.clustering[i]).cuda()  # set only one variable
            else:
                linear_module = leaf(dim, 1, bias=True).cuda()
            linear_module.weight.data.normal_(0, 1 / math.sqrt(dim))
            # linear_module.weight.data[0, num%2] = 1
            linear_module.bias.data.fill_(0)
            self.linear.append(linear_module)

    def forward(self, x, bs_action):
        # print(len(bs_action))
        global leaves_cnt
        leaves_cnt = 0
        function = lambda y: compute_by_tree(get_function_trainable_params(bs_action, self.learnable_operator_set),
                                             self.linear, y)
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
        self.output_size = sum(structure_choice)

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
        total_logits = self.forward(inputs)

        cumsum = np.cumsum([0] + structure_choice)
        for idx in range(len(structure_choice)):
            logits = total_logits[:, cumsum[idx]:cumsum[idx + 1]]

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
        return actions, log_probs

    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.controller_hid)
        return (tools.get_variable(zeros, True, requires_grad=False),
                tools.get_variable(zeros.clone(), True, requires_grad=False))


def get_reward(bs, actions, learnable_tree, tree_params, tree_optim):
    regression_errors = []
    formulas = []
    batch_size = bs
    global count, leaves_cnt
    for bs_idx in range(batch_size):
        bs_action = [v[bs_idx] for v in actions]
        cand_func = (learnable_tree, bs_action)
        reset_params(tree_params)
        tree_optim = torch.optim.Adam(tree_params, lr=1e-2)
        error_hist = []
        #x_t = func.get_pts(num_paths, dim - 1)
        for _ in range(20):
            x_t = func.get_pts(num_paths, dim - 1)
            loss = func.get_loss(cand_func, func.true_solution, x_t, sigma)
            tree_optim.zero_grad()
            loss.backward()
            #error_hist.append(loss.item())
            tree_optim.step()
        tree_optim = torch.optim.LBFGS(tree_params, lr=1, max_iter=20)
        print('---------------------------------- batch idx {} -------------------------------------'.format(bs_idx))
        def closure():
            tree_optim.zero_grad()
            x_t = func.get_pts(num_paths, dim - 1)
            loss = func.get_loss(cand_func, func.true_solution, x_t, sigma)
            print('loss before: ', loss.item())
            loss.backward()
            error_hist.append(loss.item())
            return loss
        tree_optim.step(closure)
        '''
        ## USING COARSE TUNE, ADJUST STRUCTURE BY SETTING PARAMETERS EQUAL
        loss = func.loss = func.get_loss(cand_func, func.true_solution, x_t, sigma)
        if clustering and loss == loss:
            clusters = []
            for param in tree_params:
                if param.shape[-1] == args.dim:
                    cluster = [hcluster.fclusterdata(np.expand_dims(param[0, :].cpu().detach().numpy(), -1), thresh,
                                                     criterion="distance") - 1]
                    clusters.append(cluster)

            learnable_tree = learnable_compuatation_tree(clusters)
            tree_params = []
            for idx, v in enumerate(learnable_tree.learnable_operator_set):
                if idx not in leaves_index:
                    for modules in learnable_tree.learnable_operator_set[v]:
                        for param in modules.parameters():
                            tree_params.append(param)
            for module in learnable_tree.linear:
                for param in module.parameters():
                    tree_params.append(param)

            ## COARSE TUNE WITH NEW STRUCTURE TO SPEED UP FINE-TUNE:
            cand_func = (learnable_tree, bs_action)
            tree_optim = torch.optim.Adam(tree_params, lr=1e-2)
            for _ in range(20):
                x_t = func.get_pts(num_paths, dim - 1)
                #x_t.requires_grad = True
                loss = func.get_loss(cand_func, func.true_solution, x_t, sigma)
                tree_optim.zero_grad()
                #print('loss before: ', loss.item())
                #error_hist.append(loss.item())
                loss.backward()
                tree_optim.step()
            tree_optim = torch.optim.LBFGS(tree_params, lr=1, max_iter=20)
            tree_optim.step(closure)
            
            tree_optim = torch.optim.Adam(tree_params, lr=1e-3)
            for _ in range(100):
                x_t = func.get_pts(num_paths, dims=args.dim - 1)
                x_t.requires_grad = True
                loss = func.get_loss(cand_func, func.true_solution, x_t, sigma)
                tree_optim.zero_grad()
                print('loss before: ', loss.item())
                error_hist.append(loss.item())
                loss.backward()
                tree_optim.step()

            ## END COARSE TUNE CODE

            '''
        #x_t = func.get_pts(num_paths, dim - 1)
        #loss = func.get_loss(cand_func, func.true_solution, x_t, sigma)
        #print(f'loss after, {loss}')
        #error_hist.append(loss.item())
        print('min: ', min(error_hist))
        regression_errors.append(min(error_hist))

        count = 0
        leaves_cnt = 0
        formula = inorder_visualize(basic_tree(), bs_action, learnable_tree)
        count = 0
        leaves_cnt = 0
        formulas.append(formula)

    return regression_errors, formulas


def discount(x, amount):
    return scipy.signal.lfilter([1], [1, -amount], x[::-1], axis=0)[::-1]


def true(x):
    return -0.5 * (torch.sum(x ** 2, dim=1, keepdim=True))

# restructure and re-tune a coarse-tuned tree
def restruct_and_tune(action, pre_its, tune_its, thresh):
    trainable_tree = learnable_compuatation_tree().cuda()
    cand_func = (trainable_tree, action)
    params = []
    for idx, v in enumerate(trainable_tree.learnable_operator_set):
        if idx not in leaves_index:
            for modules in trainable_tree.learnable_operator_set[v]:
                for param in modules.parameters():
                    params.append(param)
    for module in trainable_tree.linear:
        for param in module.parameters():
            params.append(param)

    tree_optim = torch.optim.Adam(params, lr=1e-2)
    for _ in range(20):
        x_t = func.get_pts(num_paths, dims=args.dim - 1)
        loss = func.get_loss(cand_func, func.true_solution, x_t, sigma)
        tree_optim.zero_grad()
        loss.backward()
        tree_optim.step()

    tree_optim = torch.optim.LBFGS(params, lr=1, max_iter=20)

    def closure():
        x_t = func.get_pts(num_paths, dims=args.dim - 1)
        tree_optim.zero_grad()
        loss = func.get_loss(cand_func, func.true_solution, x_t, sigma)
        loss.backward()
        return loss

    tree_optim.step(closure)

    tree_optim = torch.optim.Adam(params, lr=1e-3)
    for _ in range(pre_its):
        x_t = func.get_pts(num_paths, dims=args.dim - 1)
        loss = func.get_loss(cand_func, func.true_solution, x_t, sigma)
        tree_optim.zero_grad()
        loss.backward()
        tree_optim.step()

    for param in params:
        if param.shape[-1] == args.dim:
            for x in param[0]:
                if x != x or x == float("inf") or x == float("-inf"):
                    print('Could Not Cluster')
                    return None, loss.item()

    clusters = []
    for param in params:
        if param.shape[-1] == args.dim:
            cluster = [hcluster.fclusterdata(np.expand_dims(param[0, :].cpu().detach().numpy(), -1), thresh,
                                                 criterion="distance") - 1]
            clusters.append(cluster)

    new_tree = learnable_compuatation_tree(clusters)
    params = []
    for idx, v in enumerate(new_tree.learnable_operator_set):
        if idx not in leaves_index:
            for modules in new_tree.learnable_operator_set[v]:
                for param in modules.parameters():
                    params.append(param)
    for module in new_tree.linear:
        for param in module.parameters():
            params.append(param)

    ## COARSE TUNE WITH NEW STRUCTURE TO SPEED UP FINE-TUNE:
    tree_optim = torch.optim.Adam(params, lr=1e-2)
    for _ in range(20):
        x_t = func.get_pts(num_paths, dims=args.dim - 1)
        loss = func.get_loss(cand_func, func.true_solution, x_t, sigma)
        tree_optim.zero_grad()
        loss.backward()
        tree_optim.step()

    tree_optim = torch.optim.LBFGS(params, lr=1, max_iter=20)
    def closure():
        x_t = func.get_pts(num_paths, dims=args.dim - 1)
        tree_optim.zero_grad()
        loss = func.get_loss(cand_func, func.true_solution, x_t, sigma)
        loss.backward()
        return loss
    tree_optim.step(closure)

    #perform first steps of fine tuning:
    tree_optim = torch.optim.Adam(params, lr=1e-3)
    for _ in range(tune_its):
        x_t = func.get_pts(num_paths, dims=args.dim - 1)
        loss = func.get_loss(cand_func, func.true_solution, x_t, sigma)
        tree_optim.zero_grad()
        loss.backward()
        tree_optim.step()

    final_loss = loss.item()
    if final_loss != final_loss:
        final_loss = 1e10

    return clusters, final_loss


def best_error(best_action, learnable_tree):
    # t = torch.rand(args.domainbs, 1).cuda()
    # x1 = (torch.rand(args.domainbs, args.dim - 1).cuda()) * (args.right - args.left) + args.left
    # x = torch.cat((t, x1), 1)
    # x.requires_grad = True
    x_t = func.get_pts(num_paths, dims=args.dim - 1)
    x_t.requires_grad = True
    bs_action = best_action

    lhs_func = (learnable_tree, bs_action)

    # bd_pts = get_boundary(args.bdbs, dim)
    # bc_true = func.true_solution(bd_pts)
    # bd_nn = learnable_tree(bd_pts, bs_action)
    # bd_error = torch.nn.functional.mse_loss(bc_true, bd_nn)
    # function_error = torch.nn.functional.mse_loss(func.LHS_pde(lhs_func, x), func.RHS_pde(x))
    regression_error = func.get_loss(lhs_func, func.true_solution, x_t, sigma)

    print(f'error: {regression_error}')

    return regression_error


def train_controller(Controller, Controller_optim, trainable_tree, tree_params, hyperparams):
    ### obtain a new file name ###
    file_name = os.path.join(hyperparams['checkpoint'], 'var_' + str(args.var) + 'log{}.txt')
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
        actions, log_probs = controller.sample(batch_size=bs, step=step)
        binary_code = ''
        for action in actions:
            binary_code = binary_code + str(action[0].item())
        rewards, formulas = get_reward(bs, actions, trainable_tree, tree_params, tree_optim)
        rewards = torch.cuda.FloatTensor(rewards).view(-1, 1)
        rewards[rewards != rewards] = 1e10

        error = rewards
        batch_min_idx = torch.argmin(error)
        batch_min_action = [v[batch_min_idx] for v in actions]

        clusters, new_best_loss = restruct_and_tune(batch_min_action, pre_its=100, tune_its=100, thresh=thresh)
        error[batch_min_idx] = new_best_loss
        batch_smallest = error.min()

        # discount
        if 1 > hyperparams['discount'] > 0:
            rewards = discount(rewards, hyperparams['discount'])

        base = args.base
        rewards[rewards > base] = base
        rewards[rewards != rewards] = 1e10
        rewards = 1 / (1 + torch.sqrt(rewards))
        #print(f'rewards: {rewards}')
        batch_best_formula = formulas[batch_min_idx]
        if batch_smallest != batch_smallest:
            batch_smallest = 1e10


        candidates.add_new(candidate(action=batch_min_action, expression=batch_best_formula, error=batch_smallest, clusters=clusters))

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

        min_error = error.min().item()
        # print('******************** ', min_error)
        if smallest_error > min_error:
            smallest_error = min_error

            min_idx = torch.argmin(error)
            min_action = [v[min_idx] for v in actions]
            best_formula = formulas[min_idx]

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
        print('error:{} action:{} formula:{}'.format(candidate_.error.item(), [v.item() for v in candidate_.action],
                                                     candidate_.expression))
        action_string = ''
        for v in candidate_.action:
            action_string += str(v.item()) + '-'
        logger.append([666, 0, 0, action_string, candidate_.error.item(), candidate_.expression])
        # logger.append([666, 0, 0, 0, candidate_.error.item(), candidate_.expression])
    finetune = 5000
    global count, leaves_cnt
    '''
    print(f'Reordering Candidates with Medium-Tune')
    cand_list = []
    for candidate_ in candidates.candidates:
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

        reset_params(params)
        ## COARSE TUNE TO SPEED UP FINE-TUNE:
        cand_func = (trainable_tree, candidate_.action)
        x_t = func.get_pts(num_paths, dims=args.dim - 1)
        loss = func.get_loss(cand_func, func.true_solution, x_t, sigma)
        loss_before = loss.item()

        tree_optim = torch.optim.Adam(params, lr=1e-2)
        for _ in range(20):
            x_t = func.get_pts(num_paths, dims=args.dim - 1)
            loss = func.get_loss(cand_func, func.true_solution, x_t, sigma)
            tree_optim.zero_grad()
            loss.backward()
            tree_optim.step()
        tree_optim = torch.optim.LBFGS(params, lr=1, max_iter=20)

        def closure():
            x_t = func.get_pts(num_paths, dims=args.dim - 1)
            tree_optim.zero_grad()
            loss = func.get_loss(cand_func, func.true_solution, x_t, sigma)
            loss.backward()
            return loss

        tree_optim.step(closure)
        ## END COARSE TUNE CODE

        ## USING COARSE TUNE, ADJUST STRUCTURE BY SETTING PARAMETERS EQUAL
        # trying to get rid of the errors by checking if any of the parameters are infinite or nan
        clusterable = True
        for param in params:
            if param.shape[-1] == args.dim:
                for x in param[0]:
                    if x != x or x == float("inf") or x == float("-inf"):
                        clusterable = False

        if clustering and clusterable:
            clusters = []
            for param in params:
                if param.shape[-1] == args.dim:
                    cluster = [hcluster.fclusterdata(np.expand_dims(param[0, :].cpu().detach().numpy(), -1), thresh,
                                                     criterion="distance") - 1]
                    clusters.append(cluster)

            trainable_tree = learnable_compuatation_tree(clusters)
            params = []
            for idx, v in enumerate(trainable_tree.learnable_operator_set):
                if idx not in leaves_index:
                    for modules in trainable_tree.learnable_operator_set[v]:
                        for param in modules.parameters():
                            params.append(param)
            for module in trainable_tree.linear:
                for param in module.parameters():
                    params.append(param)

            ## COARSE TUNE WITH NEW STRUCTURE TO SPEED UP FINE-TUNE:
            cand_func = (trainable_tree, candidate_.action)
            tree_optim = torch.optim.Adam(params, lr=1e-2)
            for _ in range(20):
                x_t = func.get_pts(num_paths, dims=args.dim - 1)
                loss = func.get_loss(cand_func, func.true_solution, x_t, sigma)
                tree_optim.zero_grad()
                loss.backward()
                tree_optim.step()
            tree_optim = torch.optim.LBFGS(params, lr=1, max_iter=50)

            def closure():
                x_t = func.get_pts(num_paths, dims=args.dim - 1)
                tree_optim.zero_grad()
                loss = func.get_loss(cand_func, func.true_solution, x_t, sigma)
                loss.backward()
                return loss
            tree_optim.step(closure)

            # perform first 100 steps of finetune:
            tree_optim = torch.optim.Adam(params, lr=1e-3)
            for _ in range(250):
                x_t = func.get_pts(num_paths, dims=args.dim - 1)
                loss = func.get_loss(cand_func, func.true_solution, x_t, sigma)
                tree_optim.zero_grad()
                loss.backward()
                tree_optim.step()
        #save learned trees and loss:
        if loss.item() == loss.item():
            cand_list.append([loss.item(), trainable_tree, candidate_.action])
        formula = inorder_visualize(basic_tree(), candidate_.action, trainable_tree)
        print(f'Candidate: {formula}, loss: {loss.item()}')
    cand_list.sort()
    print('Re-sorted Candidates:')
    for cand in cand_list:
        print(f'Loss: {cand[0]}, Action: {cand[2]}')
    '''
    #Actual Fine-Tune Using the Resorted Candidates:
    i = 0
    stopping = False
    '''
    cand_number = 0
    while i < len(cand_list) and not stopping:
        cand = cand_list[i]
        trainable_tree = cand[1]
        action = cand[2]

        params = []
        for idx, v in enumerate(trainable_tree.learnable_operator_set):
            if idx not in leaves_index:
                for modules in trainable_tree.learnable_operator_set[v]:
                    for param in modules.parameters():
                        params.append(param)
        for module in trainable_tree.linear:
            for param in module.parameters():
                params.append(param)
    '''
    while not stopping:
        cand_number = 0
        for candidate_ in candidates.candidates:
            trainable_tree = learnable_compuatation_tree(candidate_.clusters)
            params = []
            for idx, v in enumerate(trainable_tree.learnable_operator_set):
                if idx not in leaves_index:
                    for modules in trainable_tree.learnable_operator_set[v]:
                        for param in modules.parameters():
                            params.append(param)
            for module in trainable_tree.linear:
                for param in module.parameters():
                    params.append(param)

            ## COARSE TUNE TO SPEED UP FINE-TUNE:
            x_t = func.get_pts(num_paths, dims=args.dim - 1)
            x_t.requires_grad = True
            cand_func = (trainable_tree, candidate_.action)

            tree_optim = torch.optim.Adam(params, lr=1e-2)
            for _ in range(20):
                loss = func.get_loss(cand_func, func.true_solution, x_t, sigma)
                tree_optim.zero_grad()
                loss.backward()
                tree_optim.step()

            tree_optim = torch.optim.LBFGS(params, lr=1, max_iter=20)
            def closure():
                tree_optim.zero_grad()
                loss = func.get_loss(cand_func, func.true_solution, x_t, sigma)
                loss.backward()
                return loss

            tree_optim.step(closure)
            ## END COARSE TUNE CODE
            '''
            ## USING COARSE TUNE, ADJUST STRUCTURE BY SETTING PARAMETERS EQUAL
            if clustering:
                clusters = []
                for param in params:
                    if param.shape[-1] == args.dim:
                        cluster = [hcluster.fclusterdata(np.expand_dims(param[0, :].cpu().detach().numpy(), -1), thresh,
                                                         criterion="distance") - 1]
                        clusters.append(cluster)
    
                trainable_tree = learnable_compuatation_tree(clusters)
                params = []
                for idx, v in enumerate(trainable_tree.learnable_operator_set):
                    if idx not in leaves_index:
                        for modules in trainable_tree.learnable_operator_set[v]:
                            for param in modules.parameters():
                                params.append(param)
                for module in trainable_tree.linear:
                    for param in module.parameters():
                        params.append(param)
    
                ## COARSE TUNE WITH NEW STRUCTURE TO SPEED UP FINE-TUNE:
                x_t = func.get_pts(num_paths, dims=args.dim - 1)
                x_t.requires_grad = True
                cand_func = (trainable_tree, candidate_.action)
                tree_optim = torch.optim.Adam(params, lr=1e-2)
                for _ in range(20):
                    loss = func.get_loss(cand_func, func.true_solution, x_t, sigma)
                    tree_optim.zero_grad()
                    loss.backward()
                    tree_optim.step()
                tree_optim = torch.optim.LBFGS(params, lr=1, max_iter=50)
    
                def closure():
                    tree_optim.zero_grad()
                    loss = func.get_loss(cand_func, func.true_solution, x_t, sigma)
                    loss.backward()
                    return loss
    
                tree_optim.step(closure)
                ## END COARSE TUNE CODE
            '''
            relatives = []
            tree_optim = torch.optim.Adam(params, lr=1e-3)
            error_list = []
            idx = 5000
            for current_iter in range(finetune):
                error = best_error(candidate_.action, trainable_tree)
                #error = best_error(action, trainable_tree)
                error_list.append(error.item())
                tree_optim.zero_grad()
                error.backward()
                tree_optim.step()
                # for para in params:
                #     if para is not None:
                #         print(para.grad)

                count = 0
                leaves_cnt = 0
                formula = inorder_visualize(basic_tree(), candidate_.action, trainable_tree)
                #formula = inorder_visualize(basic_tree(), action, trainable_tree)
                leaves_cnt = 0
                count = 0
                suffix = 'Finetune-- Iter {current_iter} Error {error:.5f} Formula {formula}'.format(
                    current_iter=current_iter, error=error, formula=formula)
                if (current_iter + 1) % 100 == 0:
                    logger.append([current_iter, 0, 0, 0, error.item(), formula])
                # if smallest_error <= 1e-10:
                #     logger.append([current_iter, 0, 0, 0, error.item(), formula])
                #     return
                cosine_lr(tree_optim, 1e-3, current_iter, finetune)
                print(suffix)
                '''
                if (current_iter + 1) % 10 == 0:
                    _, relative, _ = func.get_errors(trainable_tree, candidate_.action, args.dim - 1)
                    relatives.append(relative.item())
                '''
                # adding a halt condition when the error is of the same order as machine epsilon (or the last 100 have been close)
                if current_iter > 100:
                    if sum(error_list[(current_iter - 5):])/len(error_list[(current_iter - 5):]) < 1.3e-14:
                        stopping = True
                #if current_iter > idx + 200:
                #    stopping = True
                #if current_iter > 1500:
                #    stopping = True
                if current_iter == finetune - 1 or stopping:
                    relative_l2, relative, mse = func.get_errors(trainable_tree, candidate_.action, args.dim - 1)
                    logger.append([f'RL2: {relative_l2}', f'REL: {relative}', f'MSE: {mse}', f'Loss: {error.item()}', 0, 0])
                if stopping:
                    break
            '''
            plt.figure(cand_number)
            plt.plot(error_list)
            plt.yscale('log')
            plt.xlabel('Iteration')
            plt.ylabel('Loss (log scale)')
            title = 'Equation (3.10): ' + str(args.dim - 1) + ' Dimensional Problem - Finetune Loss Plot'
            plt.title(title)
            name = 'cand' + str(cand_number) + '_dims' + str(args.dim) + '_var' + str(args.var) + '_plot.png'
            plt.savefig(name, format='png')
            # cand_number += 1

            plt.figure(20 + cand_number)
            plt.plot(relatives)
            plt.yscale('log')
            plt.xlabel('Iteration (in hundreds)')
            plt.ylabel('Relative Error (log scale)')
            title = 'Equation (3.10): ' + str(args.dim - 1) + ' Dimensional Problem - Relative Error Plot'
            plt.title(title)
            name = 'cand' + str(cand_number) + '_dims' + str(args.dim) + '_var' + str(args.var) + '_relative_plot.png'
            plt.savefig(name, format='png')
            cand_number += 1
            '''

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

    train_controller(controller, controller_optim, trainable_tree, params, hyperparams)
    end = time.time()
    elapsed = end - start
    with open(f"timing{args.dim}.txt", "a") as file:
        file.write(f"With Top 1 Restructure in Searching: {dim} Dimensional Problem Completed in {elapsed} seconds \n")
