import numpy as np
import torch
from torch import sin, cos, exp
import math

def nearest_solution(lam, x):
    dim = x.shape[1]
    mu = torch.Tensor([math.pi]).cuda()
    if lam/(dim*mu**2) > 5/8:
        sin_mu_x = torch.sin(mu * x)
        return torch.prod(sin_mu_x, dim=1)
    else:
        cos_mu_x = torch.cos(mu/2 * x)
        return torch.prod(cos_mu_x, dim=1)

def get_pts(num_samples, dims):
    x = torch.empty((num_samples, dims)).cuda()
    torch.rand(num_samples, dims, out=x)
    x.requires_grad = True
    x = 2*x-1
    return x

def get_bdry_pts(num_samples, dims):
    x = torch.empty((num_samples, dims)).cuda()
    torch.rand(num_samples, dims, out=x) * 2 - 1
    x = 2*x-1
    edges = dims*2
    for i in range(edges):
        if i % 2 == 0:
            val = 1
        else:
            val = -1
        x[int(num_samples/edges)*i:int(num_samples/edges)*(i+1),int(np.floor(i/2))] = val
    x.requires_grad = True
    return x

def get_hes_diag(grad, x, fast_hes=False):
    if grad.requires_grad:
        hes_diag = torch.empty_like(x).cuda()
        if fast_hes:
            v = torch.ones_like(x).cuda()
            hes_diag[..., :] = torch.autograd.grad(grad, x, grad_outputs=v, create_graph=True)[0]
        else:
            v = torch.ones(x.shape[0]).cuda()
            for i in range(x.shape[1]):
                hes_diag[:, i] = torch.autograd.grad(grad[..., i], x, grad_outputs=v, create_graph=True)[0][:, i]
    else:
        hes_diag = torch.zeros_like(grad).cuda()
    return hes_diag


def get_eig_guess(func, x):
    fast = False
    # We have two cases:  either we pass in the candidate function in the form
    # (learnable_tree, bs_action) or the true function
    if type(func) is tuple:
        learnable_tree = func[0]
        bs_action = func[1]
        u = lambda y: learnable_tree(y, bs_action)
    else:
        u = lambda y: func(y)

    u_x = u(x).squeeze().cuda()

    # print(f'shape of func output: {u_x.shape}')
    # compute gradient:
    v = torch.ones_like(u_x).cuda()
    grad_u = torch.autograd.grad(u_x, x, grad_outputs=v, create_graph=True)[0]

    # Compute LHS:
    hes_diag = get_hes_diag(grad_u, x, fast)
    trace_hessian = torch.sum(hes_diag, dim=1)
    LHS = -trace_hessian

    return torch.mean(LHS / u_x)

def LHS(func, x):
    fast = False
    # We have two cases:  either we pass in the candidate function in the form
    # (learnable_tree, bs_action) or the true function
    if type(func) is tuple:
        learnable_tree = func[0]
        bs_action = func[1]
        u = lambda y: learnable_tree(y, bs_action)
    else:
        u = lambda y: func(y)

    u_x = u(x).squeeze().cuda()

    # print(f'shape of func output: {u_x.shape}')
    # compute gradient:
    v = torch.ones_like(u_x).cuda()
    grad_u = torch.autograd.grad(u_x, x, grad_outputs=v, create_graph=True)[0]

    # Compute LHS:
    hes_diag = get_hes_diag(grad_u, x, fast)
    trace_hessian = torch.sum(hes_diag, dim=1)
    LHS = -trace_hessian

    return LHS

def get_eig_loss(func, lam, x):
    # parameters:

    # We have two cases:  either we pass in the candidate function in the form
    # (learnable_tree, bs_action) or the true function
    if type(func) is tuple:
        learnable_tree = func[0]
        bs_action = func[1]
        u = lambda y: learnable_tree(y, bs_action)
    else:
        u = lambda y: func(y)

    u_x = u(x).squeeze().cuda()
    #print(f'shape of func output: {u_x.shape}')
    # compute gradient:
    v = torch.empty_like(u_x).cuda()
    grad_u = torch.autograd.grad(u_x, x, grad_outputs=v, create_graph=True)[0]

    # Compute RHS:
    RHS = lam*u_x

    # Compute LHS:
    hes_diag = get_hes_diag(grad_u, x, fast=False)
    trace_hessian = torch.sum(hes_diag, dim = 1)
    LHS = -trace_hessian

    #L2 Residual is the loss for the eigenvalue
    loss = torch.mean((LHS - RHS)**2)

    return loss

def get_loss(func, lam, x, bd_x, a_n=1, a_b=1000, a_d=100):

    # We have two cases:  either we pass in the candidate function in the form
    # (learnable_tree, bs_action) or the true function

    fast = False
    # We have two cases:  either we pass in the candidate function in the form
    # (learnable_tree, bs_action) or the true function
    if type(func) is tuple:
        learnable_tree = func[0]
        bs_action = func[1]
        u = lambda y: learnable_tree(y, bs_action)
    else:
        u = lambda y: func(y)

    u_x = u(x).squeeze().cuda()
    # print(f'shape of func output: {u_x.shape}')
    # compute gradient:
    v = torch.empty_like(u_x).cuda()
    grad_u = torch.autograd.grad(u_x, x, grad_outputs=v, create_graph=True)[0]

    # Compute RHS:
    RHS = lam*u_x

    # Step 1: loss1, compare LHS and RHS on domain
    loss1 = a_d * torch.mean((LHS(u, x) - RHS)**2)

    # Step 2:  loss2, compare LHS and RHS on boundary (on boundary, true solution is zero)
    # beta = 1000
    bd_u = u(bd_x).squeeze()**2
    loss2 = a_b * torch.mean(bd_u)


    # Step 3:  loss3, used to enforce a non-zero solution (See MartNet paper)
    c = 1
    p = 1
    diff = torch.abs(u_x**p - c)**2
    loss3 = a_n * torch.min(diff)

    loss = loss1 + loss2 + loss3

    return loss


# Code to get L1, L2 relatives errors, and MSE
def get_errors(learnable_tree, bs_action, bs_leaf_action, dim):
    leaf_modes = [v[0].item() for v in bs_leaf_action]
    for leaf in learnable_tree.linear:
        leaf.set_mode(leaf_modes[leaf.leaf_index])
        mw = leaf.mult_weight.item()
    u = lambda y: learnable_tree(y, bs_action)
    mse_list = []
    denom = []
    relative_num = []
    relative_denom = []
    for _ in range(1000):
        x = get_pts(num_samples=int(10000/dim), dims=dim)
        u_x = 1/mw*u(x).squeeze()
        mse_list.append(torch.mean((true_solution(x) - u_x) ** 2))
        relative_num.append(torch.mean(torch.abs(true_solution(x) - u_x)))
        relative_denom.append(torch.mean(torch.abs(true_solution(x))))
        denom.append(torch.mean(true_solution(x) ** 2))
    relative_l2 = torch.sqrt(sum(mse_list)) / torch.sqrt(sum(denom))
    relative = sum(relative_num) / sum(relative_denom)
    mse = 1 / 1000 * sum(mse_list)
    return relative_l2, relative, mse


def true_solution(x):
    # parameters:
    mu = torch.Tensor([math.pi]).cuda()
    sin_mu_x = torch.sin(mu * x)
    return torch.prod(sin_mu_x, dim=1)



unary_functions = [lambda x: x + 0 * x ** 2,
                   lambda x: 0 * x ** 2,
                   lambda x: 1 + 0 * x ** 2,
                   lambda x: x ** 2,
                   lambda x: x ** 3,
                   lambda x: x ** 4,
                   torch.exp,
                   torch.sinh,
                   lambda x: torch.sinh(3*x),
                   lambda x: torch.sinh(6*x),
                   lambda x: torch.sinh(9*x),
                   torch.cosh,
                   torch.sin,
                   torch.cos,
                   lambda x: torch.sin(3*x),
                   lambda x: torch.sin(6*x),
                   lambda x: torch.sin(9*x),
                   lambda x: torch.sin(12*x),
                   lambda x: torch.sin(15*x),
                   lambda x: torch.sin(18*x),
                   lambda x: torch.sin(21*x),
                   lambda x: torch.sin(24*x),


                   ]




unary_functions_str = ['({}*{}+{})',
                       '({}*(0)+{})',
                       '({}*(1)+{})',
                       # '5',
                       # '-{}',
                       '({}*({})**2+{})',
                       '({}*({})**3+{})',
                       '({}*({})**4+{})',
                       # '({})**5',
                       '({}*exp({})+{})',
                       '({}*sinh({})+{}',
                       '({}*sinh(3*({}))+{})',
                       '({}*sinh(6*({}))+{})',
                       '({}*sinh(9*({}))+{})',
                       '({}*cosh({})+{}',
                       '({}*sin({})+{})',
                       '({}*cos({})+{})',
                       '({}*sin(3*({}))+{})',
                       '({}*sin(6*({}))+{})',
                       '({}*sin(9*({}))+{})',
                       '({}*sin(12*({}))+{})',
                       '({}*sin(15*({}))+{})',
                       '({}*sin(18*({}))+{})',
                       '({}*sin(21*({}))+{})',
                       '({}*sin(24*({}))+{})',

                       ]

unary_functions_str_leaf = ['({:.6f}*{})',
                            '({:.6f}*(0))',
                            '({:.6f}*(1))',
                            # '5',
                            # '-{}',
                            '(({:.6f}*({}))**2)',
                            '(({:.6f}*({}))**3)',
                            '(({:.6f}*({}))**4)',
                            # '({})**5',
                            '(exp({:.6f}*({})))',
                            '(sinh({:.6f}*({})))',
                            '(sinh(3*({:.6f}*({}))))',
                            '(sinh(6*({:.6f}*({}))))',
                            '(sinh(9*({:.6f}*({}))))',
                            '(cosh({:.6f}*({})))',
                            '(sin({:.6f}*({})))',
                            '(cos({:.6f}*({})))',
                            '(sin(3*({:.6f}*({}))))',
                            '(sin(6*({:.6f}*({}))))',
                            '(sin(9*({:.6f}*({}))))',
                            '(sin(12*({:.6f}*({}))))',
                            '(sin(15*({:.6f}*({}))))',
                            '(sin(18*({:.6f}*({}))))',
                            '(sin(21*({:.6f}*({}))))',
                            '(sin(24*({:.6f}*({}))))',

                            ]

binary_functions_str = ['(({})+({}))',
                        '(({})*({}))',
                        '(({})-({}))']

binary_functions = [lambda x, y: x + y,
                    lambda x, y: x * y,
                    lambda x, y: x - y]

#unary_functions = [lambda x: sin(math.pi * x)]
#unary_functions_str = ['({}*sin(Pi*({}))+{})']
#unary_functions_str_leaf = ['(sin(Pi*({:.6f}*({}))))']



if __name__ == '__main__':
    batch_size = 200
    left = 0
    right = 1
    points = (torch.rand(batch_size, 1)) * (right - left) + left
    x = torch.autograd.Variable(points.cuda(), requires_grad=True)
    function = true_solution

    '''
    PDE loss
    '''
    LHS = LHS_pde(function, x)
    RHS = RHS_pde(x)
    pde_loss = torch.nn.functional.mse_loss(LHS, RHS)

    '''
    boundary loss
    '''
    bc_points = torch.FloatTensor([[left], [right]]).cuda()
    bc_value = true_solution(bc_points)
    bd_loss = torch.nn.functional.mse_loss(function(bc_points), bc_value)

    print('pde loss: {} -- boundary loss: {}'.format(pde_loss.item(), bd_loss.item()))
