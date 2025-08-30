import numpy as np
import torch
from torch import sin, cos, exp
import math

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

def compute_k_params(theta, sigma, nu):
    lam_p = sqrt((theta ** 2) / (sigma ** 4) + 2 / (sigma ** 2 * nu)) - theta / sigma ** 2
    lam_n = sqrt((theta ** 2) / (sigma ** 4) + 2 / (sigma ** 2 * nu)) + theta / sigma ** 2
    return lam_p, lam_n

def k(y, lam_p, lam_n, nu):
    y_small = torch.where(y < 0, torch.tensor(0, dtype=y.dtype), y).cuda()
    y_large = torch.where(y > 0, torch.tensor(0, dtype=y.dtype), y).cuda()
    return torch.exp(-lam_p*y_large)/(nu*y_large) + torch.exp(-lam_n*torch.abs(y_small))/(nu*torch.abs(y_small))


def get_loss(func, lam, x, bd_x, fast):
    # parameters:
    k_mean = 2/5
    k_var = 28/125
    k_moment = (10 * (342 + 605 * sqrt(6)))/14241

    # We have two cases:  either we pass in the candidate function in the form
    # (learnable_tree, bs_action) or the true function
    if type(func) is tuple:
        learnable_tree = func[0]
        bs_action = func[1]
        u = lambda y: learnable_tree(y, bs_action)
    else:
        u = lambda y: func(y)

    u_x = u(x).squeeze().cuda()

    # compute gradient (w.r.t. variable x):
    v = torch.ones_like(u_x).cuda()
    grad_u = torch.autograd.grad(u_x, x[1:], grad_outputs=v, create_graph=True)[0]

    # compute second derivatives:
    hes_diag = get_hes_diag(grad_u, x[1:], fast)
    trace_hessian = torch.sum(hes_diag, dim=1)
    LHS = -trace_hessian

    ### INT (u(t, x+y) - u(t,x))k(y)dy =
    x_shift = torch.empty_like(x).cuda()
    x_shift[:, :] = x[:, :]
    x_shift[..., 1:] += k_mean

    # 1st derivs
    v = torch.ones_like(u_x).cuda()
    u_x_shift = u(x_shift).squeeze()
    du_x_shift = torch.autograd.grad(u_x_shift, x_shift, grad_outputs=v, create_graph=True)[0]

    # 2nd derivs
    hes_diag1 = torch.empty_like(x).cuda()
    v1 = torch.ones_like(x).cuda()
    if du.requires_grad:
        hes_diag1[..., :] = torch.autograd.grad(du_x_shift[..., 1:], x_shift, grad_outputs=v1, create_graph=True)[0][
                            ..., 1:]
    else:
        hes_diag1 = torch.zeros_like(du_x_shift).cuda()

    expect = u_x_shift + 1 / 2 * torch.sum(hes_diag1, dim=-1) * k_var ** 2
    n2 = expect - u_tx






    return loss

# Code to get L1, L2 relatives errors, and MSE
def get_errors(learnable_tree, bs_action, bs_leaf_action, dim):
    leaf_modes = [v[0].item() for v in bs_leaf_action]
    for leaf in learnable_tree.linear:
        leaf.set_mode(leaf_modes[leaf.leaf_index])
    u = lambda y: learnable_tree(y, bs_action)
    mse_list = []
    denom = []
    relative_num = []
    relative_denom = []
    for _ in range(1000):
        x = get_pts(num_samples=int(10000/dim), dims=dim)
        mse_list.append(torch.mean((true_solution(x) - u(x).squeeze()) ** 2))
        relative_num.append(torch.mean(torch.abs(true_solution(x) - u(x).squeeze())))
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
                   torch.sin,
                   torch.cos,
                   torch.sinh,
                   torch.cosh,
                   lambda x: torch.sin(3*x),
                   lambda x: torch.sin(6*x),
                   lambda x: torch.sin(9*x),
                   lambda x: torch.sin(12*x),
                   lambda x: torch.sin(15*x),
                   lambda x: torch.sin(18*x),
                   lambda x: torch.sin(21*x),
                   lambda x: torch.sin(24*x),
                   lambda x: torch.sinh(3*x),
                   lambda x: torch.sinh(6*x),
                   lambda x: torch.sinh(9*x),

                   ]


binary_functions = [lambda x, y: x + y,
                    lambda x, y: x * y,
                    lambda x, y: x - y]
'''
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
                       '({}*sin({})+{})',
                       '({}*cos({})+{})',
                       '({}*sinh({})+{}',
                       '({}*cosh({})+{}',
                       '({}*sin(3*({}))+{})',
                       '({}*sin(6*({}))+{})',
                       '({}*sin(9*({}))+{})',
                       '({}*sin(12*({}))+{})',
                       '({}*sin(15*({}))+{})',
                       '({}*sin(18*({}))+{})',
                       '({}*sin(21*({}))+{})',
                       '({}*sin(24*({}))+{})',
                       '({}*sinh(3*({}))+{})',
                       '({}*sinh(6*({}))+{})',
                       '({}*sinh(9*({}))+{})',
                       ]
# 'ref({})',
# 'exp(-({})**2/2)']

unary_functions_str_leaf = ['({}*{}+{})',
                            '({}*(0)+{}',
                            '({}*(1)+{})',
                            # '5',
                            # '-{}',
                            '(({}*({})+{})**2)',
                            '(({}*({})+{})**3)',
                            '(({}*({})+{})**4)',
                            # '({})**5',
                            '(exp({}*({})+{}))',
                            '(sin({}*({})+{}))',
                            '(cos({}*({})+{}))',
                            '(sinh({}*({})+{}))',
                            '(cosh({}*({})+{}))',
                            '(sin(2*({}*({})+{})))',
                            '(sin(3*({}*({})+{})))',
                            '(sin(4*({}*({})+{})))',
                            '(sin(5*({}*({})+{})))',
                            '(sinh(2*({}*({})+{})))',
                            '(sinh(3*({}*({})+{})))',
                            '(sinh(4*({}*({})+{})))',
                            '(sinh(5*({}*({})+{})))',
                            ]
                            

unary_functions_str_leaf = ['({}*{})',
                            '({}*(0))',
                            '({}*(1))',
                            # '5',
                            # '-{}',
                            '(({}*({}))**2)',
                            '(({}*({}))**3)',
                            '(({}*({}))**4)',
                            # '({})**5',
                            '(exp({}*({})))',
                            '(sin({}*({})))',
                            '(cos({}*({})))',
                            '(sinh({}*({})))',
                            '(cosh({}*({})))',
                            '(sin(3*({}*({}))))',
                            '(sin(6*({}*({}))))',
                            '(sin(9*({}*({}))))',
                            '(sin(12*({}*({}))))',
                            '(sin(15*({}*({}))))',
                            '(sin(18*({}*({}))))',
                            '(sin(21*({}*({}))))',
                            '(sin(24*({}*({}))))',
                            '(sinh(3*({}*({}))))',
                            '(sinh(6*({}*({}))))',
                            '(sinh(9*({}*({}))))',
                            ]
'''
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
                       '({}*sin({})+{})',
                       '({}*cos({})+{})',
                       '({}*sinh({})+{}',
                       '({}*cosh({})+{}',
                       '({}*sin(3*({}))+{})',
                       '({}*sin(6*({}))+{})',
                       '({}*sin(9*({}))+{})',
                       '({}*sin(12*({}))+{})',
                       '({}*sin(15*({}))+{})',
                       '({}*sin(18*({}))+{})',
                       '({}*sin(21*({}))+{})',
                       '({}*sin(24*({}))+{})',
                       '({}*sinh(3*({}))+{})',
                       '({}*sinh(6*({}))+{})',
                       '({}*sinh(9*({}))+{})',
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
                            '(sin({:.6f}*({})))',
                            '(cos({:.6f}*({})))',
                            '(sinh({:.6f}*({})))',
                            '(cosh({:.6f}*({})))',
                            '(sin(3*({:.6f}*({}))))',
                            '(sin(6*({:.6f}*({}))))',
                            '(sin(9*({:.6f}*({}))))',
                            '(sin(12*({:.6f}*({}))))',
                            '(sin(15*({:.6f}*({}))))',
                            '(sin(18*({:.6f}*({}))))',
                            '(sin(21*({:.6f}*({}))))',
                            '(sin(24*({:.6f}*({}))))',
                            '(sinh(3*({:.6f}*({}))))',
                            '(sinh(6*({:.6f}*({}))))',
                            '(sinh(9*({:.6f}*({}))))',
                            ]

binary_functions_str = ['(({})+({}))',
                        '(({})*({}))',
                        '(({})-({}))']
'''
unary_functions = [lambda x: x + 0 * x ** 2,
                   torch.sin,
                   torch.sinh,
                   ]


binary_functions = [lambda x, y: x + y,
                    lambda x, y: x * y,
                    lambda x, y: x - y]

unary_functions_str = ['({}*{}+{})',
                       '({}*sin({})+{})',
                       '({}*sinh({})+{}',]
# 'ref({})',
# 'exp(-({})**2/2)']

unary_functions_str_leaf = ['({})',
                            '(sin({}))',
                            '(sinh({}))',]
'''

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
