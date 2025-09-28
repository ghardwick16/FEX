import numpy as np
import torch
from torch import sin, cos, exp
import math

#This function samples points from the domain.  [-1,1]^d

def get_pts(num_samples, dims):
    x = torch.empty((num_samples, dims)).cuda()
    torch.rand(num_samples, dims, out=x)
    x.requires_grad = True
    return 2*x-1

#This is a little jank, need to make sure num_samples is divisible by dims*2
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


def f_func(x, lam, mu):
    out = torch.sum((mu**2 + lam**2)*torch.sin(mu*x), dim = 1)
    return out

#Function to compare RHS and LHS of PIDE.  Note that I have added sigma (variance) as an input so that I can easily
#change it run to run

def get_loss(func, x, bd_x):
    # parameters:
    lam = 5
    k = lam**2
    mu = 5
    eps = 1

    # We have two cases:  either we pass in the candidate function in the form
    # (learnable_tree, bs_action) or the true function
    if type(func) is tuple:
        learnable_tree = func[0]
        bs_action = func[1]
        u = lambda y: learnable_tree(y, bs_action)
    else:
        u = lambda y: func(y)

    u_x = u(x).squeeze()

    # compute gradient:
    v = torch.ones_like(u_x).cuda()
    grad_u = torch.autograd.grad(u_x, x, grad_outputs=v, create_graph=True)[0]


    # Compute RHS:
    RHS = f_func(x, lam, mu)

    # Compute LH
    hes_diag = torch.empty_like(x).cuda()
    v1 = torch.ones_like(x).cuda()
    if grad_u.requires_grad:
        hes_diag[..., :] = torch.autograd.grad(grad_u, x, grad_outputs=v1, create_graph=True)[0]
    else:
        hes_diag = torch.zeros_like(grad_u).cuda()
    trace_hessian = torch.sum(hes_diag, dim=1)
    LHS = -trace_hessian + k*u_x

    # Step 1: loss1, compare LHS and RHS on domain
    loss1 = torch.mean((LHS - RHS) ** 2)

    # Step 2:  loss2, compare LHS and RHS on boundary
    bd_u = u(bd_x)
    loss2 = torch.mean((bd_u - true_solution(bd_x)) ** 2)
    beta = 100
    loss = loss1 + loss2**5
    '''
    # Ritz Loss from Boltzmann Paper:
    beta = 1000 #same as in paper
    bd_u = u(bd_x)
    domain_loss = torch.mean(1/2*grad_w**2 + 1/2*k*w**2-f_func(x, lam, mu)*w)
    boundary_loss = beta*torch.mean((bd_u - true_solution(bd_x))**2)
    loss = domain_loss + boundary_loss
    '''
    return loss

# Code to get L1, L2 relatives errors, and MSE
def get_errors(learnable_tree, bs_action, dim):
    u = lambda y: learnable_tree(y, bs_action)
    mse_list = []
    denom = []
    relative_num = []
    relative_denom = []
    for _ in range(1000):
        x = get_pts(num_samples=20000, dims=dim)
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
    mu = torch.tensor([5]).cuda()
    lam = torch.tensor([5]).cuda()
    c = -torch.sin(mu)/torch.sinh(lam)
    return torch.sum(c*torch.sinh(lam*x) + torch.sin(mu*x), dim = 1)

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
                   lambda x: torch.sin(2*x),
                   lambda x: torch.sin(3*x),
                   lambda x: torch.sin(4*x),
                   lambda x: torch.sin(5*x),
                   lambda x: torch.sinh(2*x),
                   lambda x: torch.sinh(3*x),
                   lambda x: torch.sinh(4*x),
                   lambda x: torch.sinh(5*x),
                   ]


binary_functions = [lambda x, y: x + y,
                    lambda x, y: x * y,
                    lambda x, y: x - y]

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
                       '({}*sin(2*({}))+{})',
                       '({}*sin(3*({}))+{})',
                       '({}*sin(4*({}))+{})',
                       '({}*sin(5*({}))+{})',
                       '({}*sinh(2*({}))+{})',
                       '({}*sinh(3*({}))+{})',
                       '({}*sinh(4*({}))+{})',
                       '({}*sinh(5*({}))+{})',
                       ]
# 'ref({})',
# 'exp(-({})**2/2)']

unary_functions_str_leaf = ['({})',
                            '(0)',
                            '(1)',
                            # '5',
                            # '-{}',
                            '(({})**2)',
                            '(({})**3)',
                            '(({})**4)',
                            # '({})**5',
                            '(exp({}))',
                            '(sin({}))',
                            '(cos({}))',
                            '(sinh({}))',
                            '(cosh({}))',
                            '(sin(2*({})))',
                            '(sin(3*({})))',
                            '(sin(4*({})))',
                            '(sin(5*({})))',
                            '(sinh(2*({})))',
                            '(sinh(3*({})))',
                            '(sinh(4*({})))',
                            '(sinh(5*({})))',
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

binary_functions_str = ['(({})+({}))',
                        '(({})*({}))',
                        '(({})-({}))']
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
