import numpy as np
import torch
from torch import sin, cos, exp
import math
'''
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
'''

def get_pts(num_samples: int, dims: int = 100):
    # Sample from a standard normal distribution
    normal_samples = torch.randn(num_samples, dims).cuda()

    # Normalize to project onto the unit sphere
    unit_vectors = normal_samples / torch.sqrt(torch.sum(normal_samples**2, dim=0))

    # Sample radii with uniform distribution in volume
    radii = torch.rand(num_samples).cuda()**(1/dims)

    # Scale unit vectors by radii to get uniform samples inside the sphere
    points = unit_vectors * radii.view(-1, 1)
    points.requires_grad = True

    return points


def get_bdry_pts(num_samples: int, dims: int = 100):
    # Sample from a standard normal distribution
    normal_samples = torch.randn(num_samples, dims).cuda()

    # Normalize to project onto the unit sphere
    unit_vectors = normal_samples / torch.norm(normal_samples, dim=1, keepdim=True)
    unit_vectors.requires_grad = True
    return unit_vectors

def f_func(x, c, omega):
    out = torch.sum((c - omega**2)*torch.cos(omega*x), dim=1)
    return out

#Function to compare RHS and LHS of PIDE.  Note that I have added sigma (variance) as an input so that I can easily
#change it run to run
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

def get_loss(func, x, bd_x, fast):
    # parameters:
    c = -1
    omega = 2

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
    RHS = f_func(x, c, omega)

    # Compute LHS:
    hes_diag = get_hes_diag(grad_u, x, fast)
    trace_hessian = torch.sum(hes_diag, dim=1)
    LHS = trace_hessian + c * u_x

    # Step 1: loss1, compare LHS and RHS on domain
    #print(f'RHS: {RHS.shape}, LHS: {LHS.shape}')
    loss1 = torch.mean((LHS - RHS) ** 2)

    # Step 2:  loss2, compare LHS and RHS on boundary
    bd_u = u(bd_x).squeeze()
    loss2 = torch.mean((bd_u - true_solution(bd_x)) ** 2)
    loss = loss1 + loss2
    '''
    # C(f,g) constant from paper:
    c = 20.401903689763724

    # Ritz Loss from Boltzmann Paper:
    beta = 1000 #same as in paper
    bd_u = u(bd_x).squeeze()
    domain_loss = torch.abs(torch.mean(1/2*grad_u**2 + 1/2*k*u_x**2-f_func(x, lam, mu)*u_x) + c)
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
        x = get_pts(num_samples=int(20000*2/dim), dims=dim)
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
    omega = torch.tensor([2]).cuda()
    return torch.sum(torch.cos(omega*x), dim = 1)

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
                   lambda x: torch.sinh(3*x),
                   lambda x: torch.sinh(6*x),
                   lambda x: torch.sinh(9*x),
                   lambda x: torch.cos(3*x),
                   lambda x: torch.cos(6*x),
                   lambda x: torch.cos(9*x),

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
                       '({}*sin(3*({}))+{})',
                       '({}*sin(6*({}))+{})',
                       '({}*sin(9*({}))+{})',
                       '({}*sinh(3*({}))+{})',
                       '({}*sinh(6*({}))+{})',
                       '({}*sinh(9*({}))+{})',
                       '({}*cos(3*({}))+{})',
                       '({}*cos(6*({}))+{})',
                       '({}*cos(9*({}))+{})',
                       ]
# 'ref({})',
# 'exp(-({})**2/2)']
'''
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
                            '(sin(3*({}*({})+{})))',
                            '(sin(6*({}*({})+{})))',
                            '(sin(9*({}*({})+{})))',
                            '(sinh(3*({}*({})+{})))',
                            '(sinh(6*({}*({})+{})))',
                            '(sinh(9*({}*({})+{})))',
                            '(cos(3*({}*({})+{})))',
                            '(cos(6*({}*({})+{})))',
                            '(cos(9*({}*({})+{})))',
                            ]
                            '''

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
                            '(sinh(3*({}*({}))))',
                            '(sinh(6*({}*({}))))',
                            '(sinh(9*({}*({}))))',
                            '(cos(3*({}*({}))))',
                            '(cos(6*({}*({}))))',
                            '(cos(9*({}*({}))))',
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
