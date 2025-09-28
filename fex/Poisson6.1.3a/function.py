import numpy as np
import torch
from torch import sin, cos, exp
import math


def remove_points_in_circle(points, center, radius):
    """Removes points within a circle.

    Args:
        points (torch.Tensor): Tensor of shape (N, 2) representing the points.
        center (torch.Tensor): Tensor of shape (2,) representing the circle center.
        radius (float): Radius of the circle.

    Returns:
        torch.Tensor: Tensor of shape (M, 2) containing the points outside the circle.
    """

    # Calculate the distance from each point to the center
    distances = torch.norm(points - center, dim=1)

    # Create a mask for points outside the circle
    mask = distances > radius

    # Filter points using the mask
    filtered_points = points[mask]

    return filtered_points

def get_pts(num_samples, dims):
    x = torch.empty((num_samples, dims)).cuda()
    torch.rand(num_samples, dims, out=x)
    x.requires_grad = True
    x = 2*x-1
    centers = torch.tensor([[-.5,-.5], [.5,.5],[.5,-.5]]).cuda()
    radii = [.1,.2,.2]
    for i in range(len(radii)):
      x = remove_points_in_circle(x, centers[i], radii[i])
    return x

def get_bdry_pts(square_pts, hole_pts, dims):
    x = torch.empty((square_pts, dims)).cuda()
    torch.rand(square_pts, dims, out=x) * 2 - 1
    x = 2*x-1
    edges = dims*2
    for i in range(edges):
        if i % 2 == 0:
            val = 1
        else:
            val = -1
        x[int(square_pts/edges)*i:int(square_pts/edges)*(i+1),int(np.floor(i/2))] = val
    x.requires_grad = True

    centers = torch.tensor([[-.5,-.5], [.5,.5],[.5,-.5]]).cuda()
    radii = [.1,.2,.2]
    for i in range(len(radii)):
      y = torch.empty((hole_pts, dims)).cuda()
      torch.rand(hole_pts, dims, out=y)
      y.requires_grad = True
      y = 2*y-1
      norms = radii[i] * torch.sqrt(torch.sum(y ** 2, dim = 1)).repeat(dims, 1).T ** -1
      y = y*norms
      y += centers[i]
      x = torch.cat((x,y))
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

def get_loss(func, x, bd_x, fast):
    # parameters:
    mu = 7*torch.tensor([math.pi]).cuda()

    # We have two cases:  either we pass in the candidate function in the form
    # (learnable_tree, bs_action) or the true function
    if type(func) is tuple:
        learnable_tree = func[0]
        bs_action = func[1]
        u = lambda y: learnable_tree(y, bs_action)
    else:
        u = lambda y: func(y)

    u_x = u(x).squeeze()
    #print(f'cand func output shape: {u_x.shape}')

    # compute gradient:
    v = torch.ones_like(u_x).cuda()
    grad_u = torch.autograd.grad(u_x, x, grad_outputs=v, create_graph=True)[0]

    # Compute RHS:
    sin_mu_x = torch.sin(mu*x)
    RHS = 2*mu**2*sin_mu_x[...,0]*sin_mu_x[...,1]

    # Compute LHS:
    hes_diag = get_hes_diag(grad_u, x, fast)
    trace_hessian = torch.sum(hes_diag, dim=1)
    LHS = -trace_hessian

    # Step 1: loss1, compare LHS and RHS on domain
    loss1 = torch.mean((LHS - RHS) ** 2)

    # Step 2:  loss2, compare LHS and RHS on boundary
    #beta = 1000
    bd_u = u(bd_x).squeeze()
    loss2 = torch.mean((bd_u - true_solution(bd_x)) ** 2)

    loss = loss1 + loss2
    return loss

# Code to get L1, L2 relatives errors, and MSE
def get_errors(learnable_tree, bs_action, dim):
    u = lambda y: learnable_tree(y, bs_action)
    mse_list = []
    denom = []
    relative_num = []
    relative_denom = []
    for _ in range(1000):
        x = get_pts(num_samples=10000, dims=dim)
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
    mu = 7*torch.Tensor([math.pi]).cuda()
    sin_mu_x = torch.sin(mu * x)
    return sin_mu_x[...,0]*sin_mu_x[...,1]


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
                            '(sin(2*({}*({})+{})))',
                            '(sin(3*({}*({})+{})))',
                            '(sin(4*({}*({})+{})))',
                            '(sin(5*({}*({})+{})))',
                            '(sinh(2*({}*({})+{})))',
                            '(sinh(3*({}*({})+{})))',
                            '(sinh(4*({}*({})+{})))',
                            '(sinh(5*({}*({})+{})))',
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
                            '(sin(12*({}*({}))))',
                            '(sin(15*({}*({}))))',
                            '(sin(18*({}*({}))))',
                            '(sin(21*({}*({}))))',
                            '(sin(24*({}*({}))))',
                            '(sinh(3*({}*({}))))',
                            '(sinh(6*({}*({}))))',
                            '(sinh(9*({}*({}))))',
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
