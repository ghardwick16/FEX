import numpy as np
import torch
from torch import sin, cos, exp
import math


# basic montecarlo integrator for higher dims
def montecarlo_integration(function, domain, num_samples):
    points = torch.empty((num_samples, len(domain)))
    vol = 1
    for i in range(len(domain)):
        vol *= (domain[i][1] - domain[i][0])
        points[:, i] = torch.rand(num_samples).cuda() * (domain[i][1] - domain[i][0]) + domain[i][0]
    return vol / num_samples * torch.sum(function(points))


# pre-computes grid points for an n-dim riemann integration method
def riemann_integration_points(dims, grid_points, side):
    tics = torch.linspace(0, 1, steps=grid_points)
    if side == 'right':
        axis_pts = tics[1:]
    if side == 'left':
        axis_pts = tics[:-1]
    tens_list = []
    for i in range(dims):
        tens_list.append(axis_pts)
    grid = torch.meshgrid(tens_list)
    out_tens = torch.empty(((grid_points - 1) ** dims, dims))
    for i in range(len(grid)):
        out_tens[:, i] = torch.flatten(grid[i])
    return out_tens


def integrand(func, u, du, mu, sigma, lam, tx, z):
    tx = tx.cuda()
    z = z.cuda()
    mu = torch.tensor([mu]).cuda()
    sigma = torch.tensor([sigma]).cuda()
    # u(t, x + z)
    tx_large = tx.unsqueeze(0).repeat(z.shape[0], 1, 1).cuda()
    z_large = z.unsqueeze(1).repeat(1, tx.shape[0], 1).cuda()
    # had to flatten the input to the function to make it a 2d tensor of inputs ratherthan 3d.
    input = torch.cat((torch.unsqueeze(tx_large[:, :, 0], 2), (tx_large[:, :, 1:] + z_large)), dim=-1).view(
        tx.shape[0] * z.shape[0], tx.shape[1])
    u_shift = func(input)
    u_shift = u_shift.reshape(z.shape[0], tx.shape[0])  # reshaped the output to be of the shape that we expect for the integration step
    # z dot grad u
    dot_prod = torch.sum((du[:, 1:].unsqueeze(0).repeat(z.shape[0], 1, 1) * z_large), dim=-1)
    # nu is a multivariable normal PDF with covariance sigma*I_d, mean mu.  As such, det(sigma*I_d) = (sigma^d)*1
    coef = lam / torch.sqrt((2 * torch.Tensor([math.pi]).cuda() * sigma) ** (tx.shape[1] - 1))
    z_minus_mu = z - mu
    nu = coef * torch.exp(-.5 * (sigma ** -1) * torch.sum(z_minus_mu * z_minus_mu, dim=1))
    print(u.shape)
    return (u_shift - u.unsqueeze(0).repeat(z.shape[0], 1) - dot_prod) * nu.unsqueeze(1).repeat(1, tx.shape[0])


def LHS_pde(func, tx):  # changed to let this use the pair (learnable_tree, bs_action) for computation directly
    # parameters for the LHS
    mu = .1
    sigma = 1e-4
    lam = .3
    theta = .3
    epsilon = 0
    tx = tx.cuda()
    if type(func) is tuple:
        learnable_tree = func[0]
        bs_action = func[1]
        u_func = lambda y: learnable_tree(y, bs_action)
    else:
        u_func = lambda y: func(y)

    u = u_func(tx)

    # get derivatives, ut, ux, and trace of hessian
    v = torch.ones(u.shape).cuda()
    du = torch.autograd.grad(u, tx, grad_outputs=v, create_graph=True)[0]
    ut = du[:, 0]
    # ux = du[:, 1:]

    if du.requires_grad:
        ddu = torch.autograd.grad(du, tx, grad_outputs=torch.ones_like(du), create_graph=True)[0]
    else:
        ddu = torch.zeros_like(du).cuda()
    trace_hessian = torch.sum(ddu[:, 1:], dim=1)

    # take the integral
    points = riemann_integration_points(dims=tx.shape[1]-1, grid_points=6, side='left')
    integral_dz = torch.sum(integrand(u_func, u, du, mu, sigma, lam, tx, points), dim=0)*1/points.shape[0]
    # since epsilon is zero I just got rid of the eps*x dot grad u term
    return ut + 1 / 2 * theta ** 2 * trace_hessian + integral_dz


def RHS_pde(tx):
    #  parameters for the RHS:
    mu = .1
    sigma = 1e-4
    lam = .3
    epsilon = 0
    theta = .3
    # since epsilon is zero I just removed the eps*||x||^2 term
    return torch.ones(tx.shape[0]).cuda() * (lam * mu ** 2 + theta ** 2)


def true_solution(tx):  # for the most simple case, u(t,x) = ||x||^2
    return torch.linalg.norm(tx[:, 1:], dim=1) ** 2


unary_functions = [lambda x: 0 * x ** 2,
                   lambda x: 1 + 0 * x ** 2,
                   lambda x: x + 0 * x ** 2,
                   lambda x: x ** 2,
                   lambda x: x ** 3,
                   lambda x: x ** 4,
                   torch.exp,
                   torch.sin,
                   torch.cos, ]

binary_functions = [lambda x, y: x + y,
                    lambda x, y: x * y,
                    lambda x, y: x - y]

unary_functions_str = ['({}*(0)+{})',
                       '({}*(1)+{})',
                       # '5',
                       '({}*{}+{})',
                       # '-{}',
                       '({}*({})**2+{})',
                       '({}*({})**3+{})',
                       '({}*({})**4+{})',
                       # '({})**5',
                       '({}*exp({})+{})',
                       '({}*sin({})+{})',
                       '({}*cos({})+{})', ]
# 'ref({})',
# 'exp(-({})**2/2)']

unary_functions_str_leaf = ['(0)',
                            '(1)',
                            # '5',
                            '({})',
                            # '-{}',
                            '(({})**2)',
                            '(({})**3)',
                            '(({})**4)',
                            # '({})**5',
                            '(exp({}))',
                            '(sin({}))',
                            '(cos({}))', ]

binary_functions_str = ['(({})+({}))',
                        '(({})*({}))',
                        '(({})-({}))']

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
