import numpy as np
import torch
from torch import sin, cos, exp
import math


# basic montecarlo integrator for higher dims
def montecarlo_integration(function, domain, num_samples):
    points = torch.empty((num_samples, len(domain)))
    vol = 0
    for i in range(len(domain)):
        vol *= (domain[i][1] - domain[i][0])
        points[:, i] = torch.rand(num_samples).cuda() * (domain[i][1] - domain[i][0]) + domain[i][0]
    return vol / num_samples * torch.sum(function(points))

# basic trapezoid rule integrator for 2-d case
def twod_trap(func, num_ints):
    vals = torch.empty(num_ints, num_ints)
    for i in range(num_ints):
        for j in range(num_ints):
            vals[i, j] = func(torch.tensor([i, j]) * 1 / (num_ints - 1))
    # int w.r.t. first space dim:
    integral_dx = torch.trapezoid(vals, dx=1 / (num_ints - 1), dim=0)
    # int w.r.t. the second space dim:
    integral_dy = torch.trapezoid(integral_dx, dx=1 / (num_ints - 1), dim=0)
    return integral_dy


def integrand(func, u, du, mu, sigma, lam, tx, z):
    tx = tx.cuda()
    z = z.cuda()

    t = tx[..., 0]
    x = tx[..., 1:]
    # u(t, x + z)
    tx_shift = torch.empty((z.shape[0], tx.shape[0])).cuda()
    tx_shift[:, 0] = t
    tx_shift[:, 1:] = x.expand(z.shape[0], x.shape[0]) + z
    u_shift = func(tx_shift)
    # z dot grad u
    dot_prod = torch.sum(z * du[1:].expand(z.shape[0], du[1:].shape[0]), dim=1)
    # nu is a multivariable normal PDF with covariance sigma*I_d, mean mu.  As such, det(sigma*I_d) = (sigma^d)*1
    nu = lam / torch.sqrt((2 * torch.Tensor([math.pi]) * sigma) ** z.shape[0]) * torch.exp(
        -.5 * torch.dot(torch.matmul((z - mu), sigma ** -1 * torch.eye(z.shape[0])), (z - mu)))
    # print(nu)
    return (u_shift - u.expand(u_shift.shape[0], u.shape[0]) - dot_prod) * nu


def LHS_pde(func, tx):  # changed to let this use the pair (learnable_tree, bs_action) for computation directly
    # parameters for the LHS
    mu = .1
    sigma = 1e-4
    lam = .3
    theta = .3
    epsilon = 0
    domain = torch.tensor([[0, 1]] * (tx.shape[1] - 1)).cuda()  # integrating on [0,1] on each dim of x
    tx = tx.cuda()
    if type(func) is tuple:
        learnable_tree = func[0]
        bs_action = func[1]
        u_func = lambda y: learnable_tree(y, bs_action)
    else:
        u_func = lambda y: func(y)

    t = torch.squeeze(tx[:, 0])
    x = torch.squeeze(tx[:, 1:])
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
    integral_dz = torch.empty(tx.shape[0]).cuda()
    for i in range(tx.shape[0]):
        point = tx[i, :]
        int_fun = lambda var: integrand(u_func, u[i, :], du[i, :], mu, sigma, lam, point, var)
        integral_dz[i] = twod_trap(int_fun, num_ints=5)
    # since epsilon is zero I just got ride of the eps*x dot grad u term
    return ut + 1 / 2 * theta ** 2 * trace_hessian + integral_dz


def RHS_pde(tx):
    #  parameters for the RHS:
    mu = .1
    sigma = 1e-4
    lam = .3
    epsilon = 0
    theta = .3
    # since epsilon is zero I just removed the eps*||x||^2 term
    return torch.zeros(tx.shape[0]).cuda()*(lam * mu ** 2 + theta ** 2)


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
