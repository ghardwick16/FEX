import numpy as np
import torch
from torch import sin, cos, exp
import math


def montecarlo_integration(function, domain, num_samples):
    points = torch.empty((num_samples, len(domain)))
    vol = 0
    for i in range(len(domain)):
        vol *= (domain[i][1] - domain[i][0])
        points[:, i] = torch.rand(num_samples).cuda() * (domain[i][1] - domain[i][0]) + domain[i][0]
    return vol / num_samples * torch.sum(function(points))


def integrand(func, mu, sigma, lam, tx, z):
    ### We have two cases:  either we pass in the candidate function in the form
    ### (learnable_tree, bs_action) or the true function (for measuring performance)
    tx = tx.cuda()
    z = z.cuda()

    t = tx[..., 0]
    x = tx[..., 1:]

    # u(t, x)
    u = func(tx)
    # u(t, x + z)
    if len(z.shape) > 1:
        tx_shift = torch.empty((z.shape[0], tx.shape[0])).cuda()
        for i in range(z.shape[0]):
            tx_shift[i, 0] = t
            tx_shift[i, 1:] = x + z[i, :]
    else:
        tx_shift = torch.empty_like(tx).cuda()
        tx_shift[0] = t
        tx_shift[1:] = x + z
    u_shift = func(tx_shift)
    # z dot grad u
    grad_u = torch.autograd.grad(u, tx, grad_outputs=torch.ones_like(u), create_graph=True)[0][1:].cuda()
    dot_prod = torch.empty(z.shape[0]).cuda()
    if len(z.shape) > 1:
        for i in range(z.shape[0]):
            dot_prod[i] = torch.dot(z[i, :].float(), grad_u)
    else:
        dot_prod = torch.dot(z.float(), grad_u)
    # nu
    # nu = lam/torch.sqrt(2*torch.Tensor([math.pi])*sigma)*torch.exp(-.5*((z-mu)/sigma)**2)
    # print(nu)
    return u_shift - u - dot_prod


def LHS_pde(func, tx):  # changed to let this use the pair (learnable_tree, bs_action) for computation directly
    # parameters for the LHS
    mu = .4
    sigma = .25
    lam = .3
    theta = .3
    epsilon = 0
    domain = torch.tensor([[0, 1]] * (tx.shape[1] - 1)).cuda()  # integrating on [0,1] on each dim of x

    if type(func) is tuple:
        learnable_tree = func[0]
        bs_action = func[1]
        u_func = lambda y: learnable_tree(y, bs_action)
    else:
        u_func = lambda y: func(y)

    t = torch.squeeze(tx[:, 0]).cuda()
    x = torch.squeeze(tx[:, 1:]).cuda()
    u = u_func(tx)

    # get derivatives, ut, ux, and trace of hessian
    v = torch.ones(u.shape).cuda()
    du = torch.autograd.grad(u, tx, grad_outputs=v, create_graph=True)[0]
    ut = du[:, 0]
    ux = du[:, 1:]
    if du.requires_grad:
        ddu = torch.autograd.grad(du, tx, grad_outputs=torch.ones_like(du), create_graph=True)[0]
    else:
        ddu = torch.zeros_like(du)
    trace_hessian = torch.sum(ddu[:, 1:], dim=1)

    # take the integral
    integral_dz = torch.empty(tx.shape[0])
    int_fun = lambda var: integrand(u_func, mu, sigma, lam, point, var)
    for i in range(tx.shape[0]):
        point = tx[i, :]
        integral_dz[i] = montecarlo_integration(int_fun, domain=domain, num_samples=50)
    print(x.shape)
    print(ux.shape)
    return ut + epsilon / 2 * torch.dot(x, ux) + 1 / 2 * theta ** 2 * trace_hessian + integral_dz


def RHS_pde(tx):
    #  parameters for the RHS:
    mu = .4
    sigma = .25
    lam = .3
    epsilon = 0
    theta = .3
    return lam * mu ** 2 + theta ** 2 + epsilon * torch.linalg.norm(tx[:, 1:], dim=1)


def true_solution(tx):  # for the most simple case, u(t,x) = ||x||^2
    return torch.linalg.norm(tx[:, 1:], dim=1)


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
