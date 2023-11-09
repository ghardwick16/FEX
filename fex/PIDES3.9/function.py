import numpy as np
import torch
from torch import sin, cos, exp
import math


def center_integration_points(dims, grid_points, left, right):
    num_per_dim = int(np.floor((grid_points ** (1 / dims))))
    tics = torch.linspace(left, right, steps=num_per_dim)
    tens_list = []
    for i in range(dims):
        tens_list.append(tics)
    grid = torch.meshgrid(tens_list)
    out_tens = torch.empty((num_per_dim ** dims, dims)).cuda()
    for i in range(len(grid)):
        out_tens[:, i] = torch.flatten(grid[i])
    return out_tens


def LHS_pde(func, tx):  # changed to let this use the pair (learnable_tree, bs_action) for computation directly
    # parameters for the LHS
    mu = .4
    sigma = .25
    lam = .3
    epsilon = .25
    theta = .3
    left = 0
    right = 1

    z = center_integration_points(tx.shape[1] - 1, 1000, left=left, right=right)
    t = torch.squeeze(tx[..., 0]).cuda()
    x = torch.squeeze(tx[..., 1:]).cuda()

    nu = lam / torch.sqrt(2 * torch.Tensor([math.pi]) * sigma).cuda() * torch.exp(-.5 * ((z - mu) / sigma).cuda() ** 2)
    tx_shift = tx.unsqueeze(1).repeat(1, z.shape[0], 1).cuda()
    z_large = z.unsqueeze(0).repeat(tx.shape[0], 1, 1).cuda()
    tx_shift[..., 1:] += z_large
    ### We have two cases:  either we pass in the condidate function in the form
    ### (learnable_tree, bs_action) or the true function (for measuring performance)
    if type(func) is tuple:
        learnable_tree = func[0]
        bs_action = func[1]
        u_func = lambda y: learnable_tree(y, bs_action)
    else:
        u_func = lambda y: func(y)

    u_shift = torch.squeeze(u_func(tx_shift))
    u = torch.squeeze(u_func(tx))

    # get derivatives
    v = torch.ones(u.shape).cuda()
    du = torch.autograd.grad(u, tx, grad_outputs=v, create_graph=True)[0]
    ut = du[:, 0]
    ux = torch.squeeze(du[:, 1:])

    hes_diag = torch.empty((tx.shape[0], tx.shape[1] - 1)).cuda()
    if du.requires_grad:
        for i in range(tx.shape[1] - 1):
            hes_diag[:, i] = torch.autograd.grad(du[:, 1 + i], tx, grad_outputs=v, create_graph=True)[
                                 0][:, 1 + i]
    else:
        hes_diag = torch.zeros_like(du).cuda()
    trace_hessian = torch.sum(hes_diag, dim=1)
    print(u.unsqueeze(1).repeat(1, z.shape[0]).shape)
    print(du.shape)
    print(du[:, 1:].unsqueeze(1).repeat(1, z.shape[0], 1).shape)
    print(z_large.shape)

    integrand = (u_shift - u.unsqueeze(1).repeat(1, z.shape[0]) -
                 torch.sum(du[:, 1:].unsqueeze(1).repeat(1, z.shape[0], 1) * z_large, dim=-1)) * \
                 nu.unsqueeze(0).repeat(tx.shape[0], 1) * (right - left) / z.shape[0]
    integral_dz = torch.sum(integrand, dim=-1)

    return ut + epsilon / 2 * x * ux + 1 / 2 * theta ** 2 * trace_hessian + integral_dz


def RHS_pde(tx):
    #  parameters for the RHS:
    mu = .4
    sigma = .25
    lam = .3
    epsilon = .25
    theta = .3
    norm = 1 / (tx.shape[1] - 1) * torch.sum(tx[:, 1:] ** 2, dim=-1)
    return epsilon * norm + theta ** 2 + \
        (lam * (mu ** 2 + sigma ** 2))


def true_solution(tx):
    return torch.sum(tx[:, 1:] ** 2, dim=-1).unsqueeze(1)


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
