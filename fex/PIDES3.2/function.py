import numpy as np
import torch
from torch import sin, cos, exp
import math

def LHS_pde(func, tx):  # changed to let this use the pair (learnable_tree, bs_action) for computation directly
    mu = .4
    sigma = .25
    lam = .3
    num_traps = 5  # traps super low to keep code faster
    z = torch.linspace(0, 1, num_traps).cuda()
    t = torch.squeeze(tx[..., 0]).cuda()
    x = torch.squeeze(tx[..., 1:]).cuda()

    nu = lam / torch.sqrt(2 * torch.Tensor([math.pi]) * sigma).cuda() * torch.exp(-.5 * ((z - mu) / sigma).cuda() ** 2)
    tx_expz = torch.stack((t.repeat(z.shape[0], 1).T, torch.outer(x, torch.exp(z).cuda())), dim=2)
    ### We have two cases:  either we pass in the condidate function in the form
    ### (learnable_tree, bs_action) or the true function (for measuring performance)
    if type(func) is tuple:
        learnable_tree = func[0]
        bs_action = func[1]
        u_func = lambda y: learnable_tree(y, bs_action)
    else:
        u_func = lambda y: func(y)

    u = u_func(tx)

    u_expz = torch.squeeze(u_func(tx_expz))
    v = torch.ones(u.shape).cuda()
    du = torch.autograd.grad(u, tx, grad_outputs=v, create_graph=True)[0]
    ut = du[:, 0].cuda()
    ux = du[:, 1].cuda()
    exp_z = torch.exp(z).cuda()
    print(u_expz.shape)
    print(u.repeat(1, z.shape[0]).shape)
    print(x.repeat(1, z.shape[0]).shape)
    print(exp_z.shape)
    print(ux.shape)
    print(nu.shape)
    integrand = (u_expz - u.repeat(1, z.shape[0]) - x.repeat(1, z.shape[0]) * (
            exp_z.repeat(tx.shape[0], 1) - 1) * ux.repeat(z.shape[0], 1).T) * nu.repeat(tx.shape[0], 1)
    integral_dz = torch.trapezoid(integrand, z, dim=1)
    return ut + integral_dz

def RHS_pde(tx):
    bs = tx.size(0)
    return torch.zeros(bs, 1).cuda()


def true_solution(tx):  # for the most simple case, u(t,x) = x
    return tx[..., 1:]


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
