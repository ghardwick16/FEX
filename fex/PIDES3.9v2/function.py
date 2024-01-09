import numpy as np
import torch
from torch import sin, cos, exp
import math


def get_paths(num_samples, dims):
    x_0 = 1
    mu = .4
    sigma = .25
    lam = .3
    steps = 50
    domain = [0, 1]

    # step 1: get jumps (i.e. jump times and jump sizes)
    jump_t_dist = torch.distributions.exponential.Exponential(lam)
    jump_dist = torch.distributions.normal.Normal(mu, sigma)
    jump_mat = torch.zeros(steps * num_samples * dims).cuda()
    jump_t = 0
    prev = 0
    done = False
    while not done:
        jump_t += jump_t_dist.sample() * steps
        # check to make sure next jump time is within range
        if int(jump_t) < steps * num_samples * dims:
            # check to see if we have passed max steps, if not, continue,
            # otherwise, we discard the update to jump_t, reset at the nearest multiple of
            # max steps, and continue (since this is now a new trajectory)
            if int(jump_t / steps) == int(prev / steps):
                jump_mat[int(jump_t)] = jump_dist.sample()
                prev = jump_t
            else:
                prev = jump_t = int(jump_t / steps) * steps
        else:
            done = True
    # by factoring out an x_t from the calculation below we can subtract and add values to the jump matrix to make the computation a bit faster
    jump_mat = jump_mat.reshape((num_samples, steps, dims))
    jump_term = torch.exp(jump_mat.reshape((num_samples, steps, dims))) - lam / steps * (
                torch.exp(torch.tensor([mu + 1 / 2 * sigma ** 2]).cuda()) - 1)
    # step 2: calculate trajectories of x_t, y_t given function u
    x_t = torch.empty_like(jump_mat).cuda()
    x_t[:, 0, :] = x_0
    for i in range(steps - 1):
        x_t[:, i + 1, :] = x_t[:, i, :] * jump_term[:, i, :]
    return x_t, jump_mat


def get_loss(func, true, x_t, jump_mat):
    left = 0
    right = 1
    dims = x_t.shape[-1]
    num_samples = x_t.shape[0]
    if type(func) is tuple:
        learnable_tree = func[0]
        bs_action = func[1]
        u = lambda y: learnable_tree(y, bs_action)
    else:
        u = lambda y: func(y)

    # constants for loss calculations
    mu = .4
    sigma = .25
    lam = .3
    steps = 50
    domain = [0, 1]
    theta = .3

    num_pts = 10
    dt = (domain[1] - domain[0]) / x_t.shape[1]
    # Step 1:  loss1
    # Loss1, TD_error at t_n is given, in (2.20).  Note that we made a simplifying assumption
    # that only one jump occurs per time step, so the sum is a single value (in practice, if
    # two jumps occur, we simply sum them)
    z = torch.linspace(start=left, end=right, steps=num_pts).cuda()
    phi = 1 / (torch.sqrt(2 * torch.Tensor([math.pi]).cuda() * sigma)) * torch.exp(-.5 / sigma ** 2 * (z - mu) ** 2)
    phi = phi.unsqueeze(0).unsqueeze(0).repeat(x_t.shape[0], x_t.shape[1], 1)
    t = torch.linspace(start=domain[0], end=domain[1], steps=x_t.shape[1]).repeat(x_t.shape[0], 1).unsqueeze(2).cuda()
    tx = torch.cat((t, x_t), dim=2)
    # dims are (integration pts, batch_size, time steps, dims)
    z_large = z.unsqueeze(-1).unsqueeze(0).unsqueeze(0).repeat(x_t.shape[0], x_t.shape[1], 1, dims)
    tx_shift = tx.unsqueeze(2).repeat(1, 1, z.shape[0], 1)
    tx_shift[:, :, :, 1:] += z_large
    u_shift = u(tx_shift).squeeze()
    u_tx = u(tx).squeeze()
    # (t, x_j + G(x,z))
    tx_z = tx
    tx_z[..., 1:] += jump_mat
    u_tx_z = u(tx_z)
    n2 = lam * (torch.trapezoid(u_shift * phi, dx=(right - left) / num_pts, dim=-1) - u_tx)
    f = lam * mu ** 2 + theta ** 2
    loss1 = torch.mean((-f * dt + u_tx_z[..., :-1] - dt * n2[..., :-1] - u_tx[..., 1:]) ** 2)

    # Step 2:  loss2
    final_xt = torch.cat((t[:, :, -1].unsqueeze(1), x_t[:, :, -1].unsqueeze(1)), dim=1).cuda()
    final_xt.requires_grad = True
    u_final = u(final_xt)
    true_final = true(final_xt)
    loss2 = torch.mean(u_final - true_final)

    # Step 3: loss3
    v = torch.ones(u_final.shape).cuda()
    du = torch.autograd.grad(u_final, final_xt, grad_outputs=v, create_graph=True)[0]
    dg = torch.autograd.grad(true_final, final_xt, grad_outputs=v, create_graph=True)[0]
    loss3 = torch.mean(du[:, 1:] - dg[:, 1:])

    # Step 4: add them up
    loss = loss1 + loss2 + loss3

    return loss


def true_solution(tx):  # for the most simple case, u(t,x) = x
    return torch.mean(tx[...,1:]**2, dim=-1)


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
