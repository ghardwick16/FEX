import numpy as np
import torch
from torch import sin, cos, exp
import math


def get_paths(num_samples):
    mu = .4
    sigma = .25
    lam = .3
    steps = 50
    x_0 = 1

    # step 1: get jumps (i.e. jump times and jump sizes)
    jump_t_dist = torch.distributions.exponential.Exponential(lam)
    jump_dist = torch.distributions.normal.Normal(mu, sigma)
    jump_mat = torch.zeros(steps * num_samples).cuda()
    # jump_mat.requires_grad = True
    jump_t = 0
    prev = 0
    done = False
    while not done:
        jump_t += jump_t_dist.sample() * steps
        # check to make sure next jump time is within range
        if int(jump_t) < steps * num_samples:
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
    jump_mat = jump_mat.reshape((num_samples, steps))
    jump_term = torch.exp(jump_mat.reshape((num_samples, steps))) - lam / steps * (
                torch.exp(torch.tensor([mu + 1 / 2 * sigma ** 2]).cuda()) - 1)
    # step 2: calculate trajectory of x_t
    x_t = torch.empty_like(jump_mat).cuda()
    x_t[:, 0] = x_0
    for i in range(steps - 1):
        x_t[:, i + 1] = x_t[:, i] * jump_term[:, i]
    return x_t, jump_mat


def get_loss(func, true, x_t, jump_mat):
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

    num_pts = 10
    dt = (domain[1] - domain[0]) / x_t.shape[1]
    # Step 1:  loss1
    # Loss1, TD_error at t_n is given, in (2.20).  Note that we made a simplifying assumption
    # that only one jump occurs per time step, so the sum is a single value (in practice, if
    # two jumps occur, we simply sum them)
    z = torch.linspace(start=domain[0], end=domain[1], steps=num_pts).cuda()
    phi = 1 / (torch.sqrt(2 * torch.Tensor([math.pi]).cuda() * sigma)) * torch.exp(-.5 / sigma ** 2 * (z - mu) ** 2)
    t = torch.linspace(start=domain[0], end=domain[1], steps=x_t.shape[1]).repeat(x_t.shape[0], 1).cuda()
    # dims are (integration pts, batch_size, time steps, dims)
    tx_expz = torch.cat((t.unsqueeze(1).repeat(1, num_pts, 1).unsqueeze(3), (
            x_t.unsqueeze(1).repeat(1, num_pts, 1) * torch.exp(z).unsqueeze(0).repeat(num_samples, 1).unsqueeze(
        2).repeat(1, 1, steps)).unsqueeze(3)), dim=3)
    tx_exp_jumps = torch.cat((t.unsqueeze(2), (x_t * torch.exp(jump_mat)).unsqueeze(2)), dim=2)
    u_exp_jumps = u(tx_exp_jumps)
    u_tx = u(torch.cat((t.unsqueeze(2), x_t.unsqueeze(2)), dim=2))
    u_expz = torch.squeeze(u(tx_expz))
    n2 = lam * (torch.trapezoid(u_expz * phi.unsqueeze(1).repeat(x_t.shape[0], 1, x_t.shape[1]),
                                dx=(domain[1] - domain[0]) / num_pts, dim=1) - u_tx)
    loss1 = torch.mean((u_exp_jumps[..., :-1] - dt * n2[..., :-1] - u_tx[..., 1:]) ** 2)

    # Step 2:  loss2
    final_xt = torch.cat((t[:, -1].unsqueeze(1), x_t[:, -1].unsqueeze(1)), dim=1)
    final_xt.requires_grad = True
    u_final = u(final_xt)
    true_final = true(final_xt)
    loss2 = torch.mean(u_final - true_final)

    # Step 3: loss3
    v = torch.ones(u_final.shape)
    du = torch.autograd.grad(u_final, final_xt, grad_outputs=v, create_graph=True)[0]
    dg = torch.autograd.grad(true_final, final_xt, grad_outputs=v, create_graph=True)[0]
    loss3 = torch.mean(du[:, 1:] - dg[:, 1:])

    # Step 4: add them up
    loss = loss1 + loss2 + loss3

    return loss


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
