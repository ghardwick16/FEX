import numpy as np
import torch
from torch import sin, cos, exp
import math
import itertools

def get_paths(num_samples, dims):
    x_0 = 1
    mu = .1
    sigma = 1e-4
    lam = .3
    theta = .3
    steps = 50
    domain = [0, 1]

    # step 1: get jumps (i.e. jump times and jump sizes)
    jump_t_dist = torch.distributions.exponential.Exponential(lam)
    jump_dist = torch.distributions.multivariate_normal.MultivariateNormal(mu * torch.ones(dims),
                                                                           sigma * torch.eye(dims))
    jump_mat = torch.zeros(steps * num_samples, dims).cuda()
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
                jump_mat[int(jump_t), :] = jump_dist.sample()
                prev = jump_t
            else:
                prev = jump_t = int(jump_t / steps) * steps
        else:
            done = True
    # by factoring out an x_t from the calculation below we can subtract and add values to the jump matrix to make the computation a bit faster
    jump_mat = jump_mat.reshape((num_samples, steps, dims))
    #jump_term = torch.exp(jump_mat.reshape((num_samples, steps, dims))) - lam / steps * (
    #        torch.exp(torch.tensor([mu + 1 / 2 * sigma ** 2]).cuda()) - 1)
    brownian_dist = torch.distributions.normal.Normal(0, 1 / steps)
    brownian = brownian_dist.sample((num_samples, steps, dims)).cuda()
    pre_computed = theta * brownian + jump_mat - lam * mu / steps

    # step 2: calculate trajectories of x_t, y_t given function u
    x_t = torch.empty_like(jump_mat).cuda()
    x_t[:, 0, :] = x_0
    for i in range(steps - 1):
        x_t[:, i + 1, :] = x_t[:, i, :] + pre_computed[:, i, :]
    return x_t, jump_mat, brownian


def get_loss(func, true, x_t, jump_mat, brownian):
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
    mu = .1
    sigma = 1e-4
    lam = .3
    steps = 50
    domain = [0, 1]
    theta = .3
    dt = 1 / steps
    int_pts = 50

    # Step 1:  loss1
    # Loss1, TD_error at t_n is given, in (2.20).  Note that we made a simplifying assumption
    # that only one jump occurs per time step, so the sum is a single value (in practice, if
    # two jumps occur, we simply sum them)

    t = torch.linspace(start=domain[0], end=domain[1], steps=x_t.shape[1]).repeat(x_t.shape[0], 1).unsqueeze(2).cuda()
    tx = torch.cat((t, x_t), dim=2)
    u_tx = u(tx).squeeze()

    # (t, x_j + G(x,z))
    tx_z = tx + torch.cat((torch.zeros(num_samples, steps, 1).cuda(), jump_mat), dim=2)
    # tx_z[..., 1:] += jump_mat
    u_tx_z = u(tx_z).squeeze()
    # determine if det(sigma) is close enough to 0 to use dirac delta instead of phi for integration step:
    if dims >= 3:
        z = mu * torch.ones(dims).cuda()
        delta_fn = True
    else:
        delta_fn = False
        z = torch.tensor([pt for pt in itertools.product([x / int_pts for x in range(int_pts)], repeat=dims)]).cuda()
        phi = (2 * torch.Tensor([math.pi]).cuda() * sigma) ** (-dims / 2) * torch.exp(
            -.5 / sigma ** 2 * torch.sum((z - mu) ** 2, dim=-1))
        phi = phi.unsqueeze(0).repeat(x_t.shape[0], x_t.shape[1], 1)

    if not delta_fn:
        # dims are (batch_size, time steps, integration points, dims)
        z_large = z.unsqueeze(0).unsqueeze(0).repeat(x_t.shape[0], x_t.shape[1], 1, 1).cuda()
        tx_shift = tx.unsqueeze(2).repeat(1, 1, int_pts ** dims, 1)
        tx_shift[:, :, :, 1:] += z_large
        u_shift = u(tx_shift)
        u_tx_large = u_tx.unsqueeze(2).repeat(1, 1, u_shift.shape[-1])
        n2 = lam * (1 / (int_pts ** dims) * torch.sum((u_shift - u_tx_large) * phi, dim=-1))
    else:
        tx_shift = torch.empty_like(tx).cuda()
        tx_shift[:, :, :] = tx[:, :, :]
        tx_shift[..., 1:] += mu
        n2 = lam * (u(tx_shift).squeeze() - u_tx)
    f = lam * (mu**2 + sigma**2) + theta**2
    #f = lam *dims/2 * (mu ** 2 + sigma**2) + 1/2*theta ** 2
    v = torch.ones(u_tx.shape).cuda()
    grad_u = torch.autograd.grad(u_tx, tx, grad_outputs=v, create_graph=True)[0][:, :, 1:]
    loss1 = torch.mean((2/dims*(-f * dt + theta * torch.sum(grad_u[:, :-1, :] * brownian[:, :-1, :], dim=2) + u_tx_z[...,:-1] - dt * n2[...,:-1] - u_tx[..., 1:])) ** 2)

    # Step 2:  loss2
    final_xt = torch.cat((t[:, -1, :].unsqueeze(1), x_t[:, -1, :].unsqueeze(1)), dim=2).cuda()
    u_final = u(final_xt).squeeze()
    true_final = true(final_xt).squeeze()
    loss2 = torch.mean((u_final - true_final) ** 2)

    # Step 3: loss3
    v = torch.ones(u_final.shape).cuda()
    du = torch.autograd.grad(u_final, final_xt, grad_outputs=v, create_graph=True)[0]
    dg = torch.autograd.grad(true_final, final_xt, grad_outputs=v, create_graph=True)[0]
    loss3 = torch.mean((du[:, :, 1:] - dg[:, :, 1:]) ** 2)

    # Step 4: add them up
    loss = loss1 + loss2 + loss3

    return loss
'''
def get_loss(func, true, x_t, jump_mat, brownian):  # changed to let this use the pair (learnable_tree, bs_action) for computation directly
    # parameters for the LHS
    mu = .1
    sigma = 1e-4
    lam = .3
    theta = .3
    left = 0
    right = 1

    dims = x_t.shape[-1]

    t = torch.linspace(start=left, end=right, steps=x_t.shape[1]).repeat(x_t.shape[0], 1).unsqueeze(2).cuda()
    tx = torch.cat((t, x_t), dim=2)

    ### We have two cases:  either we pass in the condidate function in the form
    ### (learnable_tree, bs_action) or the true function (for measuring performance)
    if type(func) is tuple:
        learnable_tree = func[0]
        bs_action = func[1]
        u = lambda y: learnable_tree(y, bs_action)
    else:
        u = lambda y: func(y)

    u_tx = u(tx).squeeze()

    # get derivatives
    v = torch.ones(u_tx.shape).cuda()
    du = torch.autograd.grad(u_tx, tx, grad_outputs=v, create_graph=True)[0]
    ut = du[..., 0]
    ux = torch.squeeze(du[..., 1:])

    hes_diag = torch.empty_like(x_t).cuda()
    if du.requires_grad:
      hes_diag[...,:] = torch.autograd.grad(du[...,1:], tx, grad_outputs=torch.ones_like(x_t), create_graph=True)[0][...,1:]
    else:
      hes_diag = torch.zeros_like(du).cuda()
    trace_hessian = torch.sum(hes_diag, dim=-1)

    ### INT (u(t, x+z) - u(t,x)) d nu  (=n2 in PIDES paper)
    tx_shift = torch.empty_like(tx).cuda()
    tx_shift[:, :, :] = tx[:, :, :]
    tx_shift[..., 1:] += mu
    n2 = lam * (u(tx_shift).squeeze() - u_tx)

    ### INT z * grad(u) d nu
    z_vec = lam*mu*torch.ones_like(ux)
    n3 = torch.sum(ux * z_vec, dim=-1)

    #print('theta**2:', torch.mean(1 / 2 * theta ** 2 * trace_hessian))
    LHS = ut + 1 / 2 * theta ** 2 * trace_hessian + (n2 - n3)
    RHS = lam*dims/2*(mu**2 + sigma**2) + dims/2*theta**2
    #THS = lam*(mu**2 + sigma**2) + theta**2
    loss1 = torch.sum((LHS - RHS) ** 2)
    #print('loss1:', loss1)

    # Step 2:  loss2
    final_xt = torch.cat((t[:, -1, :].unsqueeze(1), x_t[:, -1, :].unsqueeze(1)), dim=2).cuda()
    u_final = u(final_xt).squeeze()
    true_final = true(final_xt).squeeze()
    loss2 = torch.sum((u_final - true_final) ** 2)

    # Step 3: loss3
    v = torch.ones(u_final.shape).cuda()
    du = torch.autograd.grad(u_final, final_xt, grad_outputs=v, create_graph=True)[0]
    dg = torch.autograd.grad(true_final, final_xt, grad_outputs=v, create_graph=True)[0]
    loss3 = torch.sum((du[:, :, 1:] - dg[:, :, 1:]) ** 2)

    loss = loss1 + loss2 + loss3

    return loss
'''
def true_solution(tx):  # here the true solution is 1/d ||x||^2 i.e. the mean of the 2-norm squared
    # of the space dimensions

    # UNDONE:
    # modified true solution to be 1/2||x||^2
    #return 1 / 2 * torch.sum(tx[..., 1:] ** 2, dim=-1)
    return 1/(tx.shape[-1] - 1) * torch.sum(tx[...,1:]**2, dim=-1)


def get_errors(learnable_tree, bs_action, dims):
    u = lambda y: learnable_tree(y, bs_action)
    pts_per_dim = int(20000 / dims)
    mse_list = []
    denom = []
    relative_num = []
    relative_denom = []
    for _ in range(1000):
        t = torch.rand(pts_per_dim, 1).cuda()
        x1 = torch.rand(pts_per_dim, dims).cuda()
        x = torch.cat((t, x1), 1)
        #print(true_solution(x).shape)
        #print(u(x).shape)
        mse_list.append(torch.mean((true_solution(x) - u(x).squeeze()) ** 2))
        relative_num.append(torch.mean(torch.abs(true_solution(x) - u(x).squeeze())))
        relative_denom.append(torch.mean(torch.abs(true_solution(x))))
        denom.append(torch.mean(true_solution(x)**2))
    relative_l2 = torch.sqrt(sum(mse_list))/torch.sqrt(sum(denom))
    relative = sum(relative_num)/sum(relative_denom)
    mse = 1 / 1000 * sum(mse_list)
    return relative_l2, relative, mse


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
