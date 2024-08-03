import numpy as np
import torch
from torch import sin, cos, exp
import math
'''
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
    jump_term = torch.exp(jump_mat.reshape((num_samples, steps, dims))) - lam / steps * (
            torch.exp(torch.tensor([mu + 1 / 2 * sigma ** 2]).cuda()) - 1)
    pre_computed = jump_term - lam * mu / steps

    # step 2: calculate trajectories of x_t, y_t given function u
    x_t = torch.empty_like(jump_mat).cuda()
    x_t[:, 0, :] = x_0
    for i in range(steps - 1):
        x_t[:, i + 1, :] = x_t[:, i, :] * pre_computed[:, i, :]
    return x_t, jump_mat
'''
def get_pts(num_samples, dims):
  x = torch.empty((num_samples, 50, dims)).cuda()
  torch.randn(num_samples, 50, dims, out=x)
  x.requires_grad = True
  return x

def get_loss(func, true, x_t):  # changed to let this use the pair (learnable_tree, bs_action) for computation directly
    mu = .4
    sigma = .25
    lam = .3
    num_traps = 50  # traps super low to keep code faster
    t = torch.linspace(start=-1, end=2, steps=x_t.shape[1]).repeat(x_t.shape[0], 1).unsqueeze(2).cuda()
    tx = torch.cat((t, x_t), dim=2)
    z = torch.linspace(0, 1, num_traps).cuda()
    x = torch.squeeze(tx[..., 1:]).cuda()
    nu = (lam / torch.sqrt(2 * torch.Tensor([math.pi]) * sigma).cuda() * torch.exp(-.5 * ((z - mu) / sigma).cuda() ** 2)).unsqueeze(0).unsqueeze(0).repeat(tx.shape[0], tx.shape[1], 1)
    x_expz = x.unsqueeze(-1).repeat(1, 1, num_traps) * torch.exp(z.unsqueeze(0).unsqueeze(0).repeat(x.shape[0], x.shape[1], 1))
    tx_expz = torch.stack((t.repeat(1,1, num_traps), x_expz), dim=-1)
    ### We have two cases:  either we pass in the condidate function in the form
    ### (learnable_tree, bs_action) or the true function (for measuring performance)
    if type(func) is tuple:
        learnable_tree = func[0]
        bs_action = func[1]
        u_func = lambda y: learnable_tree(y, bs_action)
    else:
        u_func = lambda y: func(y)

    u = u_func(tx)
    #print(torch.squeeze(u_func(tx_expz)).shape)
    #print(u.repeat(1,1,num_traps).shape)
    #print(f'x: {x.shape}')

    #print(f'nu: {nu.shape}')
    '''
    x_exp_mu = x*torch.exp(torch.tensor(mu))
    tx_exp_mu = torch.cat((t, x_exp_mu.unsqueeze(-1)), dim=-1)
    u_exp_mu = u_func(tx_exp_mu).squeeze()

    v = torch.ones_like(x).cuda()
    u_exp_mu = torch.autograd.grad(u_exp_mu, tx_exp_mu, grad_outputs=v, create_graph=True)[0]
    v1 = torch.ones_like(tx_exp_mu)
    dd_exp_mu = torch.autograd.grad(d_exp_mu, tx_exp_mu, grad_outputs=v1, create_graph=True)[0][...,-1]
    expect = u_exp_mu + 1/2*dd_exp_mu*sigma**2
    '''

    u_expz = torch.squeeze(u_func(tx_expz))
    v = torch.ones(u.shape).cuda()
    deriv_u = torch.autograd.grad(u, tx, grad_outputs=v, create_graph=True)[0]
    #print(du.shape)
    ut = deriv_u[..., 0].cuda()
    ux = deriv_u[..., 1].unsqueeze(-1).repeat(1, 1, num_traps).cuda()
    #ux = deriv_u[..., 1]
    #print(f'ux: {ux.shape}')
    exp_z = torch.exp(z).unsqueeze(0).unsqueeze(0).repeat(tx.shape[0], tx.shape[1], 1).cuda()
    #print(f'exp_z: {exp_z.shape}'
    integrand = (u_expz - u.repeat(1, 1, num_traps) - x.unsqueeze(-1).repeat(1, 1, num_traps) * (
            exp_z - 1) * ux) * nu
    integral_dz = torch.trapezoid(integrand, z, dim=-1)
    LHS = ut + integral_dz
    RHS = torch.zeros(tx.size(0), 1).cuda()

    loss1 = torch.mean((LHS - RHS) ** 2)
    # print('loss1:', loss1)

    # Step 2:  loss2
    final_xt = torch.cat((t[:, -1, :].unsqueeze(1), x_t[:, -1, :].unsqueeze(1)), dim=2).cuda()
    u_final = u_func(final_xt).squeeze()
    true_final = true(final_xt).squeeze()
    loss2 = torch.mean((u_final - true_final) ** 2)

    # Step 3: loss3
    #v2 = torch.ones(u_final.shape).cuda()
    #v3 = torch.ones_like(u_final).cuda()
    #du = torch.autograd.grad(u_final, final_xt, grad_outputs=v2, create_graph=True)[0]
    #dg = torch.autograd.grad(true_final, final_xt, grad_outputs=v3, create_graph=True)[0]
    #loss3 = torch.mean((du[:, :, 1:] - dg[:, :, 1:]) ** 2)

    loss = loss1 + loss2
    return loss

def get_errors(learnable_tree, bs_action, dims):
    u_func = lambda y: learnable_tree(y, bs_action)
    mse_list = []
    denom = []
    relative_num = []
    relative_denom = []
    for _ in range(1000):
        x = get_pts(500, 2)
        true = true_solution(x).squeeze()
        u = u_func(x).squeeze()
        mse_list.append(torch.mean((true - u) ** 2))
        relative_num.append(torch.mean(torch.abs(true - u)))
        relative_denom.append(torch.mean(torch.abs(true)))
        denom.append(torch.mean(true**2))
    relative_l2 = torch.sqrt(sum(mse_list))/torch.sqrt(sum(denom))
    relative = sum(relative_num)/sum(relative_denom)
    mse = 1 / 1000 * sum(mse_list)
    return relative_l2, relative, mse


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
