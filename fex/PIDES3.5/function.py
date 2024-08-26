import numpy as np
import torch
from torch import sin, cos, exp
import math

def get_pts(num_samples, dims):
  x = torch.empty((num_samples, 50, dims)).cuda()
  torch.randn(num_samples, 50, dims, out=x)
  x.requires_grad = True
  return x

def get_loss(func, true, x_t):  # changed to let this use the pair (learnable_tree, bs_action) for computation directly
    mu = .4
    sigma = .25
    lam = .3
    epsilon = .25
    num_traps = 25  # traps super low to keep code faster
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

    u_expz = torch.squeeze(u_func(tx_expz))
    v = torch.ones(u.shape).cuda()
    deriv_u = torch.autograd.grad(u, tx, grad_outputs=v, create_graph=True)[0]
    #print(du.shape)
    ut = deriv_u[..., 0].cuda()
    ux = deriv_u[..., 1].cuda()
    #print(f'ux: {ux.shape}')
    exp_z = torch.exp(z).unsqueeze(0).unsqueeze(0).repeat(tx.shape[0], tx.shape[1], 1).cuda()
    #print(f'exp_z: {exp_z.shape}')
    integrand = (u_expz - u.repeat(1, 1, num_traps) - x.unsqueeze(-1).repeat(1, 1, num_traps) * (
            exp_z - 1) * ux.unsqueeze(-1).repeat(1, 1, num_traps)) * nu
    integral_dz = torch.trapezoid(integrand, z, dim=-1)
    LHS = ut + epsilon*x*ux + integral_dz
    RHS = epsilon*x

    loss1 = torch.mean((LHS - RHS) ** 2)
    # print('loss1:', loss1)

    # Step 2:  loss2
    final_xt = torch.cat((t[:, -1, :].unsqueeze(1), x_t[:, -1, :].unsqueeze(1)), dim=2).cuda()
    u_final = u_func(final_xt).squeeze()
    true_final = true(final_xt).squeeze()
    loss2 = torch.mean((u_final - true_final) ** 2)

    loss = (loss1 + loss2)  # 'loss per path per dim'
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
