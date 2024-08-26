import numpy as np
import torch
from torch import sin, cos, exp
import math
from torch.autograd.functional import hessian

#This function samples points from the domain.  Dimension of output is a little weird since this was carried over from
#a previous iteration of the code when sampled SDE's (as in the PIDE paper).  Note that this samples points of X, our
#space variable.  T is not sampled randomly, we use a linspace for it to ensure that we cover the domain of time well

def get_pts(num_samples, dims):
  x = torch.empty((num_samples, 50, dims)).cuda()
  torch.randn(num_samples, 50, dims, out=x)
  x.requires_grad = True
  return x

#Function to compare RHS and LHS of PIDE.  Note that I have added sigma (variance) as an input so that I can easily
#change it run to run

def get_loss(func, true, x_t, sigma):
    mu = .1
    lam = .3
    theta = .2
    left = 0
    right = 1
    epsilon = .05

    dims = x_t.shape[-1]

    t = torch.linspace(start=left, end=right, steps=x_t.shape[1]).repeat(x_t.shape[0], 1).unsqueeze(2).cuda()
    tx = torch.cat((t, x_t), dim=2)

    # We have two cases:  either we pass in the condidate function in the form
    # (learnable_tree, bs_action) or the true function
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
    v1 = torch.ones_like(x_t).cuda()

    # Here we calculate sigma*sigma^T*H(u)
    if du.requires_grad:
        hess = torch.autograd.grad(ux, tx, grad_outputs=v1, create_graph=True)[0][..., 1:]
        hes_diag[..., 0] = hess[...,0]
        hes_diag[...,1:] = 2*hess[...,1:]
    else:
        hes_diag = torch.zeros_like(du).cuda()
    trace_hessian = torch.sum(hes_diag, dim=-1)

    ### INT (u(t, x+z) - u(t,x)) d nu  (=n2 in PIDES paper)
    tx_shift = torch.empty_like(tx).cuda()
    tx_shift[:, :, :] = tx[:, :, :]
    tx_shift[..., 1:] += mu

    # 1st derivs
    v = torch.ones_like(u_tx).cuda()
    u_tx_shift = u(tx_shift).squeeze()
    du_tx_shift = torch.autograd.grad(u_tx_shift, tx_shift, grad_outputs=v, create_graph=True)[0]

    # 2nd derivs
    hes_diag1 = torch.empty_like(x_t).cuda()
    v1 = torch.ones_like(x_t).cuda()
    if du.requires_grad:
        hes_diag1[..., :] = torch.autograd.grad(du_tx_shift[..., 1:], tx_shift, grad_outputs=v1, create_graph=True)[0][
                            ..., 1:]
    else:
        hes_diag1 = torch.zeros_like(du_tx_shift).cuda()

    expect = u_tx_shift + 1 / 2 * torch.sum(hes_diag1, dim=-1) * sigma ** 2
    n2 = lam * (expect - u_tx)

    ### INT z * grad(u) d nu
    z_vec = lam * mu * torch.ones_like(ux)
    n3 = torch.sum(ux * z_vec, dim=-1)

    ### norm of x
    x_norm = torch.sqrt(torch.sum(tx[..., 1:] ** 2, dim=-1))

    # Step 1:  loss1
    LHS = ut + epsilon / 2 * x_norm * torch.sum(tx[..., 1:] * ux, dim=-1) + 1 / 2 * theta ** 2 * trace_hessian + (
                n2 - n3)
    RHS = lam * (mu ** 2 + sigma ** 2) + (2 * dims - 1) / dims * theta ** 2 + epsilon / dims * x_norm ** 3
    loss1 = torch.mean((LHS - RHS) ** 2)

    # Step 2:  loss2
    final_xt = torch.cat((t[:, -1, :].unsqueeze(1), x_t[:, -1, :].unsqueeze(1)), dim=2).cuda()
    u_final = u(final_xt).squeeze()
    true_final = true(final_xt).squeeze()
    loss2 = torch.mean((u_final - true_final) ** 2)

    return loss1 + loss2


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


def true_solution(tx):
    return 1/(tx.shape[-1]-1)*torch.sum(tx[..., 1:] ** 2, dim=-1)


unary_functions = [lambda x: 0 * x ** 2,
                   lambda x: 1 + 0 * x ** 2,
                   lambda x: x + 0 * x ** 2,
                   lambda x: x ** 2,
                   lambda x: x ** 3,
                   lambda x: x ** 4,
                   torch.exp,
                   torch.sin,
                   torch.cos, ]

unary_function_derivatives = [lambda x: 0 * x ** 2,
                              lambda x: 0 + 0 * x ** 2,
                              lambda x: 1 + 0 * x ** 2,
                              lambda x: 2 * x,
                              lambda x: 3 * x ** 2,
                              lambda x: 4 * x ** 3,
                              torch.exp,
                              torch.cos,
                              lambda x: -1 * torch.sin(x), ]

unary_function_2nd_derivatives = [lambda x: 0 * x ** 2,
                                  lambda x: 0 + 0 * x ** 2,
                                  lambda x: 0 + 0 * x ** 2,
                                  lambda x: 2 + 0 * x **2,
                                  lambda x: 6 * x,
                                  lambda x: 12 * x ** 2,
                                  torch.exp,
                                  lambda x: -1 * torch.sin(x),
                                  lambda x: -1 * torch.cos(x), ]

binary_functions = [lambda x, y: x + y,
                    lambda x, y: x * y,
                    lambda x, y: x - y]

#binary_functions = [lambda x, y: x*y]

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
#binary_functions_str = ['(({})*({}))']

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
