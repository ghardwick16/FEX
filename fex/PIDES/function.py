import numpy as np
import torch
from torch import sin, cos, exp
import math

scale = math.pi/4

def LHS_pde(u, x, dim_set):

    v = torch.ones(u.shape).cuda()
    z = x[:, 1:]  # gets integrated over R, should just be a lnspace of same size as x[1:]
    # since the first dimension of x is time
    ux = torch.autograd.grad(u, x, grad_outputs=v, create_graph=True)[0]
    integrand = u*torch.exp(z) - u - x[:, 1:]*(torch.exp(z) - 1)*ux[:, 1:]
    integral_dz = torch.trapezoid(integrand, z, dim=-1)
    LHS = ux[:, 0] + integral_dz
    return LHS

def RHS_pde(x):
    bs = x.size(0)
    return torch.zeros(bs, 1).cuda()

def true_solution(x): #for the most simple case, u(t,x) = x
    return x


unary_functions = [lambda x: 0*x**2,
                   lambda x: 1+0*x**2,
                   lambda x: x+0*x**2,
                   lambda x: x**2,
                   lambda x: x**3,
                   lambda x: x**4,
                   torch.exp,
                   torch.sin,
                   torch.cos,]

binary_functions = [lambda x,y: x+y,
                    lambda x,y: x*y,
                    lambda x,y: x-y]


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
                       '({}*cos({})+{})',]
                       # 'ref({})',
                       # 'exp(-({})**2/2)']

unary_functions_str_leaf= ['(0)',
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
                           '(cos({}))',]


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
    LHS = LHS_pde(function(x), x)
    RHS = RHS_pde(x)
    pde_loss = torch.nn.functional.mse_loss(LHS, RHS)

    '''
    boundary loss
    '''
    bc_points = torch.FloatTensor([[left], [right]]).cuda()
    bc_value = true_solution(bc_points)
    bd_loss = torch.nn.functional.mse_loss(function(bc_points), bc_value)

    print('pde loss: {} -- boundary loss: {}'.format(pde_loss.item(), bd_loss.item()))