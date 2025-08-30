import numpy as np
import torch
from torch import sin, cos, exp
import math
import itertools

def get_pts(num_samples, dims):
    x = torch.empty((num_samples, 1)).cuda()
    torch.rand(num_samples, 1, out=x)
    x.requires_grad = True

    t = torch.empty((num_samples, 1)).cuda()
    torch.rand(num_samples, 1, out=t)
    t.requires_grad = True
    t = t*2

    out = torch.cat((t,x), dim = 1)

    return out

def interp_torch(x, xp, fp):
    """
    One-dimensional linear interpolation on a tensor.

    Args:
        x (torch.Tensor): The x-coordinates at which to evaluate the interpolated values.
        xp (torch.Tensor): The x-coordinates of the data points, must be increasing.
        fp (torch.Tensor): The y-coordinates of the data points, same length as xp.

    Returns:
        torch.Tensor: The interpolated values, same size as x.
    """
    if len(xp) != len(fp):
        raise ValueError("xp and fp must have the same length")

    indexes = torch.searchsorted(xp, x)

    indexes = torch.clamp(indexes, 1, len(xp) - 1)

    x_low = xp[indexes - 1]
    x_high = xp[indexes]
    y_low = fp[indexes - 1]
    y_high = fp[indexes]

    weights = (x - x_low) / (x_high - x_low)

    y = torch.lerp(y_low, y_high, weights)
    return y


def carr_madan_fft_vg_put_surface_fixed_K(
    S0_list, T_list, K_fixed,
    r, q, sigma, nu, theta,
    alpha=1.5, N=4096, eta=0.25, device='cuda'
):
    """
    Returns:
      - S0_grid: (nS,) tensor of spot prices
      - T_grid:  (nT,) tensor of maturities
      - P: (nT, nS) tensor of put prices at fixed strike K_fixed
    """
    S0_list = torch.tensor(S0_list, device=device)
    T_list = torch.tensor(T_list, device=device)

    lambd = 2 * np.pi / (N * eta)
    b = N * lambd / 2
    u = torch.arange(N, device=device) * eta
    k_grid = -b + torch.arange(N, device=device) * lambd
    K_grid = torch.exp(k_grid)

    # Simpson weights
    simpson_weights = torch.ones(N, device=device)
    simpson_weights[1::2] = 4
    simpson_weights[2:-1:2] = 2
    simpson_weights *= eta / 3

    put_surface = []

    for S0 in S0_list:
        puts_T = []

        for T in T_list:
            omega = (1 / nu) * torch.log(torch.tensor([1 - theta * nu - 0.5 * sigma**2 * nu]).cuda())
            mu = torch.log(S0) + (r - q + omega) * T

            shifted_u = u - 1j * (alpha + 1)
            term = 1 - 1j * theta * nu * shifted_u + 0.5 * sigma**2 * nu * shifted_u**2
            phi = torch.exp(1j * shifted_u * mu - (1 / nu) * T * torch.log(term))

            numerator = torch.exp(-r * T) * phi
            denominator = alpha**2 + alpha - u**2 + 1j * (2 * alpha + 1) * u
            integrand = numerator / denominator * torch.exp(1j * u * b) * simpson_weights

            fft_output = torch.fft.fft(integrand).real
            C_full = torch.exp(-alpha * k_grid) / np.pi * fft_output

            # Interpolate for call price at K
            call_K = interp_torch(torch.tensor(K_fixed).log().cuda(), k_grid, C_full)

            # Use put-call parity to get the put price
            put_K = call_K + K_fixed * torch.exp(-r * T) - S0 * torch.exp(-q * T)
            puts_T.append(put_K)

        put_surface.append(torch.stack(puts_T))

    put_surface = torch.stack(put_surface, dim=1)  # shape (nT, nS)
    put_surface.cuda()
    return S0_list, T_list, put_surface

def generate_FFT_sol(t_points, x_pts):
    S0_vals = torch.linspace(100, 400, x_pts)  # Spot prices
    T_vals = torch.linspace(0, 2.0, t_points)  # Maturities
    K = 200.0  # Fixed strike

    r = 0.05
    q = 0.02
    sigma = 0.4
    nu = 0.4
    theta = -0.4

    S0_tens, T_tens, P = carr_madan_fft_vg_put_surface_fixed_K(
        S0_vals, T_vals, K, r, q, sigma, nu, theta
    )
    S_mesh, T_mesh = torch.meshgrid(S0_tens, T_tens, indexing='ij')

    S_mesh = S_mesh.unsqueeze(-1)
    T_mesh = T_mesh.unsqueeze(-1)
    points = torch.cat((T_mesh, S_mesh), dim=-1)
    P = P.T.unsqueeze(-1)
    FFT_sol = torch.cat((points, P), dim=-1)
    FFT_sol = FFT_sol.reshape((points.shape[0] * points.shape[1], FFT_sol.shape[-1]))

    return FFT_sol


def precomputed_lams(theta, sigma, nu):

    #Lambdas for k(y):
    lam_p = math.sqrt((theta ** 2) / (sigma ** 4) + 2 / (sigma ** 2 * nu)) - theta / sigma ** 2
    lam_n = math.sqrt((theta ** 2) / (sigma ** 4) + 2 / (sigma ** 2 * nu)) + theta / sigma ** 2

    return lam_p, lam_n

def k(y, lam_p, lam_n, nu):
    pos_idxs = torch.where(y < 0, torch.tensor(0, dtype=y.dtype), y).nonzero().cuda()
    neg_idxs = torch.where(y > 0, torch.tensor(0, dtype=y.dtype), y).nonzero().cuda()

    out = torch.empty_like(y).cuda()

    out[pos_idxs] = torch.exp(-lam_p*y[pos_idxs])/(nu*y[pos_idxs])
    out[neg_idxs] = torch.exp(-lam_n*torch.abs(y[neg_idxs]))/(nu*torch.abs(y[neg_idxs]))

    return out


def precomputed_expect(lam_p, lam_n, nu):
  ##Going to write a function to comput E(e^y - 1) generally eventually, but for now I just computed it on Mathematica for the specific
  ##paramters used

  val = 0.129163 -0.0438857142847

  return val


'''
## here I am calling the E(e^y - 1) 'c' for shorthand.  It is a precomputed value, as are lam_p, lam_n
def get_loss(func, lam_p, lam_n, nu, c, x):
    # parameters:
    r = .05
    q = .02
    K = 200
    # mu is E[y] given y~k(y), y_sigma is E[y^2] - E[y]^2 for y~k(y)
    mu = 8/125
    y_sigma = 496/15625

    num_samples = x.shape[0]

    # We have two cases:  either we pass in the candidate function in the form
    # (learnable_tree, bs_action) or the true function
    if type(func) is tuple:
        learnable_tree = func[0]
        bs_action = func[1]
        u = lambda y: learnable_tree(y, bs_action)
    else:
        u = lambda y: func(y)

    u_x = u(x).squeeze().cuda()

    # compute grad_u, u_t:
    v = torch.ones_like(u_x).cuda()
    grads = torch.autograd.grad(u_x, x, grad_outputs=v, create_graph=True)[0]
    grad_u = grads[..., 1:]
    u_t = grads[..., 0]
    
    ############################################################################################################
    NUMERICAL INTEGRATION
    # INT (u(t, x+y) - u(t,x))k(y)dy

    int_pts = 200
    y = torch.linspace(-5,5,int_pts).cuda()
    x_shift = torch.cat((x[:, 0].unsqueeze(1).repeat(1, int_pts).unsqueeze(2),
                         (x[:, 1].unsqueeze(1).repeat(1, int_pts) + y.unsqueeze(0).repeat(num_samples, 1)).unsqueeze(2)), dim=2)
    k_y = k(y,lam_p, lam_n, nu)
    k_y_aug = k_y.unsqueeze(0).repeat(num_samples, 1)

    integral = 10/int_pts * torch.sum(u(x_shift)*k_y_aug, dim = -1)
    
    ############# going to try using Taylor Expectation instead of numerical integration: ########################
    ### INT (u(t, x+z) - u(t,x)) d nu  (=n2 in PIDES paper)
    x_shift = torch.empty_like(x).cuda()
    x_shift[:, :] = x[:, :]
    x_shift[..., 1:] += mu

    # 1st derivs
    v = torch.ones_like(u_x).cuda()
    u_x_shift = u(x_shift).squeeze()
    du_x_shift = torch.autograd.grad(u_x_shift, x_shift, grad_outputs=v, create_graph=True)[0]

    # 2nd derivs
    hes_diag1 = torch.empty((x.shape[0], x.shape[1] - 1)).cuda()
    v1 = torch.ones((x.shape[0], x.shape[1] - 1)).cuda()
    if grad_u.requires_grad:
        hes_diag1[..., :] = torch.autograd.grad(du_x_shift[..., 1:], x_shift, grad_outputs=v1, create_graph=True)[0][
                            ..., 1:]
    else:
        hes_diag1 = torch.zeros_like(du_x_shift).cuda()

    expect = u_x_shift + 1 / 2 * torch.sum(hes_diag1, dim=-1) * y_sigma ** 2

    loss1 = torch.mean((expect - u_x - grad_u * c - u_t - (r - q) * grad_u - r * u_x) ** 2)
    #loss2 (Initial Condition)

    x_init = torch.cat((torch.zeros(num_samples).unsqueeze(1).cuda(), x[:,1].unsqueeze(1)), dim = 1)

    loss2 = torch.mean((u(x_init) - np.max()

    #loss3 (Boundary Condition)

    min = np.log(1)
    max = np.log(10000)

    x_min = torch.cat((x[:,0].unsqueeze(1), min*torch.ones(num_samples).unsqueeze(1).cuda(), ), dim = 1)
    x_max = torch.cat((x[:,0].unsqueeze(1), max*torch.ones(num_samples).unsqueeze(1).cuda(), ), dim = 1)

    loss3 = torch.mean((u(x_min) - (K*torch.exp(-r*x[:,0]) - torch.exp(-q*x[:,0] + min)))**2) + torch.mean(u(x_max)**2)

    return loss1 + loss2 + loss3


def get_loss(func, FFT_sol):
    # We have two cases:  either we pass in the candidate function in the form
    # (learnable_tree, bs_action) or the true function

    x = torch.tensor(FFT_sol[:,:2]).cuda()
    sol = torch.tensor(FFT_sol[:,2]).cuda()
    if type(func) is tuple:
        learnable_tree = func[0]
        bs_action = func[1]
        u = lambda y: learnable_tree(y, bs_action)
    else:
        u = lambda y: func(y)

    u_x = u(x).squeeze()

    loss = torch.mean((u_x - sol)**2)
    #print(loss)


    return loss
'''

# This code implements (2) from the VGmodel paper - the PIDE for V(S,T) where S is the stock price, not
# the log-price.  Trying this implementation because it makes scaling and rescaling way easier.

def get_loss(func, lam_p, lam_n, nu, c, x):
    num_samples = x.shape[0]
    # parameters:
    r = .05
    q = .02
    K = 200
    #mu = 8/125
    #y_sigma = 112/3125 - (8/125)**2
    mu = -2/5
    y_sigma = (28/125) - (-2/5)**2
    c = -0.301115 # this is E(e^y - 1) given that y is distributed as k(y)

    # We have two cases:  either we pass in the candidate function in the form
    # (learnable_tree, bs_action) or the true function
    if type(func) is tuple:
        learnable_tree = func[0]
        bs_action = func[1]
        u = lambda y: learnable_tree(y, bs_action)
    else:
        u = lambda y: func(y)

    u_x = u(x).squeeze().cuda()


    # compute grad_u, u_t:
    v = torch.ones_like(u_x).cuda()
    grads = torch.autograd.grad(u_x, x, grad_outputs=v, create_graph=True)[0]
    grad_u = grads[..., 1:]
    u_t = grads[..., 0]


    '''
    # INT (u(t, x+y) - u(t,x))k(y)dy
    int_pts = 100
    y_end = 3
    y = torch.linspace(-y_end,y_end,int_pts).cuda()
    ey = torch.exp(y)
    x_ey = torch.cat((x[:, 0].unsqueeze(1).repeat(1, int_pts).unsqueeze(2),
                         (x[:, 1].unsqueeze(1).repeat(1, int_pts)*ey.unsqueeze(0).repeat(num_samples, 1)).unsqueeze(2)), dim=2)
    k_y = k(y,lam_p, lam_n, nu)
    k_y_aug = k_y.unsqueeze(0).repeat(num_samples, 1)

    integral = torch.trapezoid(u(x_ey).squeeze()*k_y_aug, y, dim=-1)

    loss1 = torch.mean((integral - c*grad_u*x[:,1] - u_x + u_t + (r - q)*grad_u*x[:,1] - r*u_x)**2)

    
    #going to try using Taylor Expectation instead of numerical integration:
    ### INT (u(t, x+z) - u(t,x)) d nu  (=n2 in PIDES paper)
    
    
    x_shift = torch.empty_like(x).cuda()
    x_shift[:, :] = x[:, :]
    x_shift[..., 1:] += mu

    # 1st derivs
    v = torch.ones_like(u_x).cuda()
    u_x_shift = u(x_shift).squeeze()
    du_x_shift = torch.autograd.grad(u_x_shift, x_shift, grad_outputs=v, create_graph=True)[0]

    # 2nd derivs
    hes_diag1 = torch.empty((x.shape[0], x.shape[1] - 1)).cuda()
    v1 = torch.ones((x.shape[0], x.shape[1] - 1)).cuda()
    if grad_u.requires_grad:
        hes_diag1[..., :] = torch.autograd.grad(du_x_shift[..., 1:], x_shift, grad_outputs=v1, create_graph=True)[0][
                            ..., 1:]
    else:
        hes_diag1 = torch.zeros_like(du_x_shift).cuda()

    expect = u_x_shift + 1 / 2 * torch.sum(hes_diag1, dim=-1) * y_sigma ** 2
    
    loss1 = torch.mean((expect - u_x - grad_u * c - u_t + (r - q) * grad_u - r * u_x) ** 2)
    '''
    x_ey = torch.empty_like(x).cuda()
    x_ey[:,:] = x[:,:]
    x_ey[..., 1:] *= np.exp(mu)

    # 1st derivs
    v = torch.ones_like(u_x).cuda()
    u_x_ey = u(x_ey).squeeze()
    du_x_ey = torch.autograd.grad(u_x_ey, x_ey, grad_outputs=v, create_graph=True)[0]

    # 2nd derivs
    hes_diag1 = torch.empty((x.shape[0], x.shape[1]-1)).cuda()
    v1 = torch.ones((x.shape[0], x.shape[1]-1)).cuda()
    if grad_u.requires_grad:
        hes_diag1[..., :] = torch.autograd.grad(du_x_ey[..., 1:], x_ey, grad_outputs=v1, create_graph=True)[0][
                            ..., 1:]
    else:
        hes_diag1 = torch.zeros_like(du_x_ey).cuda()
    expect = u_x_ey +  1 / 2 * torch.sum(hes_diag1, dim=-1) * y_sigma ** 2

    #loss1 = torch.mean((expect - u_x - grad_u*c - u_t - (r - q)*grad_u - r*u_x)**2)
    loss1 = torch.mean((expect - c * grad_u * x[:, 1] - u_x + u_t + (r - q) * grad_u * x[:, 1] - r * u_x) ** 2)



    #loss2 (Initial Condition)
    #This has also been rescaled and shifted to match the new range of x (now in [0,1] instead of [100, 400])

    x_init = torch.cat((torch.zeros(num_samples).unsqueeze(1).cuda(), x[:,1].unsqueeze(1)), dim = 1)
    #loss2 = torch.mean((u(x_init) - torch.clamp(1/3 - torch.exp(x_init[:,1]), min=0))**2)
    loss2 = torch.mean((u(x_init) - torch.clamp(1 / 3 - x_init[:, 1], min=0)) ** 2)

    #loss3 (Boundary Condition)
    #ikewise rescaled to x in [0,1]

    min = 0
    max = 1

    x_min = torch.cat((x[:,0].unsqueeze(1), min*torch.ones(num_samples).unsqueeze(1).cuda(), ), dim = 1)
    x_max = torch.cat((x[:,0].unsqueeze(1), max*torch.ones(num_samples).unsqueeze(1).cuda(), ), dim = 1)
    #loss3 = torch.mean((u(x_min) - ((1/3)*torch.exp(-r*x[:,0])) - torch.exp(min - q*x[:,0]))**2) + torch.mean(u(x_max)**2)
    loss3 = torch.mean((u(x_min) - ((1 / 3) * torch.exp(-r * x[:, 0]))) ** 2) + torch.mean(u(x_max) ** 2)

    return loss1 + 5*(loss2 + loss3)

# Code to get L1, L2 relatives errors, and MSE
def get_errors(learnable_tree, bs_action, bs_leaf_action, dim):
    leaf_modes = [v[0].item() for v in bs_leaf_action]
    for leaf in learnable_tree.linear:
        leaf.set_mode(leaf_modes[leaf.leaf_index])
    u = lambda y: learnable_tree(y, bs_action)
    mse_list = []
    denom = []
    relative_num = []
    relative_denom = []
    for _ in range(1000):
        x = get_pts(num_samples=int(10000/dim), dims=dim)
        mse_list.append(torch.mean((true_solution(x) - u(x).squeeze()) ** 2))
        relative_num.append(torch.mean(torch.abs(true_solution(x) - u(x).squeeze())))
        relative_denom.append(torch.mean(torch.abs(true_solution(x))))
        denom.append(torch.mean(true_solution(x) ** 2))
    relative_l2 = torch.sqrt(sum(mse_list)) / torch.sqrt(sum(denom))
    relative = sum(relative_num) / sum(relative_denom)
    mse = 1 / 1000 * sum(mse_list)
    return relative_l2, relative, mse


def true_solution(x):
    # parameters:
    mu = torch.Tensor([math.pi]).cuda()
    sin_mu_x = torch.sin(mu * x)
    return torch.prod(sin_mu_x, dim=1)

def relu(x):
    return torch.clamp(x, min = 0)


unary_functions = [lambda x: x + 0 * x ** 2,
                   lambda x: 0 * x ** 2,
                   lambda x: 1 + 0 * x ** 2,
                   lambda x: x ** 2,
                   lambda x: x ** 3,
                   lambda x: x ** 4,
                   torch.exp,
                   torch.sin,
                   torch.cos,
                   torch.sinh,
                   torch.cosh,
                   lambda x: relu(x)
                   #lambda x: torch.sin(3*x),
                   #lambda x: torch.sin(6*x),
                   #lambda x: torch.sin(9*x),
                   #lambda x: torch.sin(12*x),
                   #lambda x: torch.sin(15*x),
                   #lambda x: torch.sin(18*x),
                   #lambda x: torch.sin(21*x),
                   #lambda x: torch.sin(24*x),
                   #lambda x: torch.sinh(3*x),
                   #lambda x: torch.sinh(6*x),
                   #lambda x: torch.sinh(9*x),

                   ]


binary_functions = [lambda x, y: x + y,
                    lambda x, y: x * y,
                    lambda x, y: x - y]
'''
unary_functions_str = ['({}*{}+{})',
                       '({}*(0)+{})',
                       '({}*(1)+{})',
                       # '5',
                       # '-{}',
                       '({}*({})**2+{})',
                       '({}*({})**3+{})',
                       '({}*({})**4+{})',
                       # '({})**5',
                       '({}*exp({})+{})',
                       '({}*sin({})+{})',
                       '({}*cos({})+{})',
                       '({}*sinh({})+{}',
                       '({}*cosh({})+{}',
                       '({}*sin(3*({}))+{})',
                       '({}*sin(6*({}))+{})',
                       '({}*sin(9*({}))+{})',
                       '({}*sin(12*({}))+{})',
                       '({}*sin(15*({}))+{})',
                       '({}*sin(18*({}))+{})',
                       '({}*sin(21*({}))+{})',
                       '({}*sin(24*({}))+{})',
                       '({}*sinh(3*({}))+{})',
                       '({}*sinh(6*({}))+{})',
                       '({}*sinh(9*({}))+{})',
                       ]
# 'ref({})',
# 'exp(-({})**2/2)']

unary_functions_str_leaf = ['({}*{}+{})',
                            '({}*(0)+{}',
                            '({}*(1)+{})',
                            # '5',
                            # '-{}',
                            '(({}*({})+{})**2)',
                            '(({}*({})+{})**3)',
                            '(({}*({})+{})**4)',
                            # '({})**5',
                            '(exp({}*({})+{}))',
                            '(sin({}*({})+{}))',
                            '(cos({}*({})+{}))',
                            '(sinh({}*({})+{}))',
                            '(cosh({}*({})+{}))',
                            '(sin(2*({}*({})+{})))',
                            '(sin(3*({}*({})+{})))',
                            '(sin(4*({}*({})+{})))',
                            '(sin(5*({}*({})+{})))',
                            '(sinh(2*({}*({})+{})))',
                            '(sinh(3*({}*({})+{})))',
                            '(sinh(4*({}*({})+{})))',
                            '(sinh(5*({}*({})+{})))',
                            ]
                            

unary_functions_str_leaf = ['({}*{})',
                            '({}*(0))',
                            '({}*(1))',
                            # '5',
                            # '-{}',
                            '(({}*({}))**2)',
                            '(({}*({}))**3)',
                            '(({}*({}))**4)',
                            # '({})**5',
                            '(exp({}*({})))',
                            '(sin({}*({})))',
                            '(cos({}*({})))',
                            '(sinh({}*({})))',
                            '(cosh({}*({})))',
                            '(sin(3*({}*({}))))',
                            '(sin(6*({}*({}))))',
                            '(sin(9*({}*({}))))',
                            '(sin(12*({}*({}))))',
                            '(sin(15*({}*({}))))',
                            '(sin(18*({}*({}))))',
                            '(sin(21*({}*({}))))',
                            '(sin(24*({}*({}))))',
                            '(sinh(3*({}*({}))))',
                            '(sinh(6*({}*({}))))',
                            '(sinh(9*({}*({}))))',
                            ]
'''
unary_functions_str = ['({}*{}+{})',
                       '({}*(0)+{})',
                       '({}*(1)+{})',
                       # '5',
                       # '-{}',
                       '({}*({})**2+{})',
                       '({}*({})**3+{})',
                       '({}*({})**4+{})',
                       # '({})**5',
                       '({}*exp({})+{})',
                       '({}*sin({})+{})',
                       '({}*cos({})+{})',
                       '({}*sinh({})+{})',
                       '({}*cosh({})+{})',
                       '({}*relu({})+{})',

                       #'({}*sin(3*({}))+{})',
                       #'({}*sin(6*({}))+{})',
                       #'({}*sin(9*({}))+{})',
                       #'({}*sin(12*({}))+{})',
                       #'({}*sin(15*({}))+{})',
                       #'({}*sin(18*({}))+{})',
                       #'({}*sin(21*({}))+{})',
                       #'({}*sin(24*({}))+{})',
                       #'({}*sinh(3*({}))+{})',
                       #'({}*sinh(6*({}))+{})',
                       #'({}*sinh(9*({}))+{})',
                       ]

unary_functions_str_leaf = ['({:.8f}*{})',
                            '({:.8f}*(0))',
                            '({:.8f}*(1))',
                            # '5',
                            # '-{}',
                            '(({:.8f}*({}))**2)',
                            '(({:.8f}*({}))**3)',
                            '(({:.8f}*({}))**4)',
                            # '({})**5',
                            '(exp({:.8f}*({})))',
                            '(sin({:.8f}*({})))',
                            '(cos({:.8f}*({})))',
                            '(sinh({:.8f}*({})))',
                            '(cosh({:.8f}*({})))',
                            '(relu({:.8f}*({})))',
                            #'(sin(3*({:.6f}*({}))))',
                            #'(sin(6*({:.6f}*({}))))',
                            #'(sin(9*({:.6f}*({}))))',
                            #'(sin(12*({:.6f}*({}))))',
                            #'(sin(15*({:.6f}*({}))))',
                            #'(sin(18*({:.6f}*({}))))',
                            #'(sin(21*({:.6f}*({}))))',
                            #'(sin(24*({:.6f}*({}))))',
                            #'(sinh(3*({:.6f}*({}))))',
                            #'(sinh(6*({:.6f}*({}))))',
                            #'(sinh(9*({:.6f}*({}))))',
                            ]

binary_functions_str = ['(({})+({}))',
                        '(({})*({}))',
                        '(({})-({}))']
'''
unary_functions = [lambda x: x + 0 * x ** 2,
                   torch.sin,
                   torch.sinh,
                   ]


binary_functions = [lambda x, y: x + y,
                    lambda x, y: x * y,
                    lambda x, y: x - y]

unary_functions_str = ['({}*{}+{})',
                       '({}*sin({})+{})',
                       '({}*sinh({})+{}',]
# 'ref({})',
# 'exp(-({})**2/2)']

unary_functions_str_leaf = ['({})',
                            '(sin({}))',
                            '(sinh({}))',]
'''

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
