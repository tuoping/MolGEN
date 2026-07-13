# https://github.com/willisma/SiT/
import copy

import torch as th
import numpy as np

import enum

from . import path


def mean_flat(x, mask):
    """
    Take the mean over all non-batch dimensions.
    """
    mask = mask.expand(x.shape)
    return th.sum(x * mask, dim=list(range(1, len(x.size())))) / th.sum(mask, dim=list(range(1, len(x.size()))))


from .integrators import ode, sde


class ModelType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    NOISE = enum.auto()  # the model predicts epsilon
    SCORE = enum.auto()  # the model predicts \nabla \log p(x)
    VELOCITY = enum.auto()  # the model predicts v(x)


class PathType(enum.Enum):
    """
    Which type of path to use.
    """

    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()
    Pow = enum.auto()


class WeightType(enum.Enum):
    """
    Which type of weighting to use.
    """

    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


def t_to_alpha(t, args):
    """
    Convert t to alpha for Dirichlet distribution.
    """

    return 1 * (1 - t) + t * args.alpha_max, (args.alpha_max - 1)


def divergence(v_func, x, t, model_kwarg):
    # v_func: function that outputs v(x,t)
    x.requires_grad_(True)
    v = v_func(x, t, **model_kwarg)
    div = 0.0
    for i in range(x.shape[-1]):  # iterate over dimensions
        div += th.autograd.grad(v[..., i].sum(), x, create_graph=True)[0][..., i]
    return div 


from torch_linear_assignment import batch_linear_assignment
def hungarian_over_L(cost_matrix: th.Tensor, **kwargs) -> th.Tensor:
    return batch_linear_assignment(cost_matrix)   # stays on GPU, CUDA kernel

import torch.distributions as D

def sample_t_u_shaped(n, alpha=0.8, reweight=True, eps=1e-6):
    # U-shaped proposal
    dist = D.Beta(alpha, alpha)
    t = dist.sample((n,)).clamp(eps, 1-eps)

    if reweight:
        # target p(t) = Uniform[0,1] => p(t)=1 on [0,1]
        # importance weight w(t) = p(t)/q(t) = 1 / q(t)
        q = th.exp(dist.log_prob(t))
        w = (1.0 / q)
        w = w / w.mean()          # stabilize; keeps expected weight = 1
    else:
        w = th.ones_like(t)

    return t, w


def alpha_divergence(log_p, log_q, alpha, eps=1e-6):
    # p, q: probs (batch, K). alpha != 0,1
    assert alpha >=0 and alpha <= 1, alpha
    p = log_p.exp()
    q = log_q.exp()
    assert th.all(th.isfinite(p))
    assert th.all(th.isfinite(q))
    if abs(alpha-1.0) < 1e-6:
        return (p * (log_p - log_q)).sum(dim=-1)  # forward KL
    if abs(alpha) < 1e-6:
        return (q * (log_q - log_p)).sum(dim=-1)  # reverse KL
    # s = (p.pow(alpha) * q.pow(1.0 - alpha)).sum(dim=-1)
    s = (alpha*log_p + (1-alpha)*log_q ).exp().sum(dim=-1)
    assert th.all(th.isfinite(s)), "  ".join([str((alpha*log_p + (1-alpha)*log_q ).max()), str(log_p.max()), str(log_q.max()), str(alpha)])
    return (1.0 / (alpha * (alpha - 1.0))) * (1.0 - s)

def geodesic_distance(x_string):
    B,T,L,_ = x_string.shape
    d = th.zeros_like(x_string)
    for i in range(1, T):
        d += (x_string[0,i]-x_string[0, i-1])**2
    return d/2


def grad_log_normal_iso_3d(x, mu=0, sigma=1):
    """
    s_t = ∇_x log N(x | mu, σ² I) = -(x - mu) / (sigma**2)
    same shape as x
    """
    # return -(x - mu) / (sigma**2)
    B,T,N,_ = x.shape
    def f_dx_k(dx, k):
        if isinstance(sigma, th.Tensor):
            return (-dx+k) / (sigma[:,None,None,None]**2)
        else:
            return (-dx+k) / (sigma**2)
    return path.compute_weighted((x-mu).view(B*T,N,3), 1/sigma, f_dx_k)


def latin_hypercube_torch(
    B: int,
    N: int,
    d: int,
    device,
    dtype=th.float,
):
    """
    Centered Latin hypercube sampling in [0, 1]^d.
    No jitter: each point lies exactly at the center of a bin.

    Returns:
        X: tensor of shape (N, d)
    """
    # random keys: one independent permutation for each batch and dimension
    keys = th.rand(
        B, d, N,
        device=device,
    )

    # argsort gives permutations of 0, ..., N-1
    perms = th.argsort(keys, dim=-1)  # shape: (batch_size, d, N)

    # transpose to point-major layout: (batch_size, N, d)
    X = (perms.transpose(1, 2).to(dtype) + 0.5) / N

    return X

@th.no_grad()
def lattice_polar_build_torch(k):
    assert k.dim() == 2, "input must be batched k of shape (B,6)"
    S0 = th.stack([k[:, 3] + k[:, 4] + k[:, 5], k[:, 0], k[:, 1]], dim=1)  # (B, 3)
    S1 = th.stack([k[:, 0], -k[:, 3] + k[:, 4] + k[:, 5], k[:, 2]], dim=1)  # (B, 3)
    S2 = th.stack([k[:, 1], k[:, 2], -2 * k[:, 4] + k[:, 5]], dim=1)  # (B, 3)
    S = th.stack([S0, S1, S2], dim=1)  # (B, 3, 3)
    expS = th.matrix_exp(S)  # (B, 3, 3)
    return expS


def decompose_symmetric_matrix(S: th.Tensor):
    k0 = S[:, 0, 1]
    k1 = S[:, 0, 2]
    k2 = S[:, 1, 2]
    k3 = (S[:, 0, 0] - S[:, 1, 1]) / 2
    k4 = (S[:, 0, 0] + S[:, 1, 1] - 2 * S[:, 2, 2]) / 6
    k5 = (S[:, 0, 0] + S[:, 1, 1] + S[:, 2, 2]) / 3
    k = th.vstack([k0, k1, k2, k3, k4, k5]).transpose(-1, -2)
    return k

@th.no_grad()
def lattice_polar_decompose_torch(lattices: th.Tensor):
    assert lattices.dim() == 3, "input must be batched lattices of shape (B,3,3)"
    A, U = th.linalg.eigh(lattices @ lattices.transpose(-1,-2))  # J = L^T @ L
    # S = 1/2 U log(A) U^T
    A = th.diag_embed(A.log()) / 2
    S = U @ A @ U.transpose(-1, -2)
    k = decompose_symmetric_matrix(S)
    return k


from .path import wrap_frac_pos
import math

def compute_jsd_loss(mu_t_x1, standard_bandwidth_factor, mu_theta, k_max=3):
    """
    Monte Carlo estimate of the full JSD:
    JSD(p || q) = 0.5 * E_p[log(p/m)] + 0.5 * E_q[log(q/m)]

    Args:
        mu_t_x1: (B, N, 3) - means for the wrapped Gaussian
        standard_bandwidth_factor: scalar or (B,) - 1/\sigma for the wrapped Gaussion
        mu_theta: (B, N, 3) - predicted mean
        t: scalar or (B,) - time
        k_max: int - number of periodic images per dimension
    """
    x = 0.0
    if isinstance(standard_bandwidth_factor, th.Tensor):
        bandwidth_factor = standard_bandwidth_factor[:, None, None] ** 2  # scalar or (B,)
    else:
        bandwidth_factor = standard_bandwidth_factor ** 2

    ks = th.arange(-k_max, k_max + 1, device=mu_t_x1.device)
    kx, ky, kz = th.meshgrid(ks, ks, ks, indexing='ij')
    k_vecs = th.stack([kx.ravel(), ky.ravel(), kz.ravel()], dim=-1).float()  # (K, 3)

    # Broadcast: mu_t_x1 is (B, N, 3), k_vecs is (K, 3)
    # diff: (B, N, K, 3)
    diff = (x - mu_t_x1[:, :, None, :] + k_vecs[None, None, :, :])   # @cell[:, None, :, :]
    sq_norms = th.sum(diff ** 2, dim=-1)  # (B, N, K)
    logZ_P = th.logsumexp(-sq_norms / (2) * bandwidth_factor, dim=-1) # (B, N)
    logP_ = (-sq_norms / (2) * bandwidth_factor) # - logZ_P[:,:,None]  # (B, N, K)

    pred_diff = (x - mu_theta[:, :, None, :] + k_vecs[None, None, :, :])   # @cell  # (B, None, N, 3)
    pred_sq_norms = th.sum(pred_diff ** 2, dim=-1) # (B, N, K)
    logZ_Q = th.logsumexp(-pred_sq_norms / (2) * bandwidth_factor, dim=-1) # (B, N)
    logQ_ = (-pred_sq_norms) / (2) * bandwidth_factor # - logZ_Q[:,:,None]  # (B, N, K)

    logm = th.logaddexp(logP_, logQ_) - th.log(th.tensor(2.0, device=mu_t_x1.device))# (B, N, K)

    kl_p_m = (th.exp(logP_-logZ_P[:,:,None]) * (logP_ - logQ_)).sum(dim=-1)  # (B,N)
    kl_q_m = (th.exp(logQ_-logZ_Q[:,:,None]) * (logQ_ - logm)).sum(dim=-1)  # (B,N)

    jsd = 0.5 * kl_p_m + 0.5 * kl_q_m  # (B,N)
    return jsd

class Transport:

    def __init__(
            self,
            *,
            args,
            model_type,
            path_type,
            loss_type,
            train_eps,
            sample_eps,
            score_model = None,
            latt_path = False,
            weightfunction_x = None
    ):
        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
        }
        self.args = args
        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps
        self.score_model = score_model
        self.latt_path = latt_path
        self.prior_mean = None
        self.prior_cell = None
        self.weightfunction_x = weightfunction_x

    def prior_logp(self, z):
        '''
            Standard multivariate normal prior
            Assume z is batched
        '''
        shape = th.tensor(z.size())
        N = th.prod(shape[1:])
        _fn = lambda x: -N / 2. * np.log(2 * np.pi) - th.sum(x ** 2) / 2.
        return th.vmap(_fn)(z)

    def check_interval(
            self,
            train_eps,
            sample_eps,
            *,
            diffusion_form="SBDM",
            sde=False,
            reverse=False,
            eval=False,
            last_step_size=0.0,
    ):
        t0 = 0
        t1 = 1
        eps = train_eps if not eval else sample_eps
        if (type(self.path_sampler) in [path.VPCPlan]):

            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]) and (self.model_type != ModelType.VELOCITY or sde):  # avoid numerical issue by taking a first
            # semi-implicit step

            t0 = eps if (sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1

    def sample(self, shape, device, x0std):
        """Sampling x0 & t based on shape of x1 (if needed)
          Args:
            x1 - data point; [batch, *dim]
        """
        B,T,N,C = shape
        x0 = []
        x0_mean = []
        for i in range(1):
            if self.prior_mean is None:
                ### Even sample
                # m = math.ceil(N ** (1/3))
                # g = (th.arange(m, device=device) + 0.5) / m
                # X, Y, Z = th.meshgrid(g, g, g, indexing='ij')
                # _x0_mean = th.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1)[:N].unsqueeze(0).expand(T, -1, -1).unsqueeze(0).expand(B, -1, -1, -1)
                ### Latin hypercube sample
                _x0_mean = latin_hypercube_torch(B*T, N, C, device).view(B,T,N,C)
                ### Uniform sample
                # _x0_mean = th.rand(shape, device=device)
            else:
                _x0_mean = self.prior_mean
            inv_cell = th.linalg.inv(self.prior_cell)
            x0.append((th.randn(shape, device=device)*x0std[:,:,None,None])@inv_cell + _x0_mean)
            x0_mean.append(_x0_mean)
        
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        # t = th.rand((x1.shape[0],))
        t, _ = sample_t_u_shaped(shape[0], self.args.beta_sample_t, eps=0)
        t = t*(t1-t0) + t0
        t = t.to(device)
        return t, x0, x0_mean


    def sample_latt(self, shape, device):
        """Sampling x0 & t based on shape of x1 (if needed)
          Args:
            x1 - data point; [batch, *dim]
        """
        B = shape[0]
        T = shape[1]
        if self.prior_cell is None:
            num_atoms = shape[2]
            mu = th.zeros(*shape[:2], 6)
            mu[:,:,-1] = 1.
            sigma = 0.1
            k = th.normal(mu, sigma).to(device)
            cell = lattice_polar_build_torch(k.reshape(-1, 6)).reshape(-1, 3, 3)
            volume = (cell[:,0] * th.cross(cell[:,1], cell[:,2], dim=1)).sum(dim=-1)
            target_volume = num_atoms * self.mean_atomic_volume * th.ones_like(volume)
            # residual_k = (th.log(target_volume) - th.log(volume))/3
            # return k.reshape(-1, 6) + residual_k[:,None]
            _cell = cell * ((target_volume/volume)**(1./3.))[:,None,None]
            return _cell.view(*shape[:2], 3, 3)
        else:
            return self.prior_cell.clone()

    def training_losses(
            self,
            model,
            x1,           # target tokens
            aatype1=None, # target aatype
            mask=None,
            model_kwargs=None,
            forces = None,
            E = None,
            x0std = None,
            global_step = None,
    ):
        """Loss for training the score model
        Args:
        - model: backbone model; could be score, noise, or velocity
        - x1: datapoint
        - model_kwargs: additional arguments for the model
        """
        #if global_step < 20:
        #    self.pref_symmkl = 0.1*global_step
        #    self.pref_alpha_div = 0.9 - 0.6*global_step/20
        #else:
        self.pref_symmkl = 1.
        self.pref_alpha_div = 0.3
        self.pref_reversekl = 0.3
        assert self.pref_alpha_div >=0 and self.pref_alpha_div <= 1, "  ".join([str(self.pref_alpha_div), str(global_step)])


        if model_kwargs == None:
            model_kwargs = {}
        B, T, N, C = x1.shape
        ### normal sampler of t
        t, x0, x0_mean = self.sample(x1.shape, x1.device, x0std)
        ### OT in the atom number dimension
        if self.prior_mean is None:
            x1 = x1.view(B*T, N, C)
            model_kwargs['x1'] = model_kwargs['x1'].view(B*T, N, C)
            assignment = hungarian_over_L(th.cdist(x0[0].view(B*T, N, C), x1))
            x1 = x1[th.arange(B).unsqueeze(1).expand(B, N), assignment]
            model_kwargs['x1'] = model_kwargs['x1'][th.arange(B).unsqueeze(1).expand(B, N), assignment]
            x1 = x1.view(B, T, N, C)
            model_kwargs['x1'] = model_kwargs['x1'].view(B, T, N, C)
        
        ### OT in the batch dimension
        # # Flatten each sample's atom features into a single vector: (B*T, N*C)
        # x0_flat = x0[0].view(B*T, N, C).reshape(B*T, N*C)
        # x1_flat = x1.reshape(B*T, N*C)
        # # Cost matrix is (1, B*T, B*T) — pairwise distances between batch elements
        # cost_matrix = th.cdist(x0_flat.unsqueeze(0), x1_flat.unsqueeze(0))  # (1, B*T, B*T)
        # # Hungarian assignment over the batch dimension
        # assignment = hungarian_over_L(cost_matrix)  # (1, B*T)
        # assignment = assignment.squeeze(0)          # (B*T,)
        # # Permute x1 along the batch dimension
        # x1 = x1[assignment]  # (B*T, N, C)
        # model_kwargs['x1'] = model_kwargs['x1'][assignment]
        # x1 = x1.view(B, T, N, C)
        # model_kwargs['x1'] = model_kwargs['x1'].view(B, T, N, C)

        if self.args.design:  # alterations made to the original SIT code to include dirichlet flow matching for design
            assert self.model_type == ModelType.VELOCITY
            seq_one_hot = aatype1
            alphas, _ = t_to_alpha(t, self.args)
            alphas = th.ones_like(seq_one_hot) + seq_one_hot * (alphas[:, None, None, None] - th.ones_like(seq_one_hot))
            x_d = th.distributions.Dirichlet(alphas).sample()
            xt = x_d
        else:
            if self.args.path_type not in ["Schrodinger_Linear", "Schrodinger_Linear_onemodel"]:
                xt, ut = self.path_sampler.plan_fractional(t, x0[0], x1)
                alpha_t, _ = self.path_sampler.compute_alpha_t(path.expand_t_like_x(t, xt))
                if self.latt_path:
                    latt0 = self.sample_latt(x1.shape, x1.device)
                    B,T,_,_ = model_kwargs['cell'].shape
                    # latt1 = lattice_polar_decompose_torch(model_kwargs['cell'].reshape([B*T,3,3])).reshape(B*T,6)
                    latt1 = model_kwargs['cell']
                    latt, ulatt = self.path_sampler.plan_latt_riemann(t, latt0, latt1)
                    model_kwargs['cell'] = latt
                    # model_kwargs['cell'] = lattice_polar_build_torch(latt.reshape([B*T,6])).reshape([B,T,3,3])
                    # ulatt_L = lattice_polar_build_torch(ulatt.reshape([B*T,6])).reshape([B,T,3,3])
                assert self.args.weight_loss_var_x0 == 0
            else:
                assert self.args.weight_loss_var_x0 == 0
                diffusion = self.path_sampler.compute_diffusion(x1, t, self.args.diffusion_form, self.args.diffusion_norm)  # the input x here is not used
                xt, ut, eps = self.path_sampler.plan_schrodinger_bridge_fractional(t, x0[0], x1, diffusion, cell=self.prior_cell)
                alpha_t, _ = self.path_sampler.compute_alpha_t(path.expand_t_like_x(t, xt))
                sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))


        
        assert t.shape == (B,)
        if self.latt_path:
            model_output, lattflow_output = model(xt, t, **model_kwargs)
        else:
            model_output = model(xt, t, **model_kwargs)
        assert self.args.weight_loss_var_x0 == 0
        if self.args.path_type in ["Schrodinger_Linear", "Schrodinger_Linear_onemodel"]:
            if self.latt_path:
                raise NotImplementedError("Score model with lattice path is not implemented.")
            if self.score_model is not None:
                score_model_output = self.score_model(xt, t, **model_kwargs)
            else:
                score_model_output = self.path_sampler.get_score_from_velocity(model_output, xt-x0_mean[0], t, self.x0std)

        assert model_output.size() == (B, *xt.size()[1:-1], C)

        if self.args.design:
            logits = model_output[:, :, :, -self.args.num_species:]
            model_output = model_output[:, :, :, :-self.args.num_species]

        terms = {}
        terms['t'] = t
        terms['pred'] = model_output
        terms['x0'] = x0
        if not (self.args.design):
            if self.model_type == ModelType.VELOCITY:
                # if self.args.KL == 'symm':
                match self.args.KL:
                    case "symm":
                        cell = model_kwargs['cell'].view(B*T,3,3)
                        terms['loss_l1'] = mean_flat((model_output.view(B*T,N,3)@cell - ut.view(B*T,N,3)@cell).norm(dim=-1), mask.view(B*T,N,3)[:,:,0])
                        volume = th.abs(th.det(cell))
                        
                        jsd = compute_jsd_loss(xt.view(B*T,N,3), t, (x0[0]+model_output*t[:,None,None,None]).view(B*T,N,3), 3)  # (B,N)
                        terms['loss_symmkl'] = mean_flat(jsd, mask.view(B*T,N,3)[:,:,0])
                        terms['loss_flow'] = terms['loss_symmkl'] + terms['loss_l1'] 
                    case "L1":
                        cell = model_kwargs['cell'].view(B*T,3,3)
                        terms['loss_flow'] = mean_flat((model_output.view(B*T,N,3)@cell - ut.view(B*T,N,3)@cell).norm(dim=-1), mask.view(B*T,N,3)[:,:,0])
                    case "score":
                        cell = model_kwargs['cell'].view(B*T,3,3)
                        terms['loss_l1'] = mean_flat((model_output.view(B*T,N,3)@cell - ut.view(B*T,N,3)@cell).norm(dim=-1), mask.view(B*T,N,3)[:,:,0])
                        score_ot = self.path_sampler.get_score_from_velocity(model_output, xt-x0_mean[0], t, self.x0std)
                        if self.weightfunction_x is not None:
                            e_repul, f_repul = self.weightfunction_x(xt, cell, model_kwargs['num_atoms'],)
                            forces_repulsed = forces + f_repul
                            terms['loss_score'] = mean_flat( (score_ot@cell - 1./alpha_t * forces_repulsed).norm(dim=-1) * (t**4)[:,None,None], mask[:,:,:,0]) * 0.001
                        else:
                            terms['loss_score'] = mean_flat( (score_ot@cell - 1./alpha_t * forces).norm(dim=-1) * (t**4)[:,None,None], mask[:,:,:,0]) * 0.001
                        terms['loss_flow'] = terms['loss_score'] + terms['loss_l1'] 
                    case _:
                        raise Exception(f"Wrong KL argument: {self.args.KL}")
                    
                if self.args.path_type in ["Schrodinger_Linear", "Schrodinger_Linear_onemodel"]:
                    cell = model_kwargs['cell']
                    gamma_cart = th.sqrt(2 * diffusion * (t * (1 - t))[:,None,None,None])
                    terms['loss_dsm'] = mean_flat((((score_model_output @ cell) * gamma_cart + eps)**2), mask)

                    # endpoint 0 / Gaussian prior label
                    dx0_cart = (x0[0] - x0_mean[0]) @ cell
                    target_score_cart_0 = -dx0_cart / (
                        sigma_t * x0std[:, :, None, None]**2 + 1e-12
                    )
                    terms["loss_tsm_0"] = mean_flat(
                        ((score_model_output@cell - target_score_cart_0) ** 2 * (sigma_t * x0std[:, :, None, None]) ** 2)
                        * (t < 0.5).to(x1.dtype)[:, None, None, None],
                        mask,
                    )
                    # endpoint 1 / forces label
                    terms['loss_tsm_1'] = mean_flat( ((score_model_output@cell - 1./alpha_t*forces)**2 * (x0std[:, :, None, None]) ** 2 * (t > 0.5).to(th.int)[:,None,None,None]), mask)
                    if self.args.TSMloss:
                        terms['loss'] = terms['loss_flow'] + terms['loss_dsm'] + (terms["loss_tsm_1"] + terms["loss_tsm_0"]) * self.args.pref_TSMloss
                    else:
                        terms['loss'] = terms['loss_flow'] + terms['loss_dsm']
                else:
                    terms['loss'] = terms['loss_flow']
                    if self.latt_path:
                        lowertrigflow_output = th.stack([lattflow_output[:,:,0,0], lattflow_output[:,:,1,0], lattflow_output[:,:,1,1], lattflow_output[:,:,2,0], lattflow_output[:,:,2,1], lattflow_output[:,:,2,2]], dim=-1)
                        lowertrigulatt = th.stack([ulatt[:,:,0,0], ulatt[:,:,1,0], ulatt[:,:,1,1], ulatt[:,:,2,0], ulatt[:,:,2,1], ulatt[:,:,2,2] ], dim=-1)
                        terms['loss_lattflow'] = mean_flat((lowertrigflow_output - lowertrigulatt).abs(), th.ones_like(lowertrigflow_output, device=lowertrigflow_output.device))
                        terms['loss'] = terms['loss_flow'] + terms['loss_lattflow']
                        # terms['loss'] = terms['loss_lattflow']

                if self.args.loss_consistency:
                    if th.randn(1).item() > 1 and th.ceil(2/(1-t.min())).to(int).item() < 128: # True roughly 1 out of 6 times
                        d = 1./th.randint(low=th.ceil(2/(1-t.min())).to(int).item(), high=128, size=(B,), device=xt.device)
                        if self.latt_path:
                            model_kwargs['dt'] = 2*d.view(B,1,1).expand(-1,T,-1)
                            _model_output_0, _lattflow_output_0 = model(xt, t, **model_kwargs)
                            model_kwargs['dt'] = d.view(B,1,1).expand(-1,T,-1)
                            _model_output_1, _lattflow_output_1 = model(xt, t, **model_kwargs)
                            model_kwargs['dt'] = d.view(B,1,1).expand(-1,T,-1)
                            _model_output_2, _lattflow_output_2 = model(xt, t + d, **model_kwargs)
                            terms['loss_consistency'] = ((_model_output_0 - (_model_output_1 + _model_output_2)/2.)**2).view(B,-1).mean(dim=-1) \
                                + ((_lattflow_output_0 - (_lattflow_output_1 + _lattflow_output_2)/2.)**2).view(B,-1).mean(dim=-1)
                        else:
                            model_kwargs['dt'] = 2*d.view(B,1,1).expand(-1,T,-1)
                            _model_output_0 = model(xt, t, **model_kwargs)
                            model_kwargs['dt'] = d.view(B,1,1).expand(-1,T,-1)
                            _model_output_1 = model(xt, t, **model_kwargs)
                            model_kwargs['dt'] = d.view(B,1,1).expand(-1,T,-1)
                            _model_output_2 = model(xt, t + d, **model_kwargs)
                            terms['loss_consistency'] = ((_model_output_0 - (_model_output_1 + _model_output_2)/2.)**2).view(B,-1).mean(dim=-1)
                        terms['loss'] += terms['loss_consistency']
                    else:
                        terms['loss_consistency'] = th.zeros(B, device = xt.device)

            else:
                _, drift_var = self.path_sampler.compute_drift(xt, t)
                sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
                if self.loss_type in [WeightType.VELOCITY]:
                    weight = (drift_var / sigma_t) ** 2
                elif self.loss_type in [WeightType.LIKELIHOOD]:
                    weight = drift_var / (sigma_t ** 2)
                elif self.loss_type in [WeightType.NONE]:
                    weight = 1
                else:
                    raise NotImplementedError()

                if self.model_type == ModelType.NOISE:
                    terms['loss'] = mean_flat(weight * ((model_output - x0[0]) ** 2), mask)
                else:
                    # terms["loss_continuous"]=(weight * ((model_output * sigma_t + x0[0]) ** 2)*mask)
                    terms['loss'] = mean_flat(weight * ((model_output * sigma_t + x0[0]) ** 2), mask) # loss by comparing the x_0

        # more changes for dirichlet flow matching

        if self.args.design:
            # terms['loss_continuous'] = th.tensor(th.nan, device=xt.device)
            loss_d = th.nn.functional.cross_entropy(logits.reshape(-1,self.args.num_species), aatype1.reshape(-1,self.args.num_species).argmax(dim=-1), reduction="none").reshape(x1.shape[:-1])
            terms['loss'] = mean_flat(loss_d, mask)
            terms['loss_discrete'] = loss_d
            terms['logits'] = logits

        return terms


    def get_drift(
            self
    ):
        """member function for obtaining the drift of the probability flow ODE"""

        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return (-drift_mean + drift_var * model_output)  # by change of variable

        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return (-drift_mean + drift_var * score)

        def velocity_ode(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            return model_output

        if self.model_type == ModelType.NOISE:
            if self.latt_path:
                raise Exception("ModelType.NOISE is not implemented for variable lattice")
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            if self.latt_path:
                raise Exception("ModelType.SCORE is not implemented for variable lattice")
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode

        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            # assert model_output.shape == x.shape, "Output shape from ODE solver must match input shape"
            assert model_output[0].shape == x.shape if isinstance(model_output, tuple) else model_output.shape == x.shape, "Output shape from ODE solver must match input shape"
            assert model_output.dim() == 4, "Output from ODE solver must be a 4D tensor"
            return model_output

        return body_fn

    def get_score(
            self,
    ):
        """member function for obtaining score of 
            x_t = alpha_t * x + sigma_t * eps"""
        
        def score_sde(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            return model_output
        
        if self.model_type == ModelType.NOISE:
            score_fn = lambda x, t, model, **model_kwargs: model(x, t, **model_kwargs) / - \
                self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))[0]
        elif self.model_type == ModelType.SCORE:
            score_fn = lambda x, t, model, **model_kwargs: model(x, t, **model_kwargs)
        elif self.model_type == ModelType.VELOCITY:
            if self.score_model is None:
                score_fn = lambda x, t, model, **model_kwargs: self.path_sampler.get_score_from_velocity(model(x, t, **model_kwargs), x-self.prior_mean, t, self.x0std)
            else:
                score_fn = score_sde
        else:
            raise NotImplementedError()

        return score_fn


class Sampler:
    """Sampler class for the transport model"""

    def __init__(
            self,
            transport,
    ):
        """Constructor for a general sampler; supporting different sampling methods
        Args:
        - transport: an tranport object specify model prediction & interpolant type
        """

        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()

    def __get_sde_diffusion_and_drift(
            self,
            *,
            diffusion_form="SBDM",
            diffusion_norm=1.0,
            reverse=False
    ):

        def diffusion_fn(x, t):
            diffusion = self.transport.path_sampler.compute_diffusion(x, t, form=diffusion_form, norm=diffusion_norm)
            return diffusion

        inv_cell = th.linalg.inv(self.transport.prior_cell)
        sde_drift = \
                lambda x, t, model, score_model, **kwargs: \
                    self.drift(x, t, model, **kwargs) + diffusion_fn(x, t) * self.score(x, t, score_model, **kwargs)

        sde_diffusion = diffusion_fn
        return sde_drift, sde_diffusion
    

    def __get_sde_reverse_drift(
            self,
            *,
            diffusion_form="SBDM",
            diffusion_norm=1.0,
            reverse=False
    ):

        def diffusion_fn(x, t):
            diffusion = self.transport.path_sampler.compute_diffusion(x, t, form=diffusion_form, norm=diffusion_norm)
            return diffusion

        inv_cell = th.linalg.inv(self.transport.prior_cell)
        sde_drift = \
                lambda x, t, model, score_model, **kwargs: \
                    -self.drift(x, t, model, **kwargs) + diffusion_fn(x, t) * self.score(x, t, score_model, **kwargs)

        return sde_drift

    def __get_last_step(
            self,
            sde_drift,
            *,
            last_step,
            last_step_size,
    ):
        """Get the last step function of the SDE solver"""

        if last_step is None:
            last_step_fn = \
                lambda x, t, model, score_model, **model_kwargs: \
                    x
        elif last_step == "Mean":
            last_step_fn = \
                lambda x, t, model, score_model, **model_kwargs: \
                    x + sde_drift(x, t, model, score_model, **model_kwargs) * last_step_size
        elif last_step == "Tweedie":
            alpha = self.transport.path_sampler.compute_alpha_t  # simple aliasing; the original name was too long
            sigma = self.transport.path_sampler.compute_sigma_t
            last_step_fn = \
                lambda x, t, model, score_model, **model_kwargs: \
                    x / alpha(t)[0][0] + (sigma(t)[0][0] ** 2) / alpha(t)[0][0] * self.score(x, t, score_model,
                                                                                             **model_kwargs)
        elif last_step == "Euler":
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x + self.drift(x, t, model, **model_kwargs) * last_step_size
        else:
            raise NotImplementedError()

        return last_step_fn

    def sample_sde(
            self,
            *,
            sampling_method="Euler",
            diffusion_form="SBDM",
            diffusion_norm=1.0,
            last_step="Mean",
            last_step_size=0.04,
            num_steps=250,
            score_model=None
    ):
        """returns a sampling function with given SDE settings
        Args:
        - sampling_method: type of sampler used in solving the SDE; default to be Euler-Maruyama
        - diffusion_form: function form of diffusion coefficient; default to be matching SBDM
        - diffusion_norm: function magnitude of diffusion coefficient; default to 1
        - last_step: type of the last step; default to identity
        - last_step_size: size of the last step; default to match the stride of 250 steps over [0,1]
        - num_steps: total integration step of SDE
        """

        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
        )

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        _sde = sde(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method,
            cell=self.transport.prior_cell
        )

        last_step_fn = self.__get_last_step(sde_drift, last_step=last_step, last_step_size=last_step_size)

        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, score_model, **model_kwargs)
            ts = th.ones(init.size(0), device=init.device) * t1
            x = last_step_fn(xs[-1], ts, model, score_model, **model_kwargs)
            xs.append(x)

            assert len(xs) == num_steps, "Samples does not match the number of steps"

            return xs

        return _sample


    def sample_sde_likelihood(
            self,
            *,
            sampling_method="euler_likelihood",
            diffusion_form="SBDM",
            diffusion_norm=1.0,
            last_step="Mean",
            last_step_size=0.04,
            num_steps=250,
            reverse=False,
            score_model=None
    ):
        """returns a sampling function with given SDE settings
        Args:
        - sampling_method: type of sampler used in solving the SDE; default to be Euler-Maruyama
        - diffusion_form: function form of diffusion coefficient; default to be matching SBDM
        - diffusion_norm: function magnitude of diffusion coefficient; default to 1
        - last_step: type of the last step; default to identity
        - last_step_size: size of the last step; default to match the stride of 250 steps over [0,1]
        - num_steps: total integration step of SDE
        """

        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
            reverse=reverse
        )

        reverse_sde_drift = self.__get_sde_reverse_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
            reverse=reverse
        )

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=reverse,                             # Case reverse: time integration from 1 to 0
            last_step_size=last_step_size,
        )

        _sde = sde(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method,
            reverse_drift = reverse_sde_drift,
            cell=self.transport.prior_cell
        )
        
        last_step_fn = self.__get_last_step(sde_drift, last_step=last_step, last_step_size=last_step_size)

        def _sample(init, model, **model_kwargs):
            assert not th.allclose(init, th.zeros_like(init))
            xs, logprob_xs, _logprob_xs = _sde.sample_likelihood(init, model, score_model, **model_kwargs)
            if last_step is not None:
                ts = th.ones(init.size(0), device=init.device) * t1
                x = last_step_fn(xs[-1], ts, model, score_model, **model_kwargs)
                xs.append(x)

            assert len(xs) == num_steps, "Samples does not match the number of steps"
            xs = th.stack(xs)
            return logprob_xs, _logprob_xs, xs

        return _sample

    def sample_ode(
            self,
            *,
            sampling_method="dopri5",
            num_steps=50,
            atol=1e-6,
            rtol=1e-3,
            reverse=False,
    ):
        """returns a sampling function with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps: 
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        - reverse: whether solving the ODE in reverse (data to noise); default to False
        """
        if reverse:
            drift = lambda x, t, model, **kwargs: self.drift(x, th.ones_like(t) * (1 - t), model, **kwargs)
        else:
            drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        return _ode.sample

    def sample_ode_likelihood(
            self,
            *,
            sampling_method="dopri5",
            num_steps=50,
            atol=1e-6,
            rtol=1e-3,
            reverse=False,
    ):

        """returns a sampling function for calculating likelihood with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps: 
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        """
        K_hutchinson_probe = self.transport.args.K_hutchinson_probe
        K_hutchinson_probe_chunk = self.transport.args.K_hutchinson_probe_chunk

        def _likelihood_drift(x, t, model, **model_kwargs):
            import time
            t_start = time.time()
            x, _, _ = x
            B = x.shape[0]
            if reverse:
                t = th.ones_like(t) * (1 - t)

            # for k in range(K_hutchinson_probe):
            logp_grad_samples_list = []
            drift0 = None
            with th.enable_grad():
                for k0 in range(0, K_hutchinson_probe, K_hutchinson_probe_chunk):
                    K_now = min(K_hutchinson_probe_chunk, K_hutchinson_probe - k0)
            
                    x_rep = (
                        x.detach()
                         .unsqueeze(0)
                         .repeat((K_now,) + (1,) * x.dim())
                         .reshape(K_now * B, *x.shape[1:])
                         .requires_grad_(True)
                    )
            
                    eps = th.randint(2, x_rep.size(), dtype=th.float, device=x.device) * 2 - 1
            
                    t_rep = (
                        t.detach()
                         .unsqueeze(0)
                         .repeat((K_now,) + (1,) * t.dim())
                         .reshape(K_now * B,)
                         .requires_grad_(False)
                    )
            
                    ### This way doesn't accumulate the gradient through the ODE steps
                    if reverse:
                        drift = -self.drift(x_rep, t_rep, model, **model_kwargs)
                    else:
                        drift = self.drift(x_rep, t_rep, model, **model_kwargs)
            
                    if drift0 is None:
                        # first probe copy, original batch
                        drift0 = drift[:B].detach()
            
                    grad = th.autograd.grad(th.sum((drift) * eps), x_rep)[0]
            
                    logp_grad = th.sum(
                        grad * eps,
                        dim=tuple(range(2, len(x_rep.size())))
                    )
            
                    # [K_now * B] -> [K_now, B]
                    logp_grad = logp_grad.reshape(K_now, B)
            
                    logp_grad_samples_list.append(logp_grad.detach())
            
            logp_grad_samples = th.cat(logp_grad_samples_list, dim=0)  # [K, B]

            drift = drift0.unsqueeze(0).detach()
            logp_grad_mean = logp_grad_samples.mean(dim=0)
            if K_hutchinson_probe > 1:
                logp_grad_var = logp_grad_samples.var(dim=0)/K_hutchinson_probe
            else:
                logp_grad_var = th.zeros_like(logp_grad_mean, device=x.device)
            print(f"ODE likelihood drift time: {time.time() - t_start:.4f} seconds")
            return (drift, logp_grad_mean, logp_grad_var)


        def _likelihood_drift_lattpath(input_, t, model, **model_kwargs):
            frac_, cell_, _, _ = input_
            inv_cell_ = th.linalg.inv(cell_)
            eps = th.randint(2, frac_.size(), dtype=th.float, device=frac_.device) * 2 - 1
            if reverse:
                t = th.ones_like(t) * (1 - t)
            with th.enable_grad():
                x = (frac_ @ cell_).detach().requires_grad_(True)
                frac = x @ inv_cell_
                assert x.requires_grad
                ### This way doesn't accumulate the gradient through the ODE steps
                model_kwargs['cell'] = cell_
                if reverse:
                    drift = self.drift(frac, t, model, **model_kwargs)
                    drift = list(drift)
                    drift[0] = -drift[0]
                    drift[1] = -drift[1]
                else:
                    drift = self.drift(frac, t, model, **model_kwargs)
                    drift = list(drift)
                drift_x = drift[0] @ cell_ + frac @ drift[1]
                grad = th.autograd.grad(th.sum(drift_x * eps), x)[0]
                logp_grad = th.sum(grad * eps, dim=tuple(range(2, len(x.size()))))
            
            drift[0] = drift[0].detach()
            drift[1] = drift[1].detach()
            logp_grad = logp_grad.detach()
            logp_grad_var = th.zeros_like(logp_grad, device=x.device)
            return (drift[0], drift[1], logp_grad, logp_grad_var)

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=False,
            last_step_size=0.0,
        )

        if self.transport.latt_path:
            drift = _likelihood_drift_lattpath
        else:
            drift = _likelihood_drift

        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
            latt_path=self.transport.latt_path
        )

        def _sample_fn(x, model, **model_kwargs):
            init_logp = th.zeros(x[0].size()[:2]).to(x[0]) if isinstance(x, tuple) else th.zeros(x.size()[:2]).to(x)
            init_logp_var = th.zeros(x[0].size()[:2]).to(x[0]) if isinstance(x, tuple) else th.zeros(x.size()[:2]).to(x)
            if self.transport.latt_path:
                input = (x[0], x[1], init_logp, init_logp_var)
                drift_0, drift_1, delta_logp, delta_logp_var = _ode.sample(input, model, **model_kwargs)
            else:
                input = (x, init_logp, init_logp_var)
                drift, delta_logp, delta_logp_var = _ode.sample(input, model, **model_kwargs)
            
            delta_logp = delta_logp[-1]
            # prior_logp = self.transport.prior_logp(drift)
            logp =  delta_logp
            step_size = (t1-t0)/num_steps
            if sampling_method == "rk4":
                logp_var = delta_logp_var[-1] * step_size * 5/18
            elif sampling_method == 'euler': 
                logp_var = delta_logp_var[-1] * step_size
            else:
                raise Exception(f"Variance evaluation not implemented for sampling_method: {sampling_method}")
            if self.transport.latt_path:
                return logp, [drift_0, drift_1], logp_var
            else:
                return logp, drift, logp_var

        return _sample_fn

    def sample_ode(
            self,
            *,
            sampling_method="dopri5",
            num_steps=50,
            atol=1e-6,
            rtol=1e-3,
            reverse=False,
    ):
        """returns a sampling function with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps: 
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        - reverse: whether solving the ODE in reverse (data to noise); default to False
        """
        if reverse:
            drift = lambda x, t, model, **kwargs: self.drift(x, th.ones_like(t) * (1 - t), model, **kwargs)
        else:
            drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        return _ode.sample

def create_transport(
        args,
        path_type='Linear',
        prediction="velocity",
        loss_weight=None,
        train_eps=None,
        sample_eps=None,
        score_model=None,
        latt_path = False,
        weightfunction_x = None
):
    """function for creating Transport object
    **Note**: model prediction defaults to velocity
    Args:
    - path_type: type of path to use; default to linear
    - learn_score: set model prediction to score
    - learn_noise: set model prediction to noise
    - velocity_weighted: weight loss by velocity weight
    - likelihood_weighted: weight loss by likelihood weight
    - train_eps: small epsilon for avoiding instability during training
    - sample_eps: small epsilon for avoiding instability during sampling
    """

    if prediction == "noise":
        model_type = ModelType.NOISE
    elif prediction == "score":
        model_type = ModelType.SCORE
    else:
        model_type = ModelType.VELOCITY

    if loss_weight == "velocity":
        loss_type = WeightType.VELOCITY
    elif loss_weight == "likelihood":
        loss_type = WeightType.LIKELIHOOD
    else:
        loss_type = WeightType.NONE

    path_choice = {
        "Schrodinger_Linear": PathType.LINEAR,
        "Schrodinger_Linear_onemodel": PathType.LINEAR,
        "Linear": PathType.LINEAR,
        "Pow": PathType.Pow,
        "GVP": PathType.GVP,
        "VP": PathType.VP,
    }

    path_type = path_choice[path_type]
    if (path_type in [PathType.VP]):
        train_eps = 1e-5 if train_eps is None else train_eps
        sample_eps = 1e-3 if sample_eps is None else sample_eps
    elif (path_type in [PathType.GVP, PathType.LINEAR] and model_type != ModelType.VELOCITY):
        train_eps = 1e-3 if train_eps is None else train_eps
        sample_eps = 1e-3 if sample_eps is None else sample_eps
    else:  # velocity & [GVP, LINEAR] is stable everywhere
        train_eps = 0 if train_eps is None else train_eps
        sample_eps = 0 if sample_eps is None else sample_eps

    # create flow state
    state = Transport(
        args=args,
        model_type=model_type,
        path_type=path_type,
        loss_type=loss_type,
        train_eps=train_eps,
        sample_eps=sample_eps,
        score_model=score_model,
        latt_path=latt_path,
        weightfunction_x = weightfunction_x
    )

    return state
