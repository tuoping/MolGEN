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

    log_weights = -sq_norms / (2) * bandwidth_factor  # (B, N, K)
    log_weights = log_weights - log_weights.max(dim=-1, keepdim=True).values
    weights = th.exp(log_weights)
    Z = weights.sum(dim=-1)  # (B, N, 1)

    logP_ = th.logsumexp(-sq_norms / (2) * bandwidth_factor, dim=-1) - th.log(Z)  # (B, N)
    # subtract normalization (2*pi*var)^{3/2} cancels in ratio, keep for correctness
    # but original code also drops the prefactor, so we follow suit

    pred_diff = (x - mu_theta)   # @cell  # (B, N, 3)
    logQ_ = -th.sum(pred_diff ** 2, dim=-1) / (2) * bandwidth_factor  # (B, N)

    logm = th.logaddexp(logP_, logQ_) - th.log(th.tensor(2.0, device=mu_t_x1.device))

    kl_p_m = (th.exp(logP_) * (logP_ - logm)).sum(dim=-1)  # (B,)
    kl_q_m = (th.exp(logQ_) * (logQ_ - logm)).sum(dim=-1)  # (B,)

    jsd = 0.5 * kl_p_m + 0.5 * kl_q_m  # (B,)
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
            latt_path = False
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

    def sample(self, shape, device):
        """Sampling x0 & t based on shape of x1 (if needed)
          Args:
            x1 - data point; [batch, *dim]
        """
        B,T,N,C = shape
        x0 = []
        x0_mean = []
        for i in range(1):
            ### Even sample
            m = math.ceil(N ** (1/3))
            g = (th.arange(m, device=device) + 0.5) / m
            X, Y, Z = th.meshgrid(g, g, g, indexing='ij')
            _x0_mean = th.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1)[:N].unsqueeze(0).expand(T, -1, -1).unsqueeze(0).expand(B, -1, -1, -1)
            ### Uniform sample
            # _x0_mean = th.rand(shape, device=device)
            # x0.append(wrap_frac_pos(th.randn(shape, device=device)*self.args.x0std/(N)**(1./3.) + _x0_mean))
            x0.append(th.randn(shape, device=device)*self.args.x0std/(N)**(1./3.) + _x0_mean)
            x0_mean.append(_x0_mean)
        
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        # t = th.rand((x1.shape[0],))
        t, _ = sample_t_u_shaped(shape[0], self.args.beta_sample_t, eps=0)
        # t = th.zeros(shape[0])
        t = t*(t1-t0) + t0
        t = t.to(device)
        return t, x0, x0_mean


    def sample_latt(self, shape, device):
        """Sampling x0 & t based on shape of x1 (if needed)
          Args:
            x1 - data point; [batch, *dim]
        """
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

    def training_losses(
            self,
            model,
            x1,           # target tokens
            aatype1=None, # target aatype
            mask=None,
            model_kwargs=None,
            forces = None,
            global_step = None
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
        t, x0, x0_mean = self.sample(x1.shape, x1.device)
        ### OT in the atom number dimension
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
            ### exponential sampler of t
            # exponential_dist = th.distributions.Exponential(1.0)
            # t = exponential_dist.sample((seq_one_hot.shape[0],)).to(seq_one_hot.device).float()
            alphas, _ = t_to_alpha(t, self.args)
            alphas = th.ones_like(seq_one_hot) + seq_one_hot * (alphas[:, None, None, None] - th.ones_like(seq_one_hot))
            x_d = th.distributions.Dirichlet(alphas).sample()
            xt = x_d

            # model_output = model(xt, t, cell=model_kwargs["cell"], num_atoms=model_kwargs["num_atoms"], x_cond=model_kwargs["x_cond"], x_cond_mask=model_kwargs["x_cond_mask"])
        else:
            if self.score_model is None:
                xt, ut = self.path_sampler.plan_fractional(t, x0[0], x1)

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
                diffusion = self.path_sampler.compute_diffusion(x1, t, self.args.diffusion_form, self.args.diffusion_norm).view(-1)  # the input x here is not used
                xt, ut, eps = self.path_sampler.plan_schrodinger_bridge_fractional(t, x0[0], x1, diffusion)
                alpha_t, _ = self.path_sampler.compute_alpha_t(path.expand_t_like_x(t, xt))
                sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
                lambda_t = self.path_sampler.compute_lambda_schrodinger_bridge(t, diffusion)

        
        assert t.shape == (B,)
        if self.latt_path:
            model_output, lattflow_output = model(xt, t, **model_kwargs)
        else:
            model_output = model(xt, t, **model_kwargs)
        assert self.args.weight_loss_var_x0 == 0
        if self.score_model is not None:
            if self.latt_path:
                score_model_output, lattscore_output = self.score_model(xt, t, **model_kwargs)
            else:
                score_model_output = self.score_model(xt, t, **model_kwargs)

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
                if self.args.KL == 'symm':
                    cell = model_kwargs['cell'].view(B*T,3,3)
                    terms['loss_l1'] = mean_flat((model_output.view(B*T,N,3)@cell - ut.view(B*T,N,3)@cell).abs(), mask.view(B*T,N,3))
                    volume = th.abs(th.det(cell))
                    jsd = compute_jsd_loss(xt.view(B*T,N,3), 1./(self.args.x0std/(N)**(1./3.)), (x0[0]+model_output*t[:,None,None,None]).view(B*T,N,3), 3) * (volume) ** (2./3.)
                    # jsd = compute_jsd_loss(xt.view(B*T,N,3), t*(N)**(1./3.)/self.args.x0std, (x0[0]+model_output*t[:,None,None,None]).view(B*T,N,3), 3) * (volume) ** (2./3.)
                    terms['loss_symmkl'] = mean_flat(jsd, mask.view(B*T,N*3).mean(dim=-1)) 
                    terms['loss_flow'] = terms['loss_symmkl'] + terms['loss_l1'] 
                elif self.args.KL == "L1":
                    cell = model_kwargs['cell'].view(B*T,3,3)
                    terms['loss_flow'] = mean_flat((model_output.view(B*T,N,3)@cell - ut.view(B*T,N,3)@cell).abs(), mask.view(B*T,N,3))
                else:
                    raise Exception(f"Wrong KL argument: {self.args.KL}")
                if self.score_model is not None:
                    cell = model_kwargs['cell']
                    terms['loss_dsm'] = mean_flat(((lambda_t[:,None,None,None]*score_model_output + eps)**2)@cell, mask)
                    # terms['loss_tsm_0'] = mean_flat( ((score_model_output - 1./sigma_t*grad_log_normal_iso_3d(x0[0], mu=x0_mean[0], sigma=th.sqrt(2*diffusion * 1e-4)))**2 * (t < 0.5).to(th.int)[:,None,None,None])@cell, mask)
                    terms['loss_tsm_0'] = mean_flat( ((score_model_output - 1./sigma_t*grad_log_normal_iso_3d(x0[0], mu=x0_mean[0], sigma=self.args.x0std/(N)**(1./3.)))**2 * (t < 0.5).to(th.int)[:,None,None,None])@cell, mask) 
                    terms['loss_tsm_1'] = mean_flat( ((score_model_output - 1./alpha_t*forces)**2 * (t > 0.5).to(th.int)[:,None,None,None])@cell, mask)
                    terms['loss'] = terms['loss_flow'] + terms['loss_dsm'] + terms['loss_tsm_0'] + terms["loss_tsm_1"]
                else:
                    terms['loss'] = terms['loss_flow']
                    if self.latt_path:
                        lowertrigflow_output = th.stack([lattflow_output[:,:,0,0], lattflow_output[:,:,1,0], lattflow_output[:,:,1,1], lattflow_output[:,:,2,0], lattflow_output[:,:,2,1], lattflow_output[:,:,2,2]], dim=-1)
                        lowertrigulatt = th.stack([ulatt[:,:,0,0], ulatt[:,:,1,0], ulatt[:,:,1,1], ulatt[:,:,2,0], ulatt[:,:,2,1], ulatt[:,:,2,2] ], dim=-1)
                        terms['loss_lattflow'] = mean_flat((lowertrigflow_output - lowertrigulatt).abs(), th.ones_like(lowertrigflow_output, device=lowertrigflow_output.device))
                        terms['loss'] = terms['loss_flow'] + terms['loss_lattflow']
                        # terms['loss'] = terms['loss_lattflow']
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
                score_fn = lambda x, t, model, **model_kwargs: self.path_sampler.get_score_from_velocity(model(x, t, **model_kwargs), x, t)
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

        sde_drift = \
                lambda x, t, model, score_model, **kwargs: \
                    -self.drift(x, 1-t, model, **kwargs) + diffusion_fn(x, t) * self.score(x, 1-t, score_model, **kwargs)

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
                lambda x, t, model, **model_kwargs: \
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
            sampling_method="Euler_likelihood",
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
            reverse_drift = reverse_sde_drift
        )

        last_step_fn = self.__get_last_step(sde_drift, last_step=last_step, last_step_size=last_step_size)

        def _sample(init, model, **model_kwargs):
            assert not th.allclose(init, th.zeros_like(init))
            xs, logprob_xs, _logprob_xs = _sde.sample_likelihood(init, model, score_model, **model_kwargs)
            ts = th.ones(init.size(0), device=init.device) * t1
            x = last_step_fn(xs[-1], ts, model, score_model, **model_kwargs)
            xs.append(x)

            assert len(xs) == num_steps, "Samples does not match the number of steps"

            return logprob_xs, _logprob_xs, xs[-1]

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
        
        def _likelihood_drift(x, t, model, **model_kwargs):
            x, _ = x
            x = x.detach().requires_grad_(True)
            eps = th.randint(2, x.size(), dtype=th.float, device=x.device) * 2 - 1
            if reverse:
                t = th.ones_like(t) * (1 - t)
            with th.enable_grad():
                # x.requires_grad = True
                assert x.requires_grad
                ### This way doesn't accumulate the gradient through the ODE steps
                if reverse:
                    drift = -self.drift(x, t, model, **model_kwargs)
                else:
                    drift = self.drift(x, t, model, **model_kwargs)
                grad = th.autograd.grad(th.sum(drift * eps), x)[0]
                logp_grad = th.sum(grad * eps, dim=tuple(range(2, len(x.size()))))
            return (drift, logp_grad)

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=False,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=_likelihood_drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        def _sample_fn(x, model, **model_kwargs):
            init_logp = th.zeros(x.size()[:2]).to(x)
            input = (x, init_logp)
            drift, delta_logp = _ode.sample(input, model, **model_kwargs)
            drift, delta_logp = drift[-1], delta_logp[-1]
            # prior_logp = self.transport.prior_logp(drift)
            if reverse:
                logp =  delta_logp
            else:
                logp =  delta_logp
            return logp, drift

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
        latt_path = False
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
        latt_path=latt_path
    )

    return state
