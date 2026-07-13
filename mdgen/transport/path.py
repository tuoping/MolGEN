# https://github.com/willisma/SiT/
import torch as th
import numpy as np
def expand_t_like_x(t, x):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t

def wrap_frac_pos(F):
    return th.remainder(F, 1.0)



def matrix_log_eig(A: th.Tensor) -> th.Tensor:
    """Matrix log via eigendecomposition. Works for any diagonalizable matrix."""
    eigenvalues, V = th.linalg.eig(A)  # complex
    log_diag = th.diag_embed(th.log(eigenvalues))
    result = V @ log_diag @ th.linalg.inv(V)
    return result.real  # discard numerical imaginary noise

def geodesic_gl3(L0: th.Tensor, L1: th.Tensor, t: th.Tensor) -> th.Tensor:
    """
    Geodesic on GL+(3) with left-invariant metric.
    L_t = L0 @ expm(t * logm(L0^{-1} @ L1))
    Args:
        L0: (batch, 3, 3) source matrices
        L1: (batch, 3, 3) target matrices
        t:  (batch, 1, 1) or scalar, time in [0, 1]
    Returns:
        Lt: (batch, 3, 3)
    """
    L0_inv = th.linalg.inv(L0)
    V = matrix_log_eig(L0_inv @ L1)
    Lt = L0 @ th.linalg.matrix_exp(t * V)
    return Lt
    
def velocity_gl3(L0: th.Tensor, L1: th.Tensor, t: th.Tensor) -> th.Tensor:
    """
    Conditional vector field: u_t = L_t @ V where V = logm(L0^{-1} @ L1)
    Returns:
        u_Lt: (batch, 3, 3) velocity in ambient R^{3x3}
    """
    L0_inv = th.linalg.inv(L0)
    V = matrix_log_eig(L0_inv @ L1)
    Lt = L0 @ th.linalg.matrix_exp(t * V)
    u_Lt = Lt @ V
    return u_Lt   


def compute_weighted(dx, std_t, f_x, k_max=3):
    """
    Monte Carlo estimate of the full JSD:
    JSD(p || q) = 0.5 * E_p[log(p/m)] + 0.5 * E_q[log(q/m)]

    Args:
        dx: (B, N, 3) - x-\mu, where \mu is the means for the wrapped Gaussian
        standard_bandwidth_factor: scalar or (B,) - 1/\sigma for the wrapped Gaussion
        f_x: function - target function to weight
            input: x is (B, N, 3), 
            input: k_vecs is (K, 3)
            output: (B, N, K)
        k_max: int - number of periodic images per dimension
    """
    inv_std_t = th.linalg.inv(std_t)

    ks = th.arange(-k_max, k_max + 1, device=dx.device)
    kx, ky, kz = th.meshgrid(ks, ks, ks, indexing='ij')
    k_vecs = th.stack([kx.ravel(), ky.ravel(), kz.ravel()], dim=-1).float()  # (K, 3)

    # Broadcast: mu_t_x1 is (B, N, 3), k_vecs is (K, 3)
    # diff: (B, N, K, 3)
    diff = (dx[:, :, None, :] + k_vecs[None, None, :, :]) @ inv_std_t   # @cell[:, None, :, :]
    sq_norms = th.sum(diff ** 2, dim=-1)  # (B, N, K)

    log_weights = -sq_norms / (2) # (B, N, K)
    log_weights = log_weights - log_weights.max(dim=-1, keepdim=True).values
    weights = th.exp(log_weights)
    Z = weights.sum(dim=-1, keepdim=True)  # (B, N, 1)

    weighted_sum = ((weights/Z)[:,:,:,None] * f_x(dx[:, :, None, :], k_vecs[None, None, :, :])).sum(dim=-2)
    return weighted_sum

#################### Coupling Plans ####################

class ICPlan:
    """Linear Coupling Plan"""
    def __init__(self, sigma=0.0):
        self.sigma = sigma

    def compute_alpha_t(self, t):
        """Compute the data coefficient along the path"""
        return t, 1
    
    def compute_sigma_t(self, t):
        """Compute the noise coefficient along the path"""
        return 1 - t, -1
    
    def compute_d_alpha_alpha_ratio_t(self, t):
        """Compute the ratio between d_alpha and alpha"""
        return 1 / t

    def compute_drift(self, x, t):
        """We always output sde according to score parametrization; """
        t = expand_t_like_x(t, x)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        drift = alpha_ratio * x
        diffusion = alpha_ratio * (sigma_t ** 2) - sigma_t * d_sigma_t

        return -drift, diffusion

    def compute_diffusion(self, x, t, form="constant", norm=1.0):
        """Compute the diffusion term of the SDE
        Args:
          x: [batch_dim, ...], data point
          t: [batch_dim,], time vector
          form: str, form of the diffusion term
          norm: float, norm of the diffusion term
        """
        t = expand_t_like_x(t, x)
        choices = {
            "constant": norm * th.ones_like(t),
            "SBDM": norm * self.compute_drift(x, t)[1],
            "sigma": norm * self.compute_sigma_t(t)[0],
            "linear": norm * (1 - t),
            "decreasing": norm * 0.25 * (th.cos(np.pi * t) + 1) ** 2,
            "increasing-decreasing": norm * th.sin(np.pi * t) ** 2,
        }

        try:
            diffusion = choices[form]
        except KeyError:
            raise NotImplementedError(f"Diffusion form {form} not implemented")
        assert diffusion.dim() == 4, "Diffusion term must be a 4D tensor"
        return diffusion

    def get_score_from_velocity(self, velocity, x, t, x0std):
        """
        Convert the learned IC control / velocity field b_t(x) to the
        log-density score ∇_x log p_t(x).

        Sign convention:
            This returns ∇ log p_t, NOT ∇U_t.
            Therefore it matches the SDE drift convention

                drift = b_t + diffusion * score @ metric

            used in transport.py.

        IC path:
            x_t = alpha_t * x1 + sigma_t * x0
            b_t = E[d/dt x_t | x_t]

        For alpha_t=t, sigma_t=1-t, zero-mean unit-Gaussian x0:

            score = (t * velocity - x) / (1 - t)
        """
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * velocity - mean) / var /(x0std**2)[:,None,None]
        assert score.dim() == 4, "Score term must be a 4D tensor"
        return score
    
    def get_noise_from_velocity(self, velocity, x, t):
        """Wrapper function: transfrom velocity prediction model to denoiser
        Args:
            velocity: [batch_dim, ...] shaped tensor; velocity model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = reverse_alpha_ratio * d_sigma_t - sigma_t
        noise = (reverse_alpha_ratio * velocity - mean) / var
        return noise

    def get_velocity_from_score(self, score, x, t):
        """Wrapper function: transfrom score prediction model to velocity
        Args:
            score: [batch_dim, ...] shaped tensor; score model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = expand_t_like_x(t, x)
        drift, var = self.compute_drift(x, t)
        velocity = var * score - drift
        return velocity

    def compute_mu_t(self, t, x0, x1):
        """Compute the mean of time-dependent density p_t"""
        t = expand_t_like_x(t, x1)
        alpha_t, _ = self.compute_alpha_t(t)
        sigma_t, _ = self.compute_sigma_t(t)
        return alpha_t * x1 + sigma_t * x0
    
    def compute_xt(self, t, x0, x1):
        """Sample xt from time-dependent density p_t; rng is required"""
        xt = self.compute_mu_t(t, x0, x1)
        return xt
    
    def compute_ut(self, t, x0, x1, xt):
        """Compute the vector field corresponding to p_t"""
        t = expand_t_like_x(t, x1)
        _, d_alpha_t = self.compute_alpha_t(t)
        _, d_sigma_t = self.compute_sigma_t(t)
        return d_alpha_t * x1 + d_sigma_t * x0
    
    def plan(self, t, x0, x1):
        xt = self.compute_xt(t, x0, x1)
        ut = self.compute_ut(t, x0, x1, xt)
        return xt, ut
    
    def plan_fractional(self, t, x0, x1):
        t = expand_t_like_x(t, x1)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        xt = wrap_frac_pos(x0 + t*(wrap_frac_pos(x1 - x0 - 0.5) - 0.5))
        ut = wrap_frac_pos(x1 - x0 - 0.5) - 0.5
        return xt, ut 

    def plan_latt(self, t, latt0, latt1):
        t = expand_t_like_x(t, latt1)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        latt = latt1 * alpha_t + latt0 * sigma_t
        ulatt = d_alpha_t * latt1 + d_sigma_t * latt0
        return latt, ulatt
    
    def plan_latt_riemann(self, t, latt0, latt1):
        t = expand_t_like_x(t, latt1)
        # alpha_t, d_alpha_t = self.compute_alpha_t(t)
        # sigma_t, d_sigma_t = self.compute_sigma_t(t)
        latt = geodesic_gl3(latt0, latt1, t)
        ulatt = velocity_gl3(latt0, latt1, t)
        return latt, ulatt
    
    def compute_marginal_std(self, t, diffusion, cell=th.eye(3).unsqueeze(0).unsqueeze(0)):
        """Compute the marginal standard deviation of the time-dependent density p_t"""
        inv_cell = th.linalg.inv(cell)
        return (th.sqrt(2*diffusion) * th.sqrt(t*(1-t))[:,None,None,None]) * inv_cell

    def sample_xt_schrodinger_bridge(self, x0, x1, t, epsilon, diffusion):
        """
        Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Diffusion rate: g(t) = sqrt(2 * diffusion)
        std deviation of the marginal Gaussian distribution at time t is: std_t = g(t) * sqrt(t * (1 - t))
        
        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        epsilon : Tensor, shape (bs, *dim)
            noise sample from N(0, 1)

        Returns
        -------
        xt : Tensor, shape (bs, *dim)

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        mu_t = self.compute_mu_t(t, x0, x1)
        std_t = self.compute_marginal_std(t, diffusion)
        std_t = expand_t_like_x(std_t, x0)
        return mu_t + epsilon@std_t

    def compute_ut_schrodinger_bridge(self, t, x0, x1, xt):
        """
        Compute the vector field corresponding to the Schrodinger bridge path.

        Diffusion rate: g(t) = sqrt(2 * diffusion)
        Flow field is given by:
        ut = sigma_t_prime_over_sigma_t * (xt - mu_t) + ut_ode, where sigma_t_prime_over_sigma_t = (1 - 2 * t) / (2 * t * (1 - t) + 1e-8);
        Score field is given by:
        s = (xt - mu_t) / (2 * diffusion * t * (1 - t))

        Parameters
        ----------
        t : FloatTensor, shape (bs)
            time vector
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        xt : Tensor, shape (bs, *dim)
            sampled point at time t

        Returns
        -------
        ut : Tensor, shape (bs, *dim)
            vector field at time t
        """
        t = expand_t_like_x(t, x1)
        mu_t = self.compute_mu_t(t, x0, x1)
        sigma_t_prime_over_sigma_t = (1 - 2 * t) / (2 * t * (1 - t) + 1e-8)
        ut_ode = self.compute_ut(t, x0, x1, xt)
        ut = sigma_t_prime_over_sigma_t * (xt - mu_t) + ut_ode
        return ut

    def plan_schrodinger_bridge(self, t, x0, x1, diffusion):
        """
        Plan for Schrodinger equation
        Diffusion rate: g(t) = sqrt(2 * diffusion)

        """
        epsilon = th.randn_like(x0)
        xt = self.sample_xt_schrodinger_bridge(x0, x1, t, epsilon, diffusion)
        ut = self.compute_ut_schrodinger_bridge(t, x0, x1, xt)
        return xt, ut, epsilon
    
    
    def plan_schrodinger_bridge_fractional(self, t, x0, x1, diffusion, cell):
        B,T,N,_ = x0.shape
        epsilon = th.randn_like(x0)
        mu_t = x0 + t[:,None,None,None]*(wrap_frac_pos(x1 - x0 - 0.5) - 0.5)
        std_t = self.compute_marginal_std(t, diffusion, cell)
        xt = mu_t + epsilon@std_t

        sigma_t_prime_over_sigma_t = (1 - 2 * t) / (2 * t * (1 - t) + 1e-8)
        ut_ode = (wrap_frac_pos(x1 - x0 - 0.5) - 0.5).view(B*T,N,3)
        def ut_k(dx, k):
            return sigma_t_prime_over_sigma_t[:,None,None,None]*(dx + k) + ut_ode[:,:,None,:]
        
        ut = compute_weighted((epsilon@std_t).view(B*T,N,3), std_t, ut_k).view(B,T,N,3)
        return xt, ut, epsilon


class VPCPlan(ICPlan):
    """class for VP path flow matching"""

    def __init__(self, sigma_min=0.1, sigma_max=20.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.log_mean_coeff = lambda t: -0.25 * ((1 - t) ** 2) * (self.sigma_max - self.sigma_min) - 0.5 * (1 - t) * self.sigma_min 
        self.d_log_mean_coeff = lambda t: 0.5 * (1 - t) * (self.sigma_max - self.sigma_min) + 0.5 * self.sigma_min


    def compute_alpha_t(self, t):
        """Compute coefficient of x1"""
        alpha_t = self.log_mean_coeff(t)
        alpha_t = th.exp(alpha_t)
        d_alpha_t = alpha_t * self.d_log_mean_coeff(t)
        return alpha_t, d_alpha_t
    
    def compute_sigma_t(self, t):
        """Compute coefficient of x0"""
        p_sigma_t = 2 * self.log_mean_coeff(t)
        sigma_t = th.sqrt(1 - th.exp(p_sigma_t))
        d_sigma_t = th.exp(p_sigma_t) * (2 * self.d_log_mean_coeff(t)) / (-2 * sigma_t)
        return sigma_t, d_sigma_t
    
    def compute_d_alpha_alpha_ratio_t(self, t):
        """Special purposed function for computing numerical stabled d_alpha_t / alpha_t"""
        return self.d_log_mean_coeff(t)

    def compute_drift(self, x, t):
        """Compute the drift term of the SDE"""
        t = expand_t_like_x(t, x)
        beta_t = self.sigma_min + (1 - t) * (self.sigma_max - self.sigma_min)
        return -0.5 * beta_t * x, beta_t / 2
    

class GVPCPlan(ICPlan):
    def __init__(self, sigma=0.0):
        super().__init__(sigma)
    
    def compute_alpha_t(self, t):
        """Compute coefficient of x1"""
        alpha_t = th.sin(t * np.pi / 2)
        d_alpha_t = np.pi / 2 * th.cos(t * np.pi / 2)
        return alpha_t, d_alpha_t
    
    def compute_sigma_t(self, t):
        """Compute coefficient of x0"""
        sigma_t = th.cos(t * np.pi / 2)
        d_sigma_t = -np.pi / 2 * th.sin(t * np.pi / 2)
        return sigma_t, d_sigma_t
    
    def compute_d_alpha_alpha_ratio_t(self, t):
        """Special purposed function for computing numerical stabled d_alpha_t / alpha_t"""
        return np.pi / (2 * th.tan(t * np.pi / 2))

