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
            "constant": norm,
            "SBDM": norm * self.compute_drift(x, t)[1],
            "sigma": norm * self.compute_sigma_t(t)[0],
            "linear": norm * (1 - t),
            "decreasing": 0.25 * (norm * th.cos(np.pi * t) + 1) ** 2,
            "inccreasing-decreasing": norm * th.sin(np.pi * t) ** 2,
        }

        try:
            diffusion = choices[form]
        except KeyError:
            raise NotImplementedError(f"Diffusion form {form} not implemented")
        
        return diffusion

    def get_score_from_velocity(self, velocity, x, t):
        """Wrapper function: transfrom velocity prediction model to score
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
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * velocity - mean) / var
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
        return t, xt, ut
    
    def compute_marginal_std(self, t, diffusion):
        """Compute the marginal standard deviation of the time-dependent density p_t"""
        return th.sqrt(2*diffusion) * th.sqrt(t*(1-t))

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
        mu_t = self.compute_mu_t(x0, x1, t)
        std_t = self.compute_marginal_std(t, diffusion)
        std_t = expand_t_like_x(std_t, x0)
        return mu_t + std_t * epsilon

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
        return t, xt, ut, epsilon
    
    def compute_lambda_schrodinger_bridge(self, t, diffusion):
        '''
        Compute the lambda function for the Schrodinger bridge.
        Diffusion rate: g(t) = sqrt(2 * diffusion)
        lambda_t = 2*(g(t)**2*sqrt(t*(1-t)))/(g(t)**2+1e-8)
        Parameters
        ----------
        t : FloatTensor, shape (bs)
            time vector
        diffusion : FloatTensor, shape (bs)
            diffusion constant of the SDE
        Returns
        -------
        lambda_t : FloatTensor, shape (bs)
            lambda function at time t
        '''
        std_t = self.compute_marginal_std(t, diffusion)
        return 2*std_t/(2*diffusion+1e-8)


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


class PowPlan(ICPlan):
    def __init__(self, sigma=0.0):
        super().__init__(sigma)

    def compute_alpha_t(self, t, p=2.0):
        # alpha(1)=0 and d/dt alpha -> 0 as t->1
        alpha  = th.power((1 - t), (p))              # p>=2 makes slope vanish
        dalpha = -p * th.power((1 - t), (p - 1))
        return alpha, dalpha

    def compute_sigma_t(self, t, sigma_min=1e-3, sigma_max=1.0, q=2.0):
        # sigma rises to sigma_max with zero slope at t=1
        s      = 1 - th.power((1 - t), (q))          # smooth cap; q>=2
        ds     = q * th.power((1 - t), (q - 1))
        sigma  = sigma_min + (sigma_max - sigma_min) * s
        dsigma = (sigma_max - sigma_min) * ds
        return sigma, dsigma

    def compute_d_alpha_alpha_ratio_t(self, t, p=2.0, tol=1e-9):
        """Special purposed function for computing numerical stabled d_alpha_t / alpha_t"""
        alpha, dalpha = self.compute_alpha_t(t, p=p)
        if th.abs(alpha) < tol:
            raise ValueError(f"alpha is too small: {alpha}, t={t}, p={p}")
        return dalpha / alpha
    
    def plan_schrodinger_bridge(self, t, x0, x1, diffusion):
        """
        Plan for Schrodinger equation
        diffusion = (g(t)**2)/2
        """
        xt = self.compute_xt(t, x0, x1)
        ut = (1-2*t)/(t*(1-t)) * (xt - (t*x1 - (1-t)*x0)) + (x1-x0)
        score = (t*x1+(1-t)*x0 -xt)/(2*diffusion)/t/(1-t)
        return t, xt, ut, score

    

class TimeWarpPlan(ICPlan):
    def __init__(self, sigma=0.0):
        super().__init__(sigma)
    
    def _phi(self, t, kind="smoothstep"):
        if kind == "smoothstep":  # 3 t^2 - 2 t^3
            s  = t*t*(3 - 2*t)
            ds = 6 * t * (1 - t)
        else:  # cosine
            import math
            s  = 0.5 - 0.5 * np.cos(math.pi * t)
            ds = 0.5 * math.pi * np.sin(math.pi * t)
        return s, ds

    def compute_alpha_t(self, t, kind="smoothstep", p=1.0):
        # Optional p>=1 to taper even flatter near t=1: alpha=(1-phi)^p
        s, ds = self._phi(t, kind)
        alpha  = np.power((1 - s), p)
        dalpha = -p * np.power((1 - s), p - 1) * ds
        return alpha, dalpha

    def compute_sigma_t(self, t, kind="smoothstep", sigma_min=0.0):
        # Anchored to x1: sigma(0)=sigma_min (often 0), sigma(1)=1, with zero slope at ends
        s, ds = self._phi(t, kind)
        sigma  = sigma_min + (1 - sigma_min) * s
        dsigma = (1 - sigma_min) * ds
        return sigma, dsigma
    
    def compute_d_alpha_alpha_ratio_t(self, t, kind='smoothstep', tol=1e-9):
        """Special purposed function for computing numerical stabled d_alpha_t / alpha_t"""
        alpha, dalpha = self.compute_alpha_t(t, kind)
        if th.abs(alpha) < tol:
            raise ValueError(f"alpha is too small: {alpha}, t={t}, kind={kind}")
        return dalpha / alpha
