# https://github.com/willisma/SiT/
import torch as th
from torchdiffeq import odeint

class sde:
    """SDE solver class"""
    def __init__(
        self, 
        drift,
        diffusion,
        *,
        t0,
        t1,
        num_steps,
        sampler_type,
        score = None,
        num_corrector_step = 0,
    ):
        assert t0 < t1, "SDE sampler has to be in forward time"

        self.inference_steps = num_steps
        # self.t = th.linspace(t0, t1, num_steps)
        alpha = 2.0
        u = th.linspace(0, 1, num_steps)
        self.t = t1 - (t1 - t0) * (1 - u)**alpha
        self.dt = self.t[1] - self.t[0]
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type
        self.score = score
        self.num_corrector_step = num_corrector_step

    def __Euler_Maruyama_step(self, x, mean_x, t, model, score_model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        t = th.ones(x.size(0)).to(x) * t
        dw = w_cur * th.sqrt(self.dt)
        drift = self.drift(x, t, model, score_model, **model_kwargs)
        diffusion = self.diffusion(x, t)
        mean_x = x + drift * self.dt
        x = mean_x + th.sqrt(2 * diffusion) * dw
        if self.score is not None:
            for i in range(self.num_corrector_step):
                x, mean_x = self.__corrector_step(x, t, score_model, **model_kwargs)
        return x, mean_x
    
    def __corrector_step(self, x, t, score_model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        score = self.score(x, t, score_model, **model_kwargs)
        # epsilon = 2 * (0.2 * th.linalg.norm(w_cur, dim=-1).mean()
        #        / th.linalg.norm(score, dim=-1).mean()) ** 2
        epsilon = 2 * (th.sqrt(th.tensor(3.0)) * 0.005
               / th.linalg.norm(score, dim=-1).mean()) ** 2
        mean_x = x + epsilon * score
        x = mean_x + th.sqrt(2*epsilon)*w_cur
        return x, mean_x
    
    def __Heun_step(self, x, _, t, model, score_model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        dw = w_cur * th.sqrt(self.dt)
        t_cur = th.ones(x.size(0)).to(x) * t
        diffusion = self.diffusion(x, t_cur)
        xhat = x + th.sqrt(2 * diffusion) * dw
        K1 = self.drift(xhat, t_cur, model, score_model, **model_kwargs)
        xp = xhat + self.dt * K1
        K2 = self.drift(xp, t_cur + self.dt, model, score_model, **model_kwargs)
        return xhat + 0.5 * self.dt * (K1 + K2), xhat # at last time point we do not perform the heun step

    def __forward_fn(self):
        """TODO: generalize here by adding all private functions ending with steps to it"""
        sampler_dict = {
            "euler": self.__Euler_Maruyama_step,
            "Heun": self.__Heun_step,
        }

        try:
            sampler = sampler_dict[self.sampler_type]
        except:
            raise NotImplementedError("Smapler type not implemented.")
    
        return sampler

    def sample(self, init, model, score_model, **model_kwargs):
        """forward loop of sde"""
        x = init
        mean_x = init 
        
        samples = []
        sampler = self.__forward_fn()
        for ti in self.t[:-1]:
            with th.no_grad():
                x, mean_x = sampler(x, mean_x, ti, model, score_model, **model_kwargs)
                samples.append(x)

        return samples

from . import path
from . import transport
class ode:
    """ODE solver class"""
    def __init__(
        self,
        drift,
        *,
        t0,
        t1,
        sampler_type,
        num_steps,
        atol,
        rtol,
        latt_path = False
    ):
        assert t0 < t1, "ODE sampler has to be in forward time"

        self.drift = drift
        self.t = th.linspace(t0, t1, num_steps)
        # self.t = t0 + (t1 - t0) * (1 - (1 - th.linspace(0, 1, num_steps))**2)
        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type

        self.path_sampler = path.ICPlan()
        self.latt_path = latt_path

    def sample(self, x, model, **model_kwargs):
        device = x[0].device if isinstance(x, tuple) else x.device
        def _fn(t, x):
            t = th.ones(x[0].size(0)).to(device) * t if isinstance(x, tuple) else th.ones(x.size(0)).to(device) * t
            if isinstance(x, tuple) and self.latt_path:
                model_kwargs['cell'] = x[1]
                model_output = self.drift(x[0], t, model, **model_kwargs)  
            else:
                model_output = self.drift(x, t, model, **model_kwargs)
            return model_output

        t = self.t.to(device)
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
        samples = odeint(
            _fn,
            x,
            t,
            method=self.sampler_type,
            atol=atol,
            rtol=rtol
        )
        return samples
