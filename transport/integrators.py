import numpy as np
import torch as th
import torch.nn as nn
from torchdiffeq import odeint
from functools import partial
from tqdm import tqdm
from .path import expand_t_like_x

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
    ):
        assert t0 < t1, "SDE sampler has to be in forward time"

        self.num_timesteps = num_steps
        self.t = th.linspace(t0, t1, num_steps)
        self.dt = self.t[1] - self.t[0]
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type

    def __Euler_Maruyama_step(self, x, mean_x, t, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        t = th.ones(x.size(0)).to(x) * t
        dw = w_cur * th.sqrt(self.dt)
        drift = self.drift(x, t, model, **model_kwargs)
        diffusion = self.diffusion(x, t)
        mean_x = x + drift * self.dt
        x = mean_x + th.sqrt(2 * diffusion) * dw
        return x, mean_x
    
    def __Heun_step(self, x, _, t, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        dw = w_cur * th.sqrt(self.dt)
        t_cur = th.ones(x.size(0)).to(x) * t
        diffusion = self.diffusion(x, t_cur)
        xhat = x + th.sqrt(2 * diffusion) * dw
        K1 = self.drift(xhat, t_cur, model, **model_kwargs)
        xp = xhat + self.dt * K1
        K2 = self.drift(xp, t_cur + self.dt, model, **model_kwargs)
        return xhat + 0.5 * self.dt * (K1 + K2), xhat # at last time point we do not perform the heun step

    def __forward_fn(self):
        """TODO: generalize here by adding all private functions ending with steps to it"""
        sampler_dict = {
            "Euler": self.__Euler_Maruyama_step,
            "Heun": self.__Heun_step,
        }

        try:
            sampler = sampler_dict[self.sampler_type]
        except:
            raise NotImplementedError("Smapler type not implemented.")
    
        return sampler

    def sample(self, init, model, **model_kwargs):
        """forward loop of sde"""
        x = init
        mean_x = init 
        samples = []
        sampler = self.__forward_fn()
        for ti in self.t[:-1]:
            with th.no_grad():
                x, mean_x = sampler(x, mean_x, ti, model, **model_kwargs)
                samples.append(x)

        return samples

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
        timestep_shift,
        flow_threshold=1e-4,
    ):
        assert t0 < t1, "ODE sampler has to be in forward time"

        self.drift = drift
        self.t = th.linspace(t0, t1, num_steps)
        self.flow_threshold = flow_threshold

        # Apply timestep shift: same logic as training (transport.py)
        # timestep_shift = 0 or 1.0 means no transformation
        # timestep_shift > 0 and != 1.0 means apply transformation
        if timestep_shift > 0 and timestep_shift != 1.0:
            def compute_tm(t_n, timestep_shift):
                numerator = timestep_shift * t_n
                denominator = 1 + (timestep_shift - 1) * t_n
                return numerator / denominator
            self.t = th.tensor([compute_tm(t_n, timestep_shift) for t_n in self.t])

        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type

    def sample(self, x, model, **model_kwargs):
        
        device = x[0].device if isinstance(x, tuple) else x.device
        
        # Check initial update magnitude before starting ODE integration
        # If update is too small, return early without denoising
        t_init = self.t[0].to(device) if len(self.t) > 0 else th.tensor(0.0).to(device)
        t_init_batch = th.ones(x[0].size(0)).to(device) * t_init if isinstance(x, tuple) else th.ones(x.size(0)).to(device) * t_init
        initial_flow = self.drift(x, t_init_batch, model, **model_kwargs)
        
        # Compute update magnitude: flow relative to current state
        # Calculate relative update: ||flow|| / ||x||
        if isinstance(x, tuple):
            x_flat = x[0].flatten(1)
            flow_flat = initial_flow[0].flatten(1) if isinstance(initial_flow, tuple) else initial_flow.flatten(1)
        else:
            x_flat = x.flatten(1)
            flow_flat = initial_flow.flatten(1)
        
        # Compute relative update magnitude per sample
        x_norm = th.norm(x_flat, dim=1, keepdim=True)
        flow_norm = th.norm(flow_flat, dim=1, keepdim=True)
        # Avoid division by zero
        x_norm = x_norm.clamp(min=1e-8)
        relative_update = (flow_norm / x_norm).mean()
        
        # If update is too small relative to current state, return early without denoising
        if relative_update < self.flow_threshold:
            # Return a list with just the initial state to match expected format
            if isinstance(x, tuple):
                return [x]
            else:
                return [x]
        
        def _fn(t, x):
            t = th.ones(x[0].size(0)).to(device) * t if isinstance(x, tuple) else th.ones(x.size(0)).to(device) * t
            model_output = self.drift(x, t, model, **model_kwargs)
            return model_output 
            # return (model_output - x) / th.clamp(1 - expand_t_like_x(t, x), 0.05)

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