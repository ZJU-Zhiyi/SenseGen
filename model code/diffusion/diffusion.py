"""
Gaussian Diffusion: DDPM forward/reverse process.
Designed for SenseGen offset-conditioned sensing data generation.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Support both module import and direct script execution
try:
    from .config import DiffusionConfig
except ImportError:
    from config import DiffusionConfig


# ---------------------------------------------------------------------------
#  Noise Schedule
# ---------------------------------------------------------------------------

def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    """Cosine schedule from "Improved DDPM" with numerical stability."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, min=1e-5, max=0.999)


# ---------------------------------------------------------------------------
#  EMA Wrapper
# ---------------------------------------------------------------------------

class EMA(nn.Module):
    """Exponential Moving Average wrapper for model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        super().__init__()
        self.decay = decay
        self.model = model
        self.shadow = deepcopy(model)
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self):
        for s_param, m_param in zip(self.shadow.parameters(), self.model.parameters()):
            s_param.data = s_param.data * self.decay + m_param.data * (1.0 - self.decay)

    def forward(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)


# ---------------------------------------------------------------------------
#  Gaussian Diffusion
# ---------------------------------------------------------------------------

class GaussianDiffusion(nn.Module):
    """DDPM sampler with SDEdit-style anchor generation support."""

    def __init__(
        self,
        model: nn.Module,
        config: Optional[DiffusionConfig] = None,
    ):
        super().__init__()
        self.model = model
        self.config = config or DiffusionConfig()
        cfg = self.config

        # Build schedule
        if cfg.beta_schedule == "cosine":
            betas = cosine_beta_schedule(cfg.num_timesteps)
        else:
            betas = linear_beta_schedule(cfg.num_timesteps, cfg.beta_start, cfg.beta_end)

        betas = betas.float()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Precompute with numerical stability
        one_minus_ac = (1.0 - alphas_cumprod).clamp(min=1e-12)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             (1.0 - alphas_cumprod).clamp(min=0.0).sqrt())
        self.register_buffer("sqrt_recip_alphas_cumprod",
                             (1.0 / alphas_cumprod.clamp(min=1e-12)).sqrt())
        self.register_buffer("sqrt_recipm1_alphas_cumprod",
                             (1.0 / alphas_cumprod.clamp(min=1e-12) - 1.0).clamp(min=0.0).sqrt())

        # Posterior q(x_{t-1} | x_t, x_0) — clip denominator to avoid div-by-zero
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / one_minus_ac
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped",
                             torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer("posterior_mean_coef1",
                             betas * alphas_cumprod_prev.sqrt() / one_minus_ac)
        self.register_buffer("posterior_mean_coef2",
                             (1.0 - alphas_cumprod_prev) * alphas.sqrt() / one_minus_ac)

    # ------------------------------------------------------------------
    #  Forward process
    # ------------------------------------------------------------------

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample x_t ~ q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self._gather(self.sqrt_alphas_cumprod, t)
        sqrt_one_minus = self._gather(self.sqrt_one_minus_alphas_cumprod, t)
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    # ------------------------------------------------------------------
    #  Training
    # ------------------------------------------------------------------

    def training_loss(self, x0: torch.Tensor, cond: torch.Tensor,
                      noise: Optional[torch.Tensor] = None,
                      cfg_drop_prob: float = 0.0) -> torch.Tensor:
        """Compute simplified DDPM loss (L_simple) with CFG conditioning dropout.

        Args:
            x0: clean spectrogram (B, 1, H, W)
            cond: conditioning (B, 2, H, W) -- [static_offset, dynamic_offset]
            cfg_drop_prob: probability of dropping conditioning for CFG training

        Returns:
            Scalar MSE loss
        """
        B = x0.shape[0]
        t = torch.randint(0, self.config.num_timesteps, (B,), device=x0.device)

        if noise is None:
            noise = torch.randn_like(x0)

        # CFG: randomly drop conditioning
        if cfg_drop_prob > 0:
            drop_mask = torch.rand(B, device=x0.device) < cfg_drop_prob
            if drop_mask.any():
                cond = cond.clone()
                cond[drop_mask] = 0.0

        xt = self.q_sample(x0, t, noise)
        model_input = torch.cat([xt, cond], dim=1)  # (B, 3, H, W)
        pred_noise = self.model(model_input, t)

        # Clip extreme predictions for numerical stability
        pred_noise = torch.clamp(pred_noise, -10.0, 10.0)

        return F.mse_loss(pred_noise, noise)

    # ------------------------------------------------------------------
    #  Reverse process helpers
    # ------------------------------------------------------------------

    def _gather(self, buf: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Gather values from buffer by timestep indices."""
        return buf.gather(-1, t.clamp(max=buf.shape[-1]-1)).reshape(-1, 1, 1, 1)

    def predict_start_from_noise(self, xt: torch.Tensor, t: torch.Tensor,
                                  noise: torch.Tensor) -> torch.Tensor:
        """x_0 prediction from x_t and predicted noise."""
        sqrt_recip = self._gather(self.sqrt_recip_alphas_cumprod, t)
        sqrt_recipm1 = self._gather(self.sqrt_recipm1_alphas_cumprod, t)
        return sqrt_recip * xt - sqrt_recipm1 * noise

    def q_posterior(self, x0: torch.Tensor, xt: torch.Tensor,
                    t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and log variance of q(x_{t-1} | x_t, x_0)."""
        mean_coef1 = self._gather(self.posterior_mean_coef1, t)
        mean_coef2 = self._gather(self.posterior_mean_coef2, t)
        mean = mean_coef1 * x0 + mean_coef2 * xt
        log_var = self._gather(self.posterior_log_variance_clipped, t)
        return mean, log_var

    # ------------------------------------------------------------------
    #  DDPM Sampling (ancestral)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, cond: torch.Tensor,
                 clip_denoised: bool = True, cfg_scale: float = 1.0) -> torch.Tensor:
        """Single DDPM reverse step: x_t -> x_{t-1}.

        Args:
            cfg_scale: classifier-free guidance scale. 1.0 = no guidance.
        """
        if cfg_scale == 1.0:
            model_input = torch.cat([xt, cond], dim=1)
            pred_noise = self.model(model_input, t)
        else:
            # CFG: batch conditional and unconditional forward
            zero_cond = torch.zeros_like(cond)
            xt_double = torch.cat([xt, xt], dim=0)
            cond_double = torch.cat([cond, zero_cond], dim=0)
            t_double = torch.cat([t, t], dim=0)
            model_input = torch.cat([xt_double, cond_double], dim=1)
            preds = self.model(model_input, t_double)
            pred_cond, pred_uncond = preds.chunk(2, dim=0)
            pred_noise = pred_uncond + cfg_scale * (pred_cond - pred_uncond)

        x0 = self.predict_start_from_noise(xt, t, pred_noise)

        if clip_denoised:
            x0 = x0.clamp(-3.0, 3.0)

        mean, log_var = self.q_posterior(x0, xt, t)
        noise = torch.randn_like(xt) if t.min() > 0 else torch.zeros_like(xt)
        return mean + (0.5 * log_var).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, cond: torch.Tensor, shape: Tuple[int, ...],
                      device: torch.device = torch.device("cpu"),
                      progress: bool = False,
                      cfg_scale: float = 1.0) -> torch.Tensor:
        """Full DDPM reverse chain: x_T -> x_0.

        Args:
            cfg_scale: classifier-free guidance scale. 1.0 = no guidance.
        """
        B = cond.shape[0]
        img = torch.randn((B, *shape), device=device)

        indices = list(range(self.config.num_timesteps))[::-1]
        if progress:
            from tqdm import tqdm
            indices = tqdm(indices, desc="DDPM sampling")

        for step in indices:
            t = torch.full((B,), step, device=device, dtype=torch.long)
            img = self.p_sample(img, t, cond, cfg_scale=cfg_scale)

        return img

    @torch.no_grad()
    def p_sample_loop_from(self, cond: torch.Tensor, x_start: torch.Tensor,
                           t_start: int, device: torch.device = torch.device("cpu"),
                           progress: bool = False,
                           cfg_scale: float = 1.0) -> torch.Tensor:
        """SDEdit-style DDPM reverse: x_{t_start} -> x_0.

        Noises anchor x_start to timestep t_start, then denoises conditioned
        on offset. t_start=500 gives moderate variation from anchor.

        Args:
            cond: (B, 2, H, W) offset conditioning.
            x_start: (B, 1, H, W) normalized anchor sample.
            t_start: starting timestep (0-1000). Higher = more variation.
            cfg_scale: classifier-free guidance scale.
        """
        B = cond.shape[0]
        noise = torch.randn_like(x_start)
        t_tensor = torch.full((B,), t_start, device=device, dtype=torch.long)
        img = self.q_sample(x_start, t_tensor, noise)

        indices = list(range(t_start))[::-1]
        if progress:
            from tqdm import tqdm
            indices = tqdm(indices, desc=f"DDPM SDEdit t={t_start}")

        for step in indices:
            t = torch.full((B,), step, device=device, dtype=torch.long)
            img = self.p_sample(img, t, cond, cfg_scale=cfg_scale)

        return img

