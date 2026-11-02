# Copyright (c) 2026, DYROS.
# SPDX-License-Identifier: BSD-3-Clause

"""Tanh-squashed Gaussian distribution for bounded RSL-RL actions."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.distributions import Normal

from rsl_rl.modules.distribution import GaussianDistribution


class TanhSquashedGaussianDistribution(GaussianDistribution):
    r"""Gaussian policy transformed by ``tanh`` into the open interval ``(-1, 1)``.

    PPO receives actions from the transformed distribution and therefore must evaluate
    their likelihood with the tanh change-of-variables correction.  The KL used by
    RSL-RL's adaptive learning-rate scheduler is inherited from the latent Gaussian;
    KL is invariant under the same invertible transform applied to both policies.

    Entropy has no analytic form after the tanh transform.  It is estimated from the
    reparameterized sample already generated for the PPO update.  This keeps the
    entropy gradient consistent with the bounded action distribution instead of the
    unbounded latent Gaussian.

    The latent mean is clamped to ``[-mean_bound, mean_bound]``.  Beyond ``|mean| ~ 3``
    the squashed action no longer changes, so neither the reward nor the action penalties
    hold the mean back and it can drift without limit.  Once it passes ~9, float32
    ``tanh`` returns exactly 1.0, :meth:`log_prob` can only recover ``atanh(1 - epsilon)``
    (~7.25), and a tiny mean update overflows the PPO ratio to ``inf`` (NaN actor).
    The clamp also zeroes the gradient past the bound, which stops the drift itself.
    Deterministic inference keeps the plain ``tanh`` so exported actors are unchanged;
    the mismatch is at most ``1 - tanh(mean_bound)`` (9e-5 for the default 5.0).
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 1.0,
        std_type: str = "log",
        epsilon: float = 1.0e-6,
        mean_bound: float | None = 5.0,
    ) -> None:
        if not 0.0 < epsilon < 1.0:
            raise ValueError(f"epsilon must be in (0, 1), got {epsilon}")
        if mean_bound is not None and not mean_bound > 0.0:
            raise ValueError(f"mean_bound must be positive or None, got {mean_bound}")
        super().__init__(output_dim=output_dim, init_std=init_std, std_type=std_type)
        self.epsilon = float(epsilon)
        self.mean_bound = None if mean_bound is None else float(mean_bound)
        self._last_pre_tanh_sample: torch.Tensor | None = None

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the latent Gaussian and invalidate the cached entropy sample."""
        if self.mean_bound is not None:
            mlp_output = mlp_output.clamp(min=-self.mean_bound, max=self.mean_bound)
        super().update(mlp_output)
        self._last_pre_tanh_sample = None

    def sample(self) -> torch.Tensor:
        """Draw a reparameterized latent sample and squash it to ``(-1, 1)``."""
        self._last_pre_tanh_sample = self._distribution.rsample()  # type: ignore[union-attr]
        return torch.tanh(self._last_pre_tanh_sample)

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Return the squashed latent mean for deterministic inference."""
        return torch.tanh(mlp_output)

    def as_deterministic_output_module(self) -> nn.Module:
        """Return an export-friendly deterministic tanh output module."""
        return nn.Tanh()

    @property
    def entropy(self) -> torch.Tensor:
        """Estimate transformed entropy using the current reparameterized sample."""
        if self._last_pre_tanh_sample is None:
            self._last_pre_tanh_sample = self._distribution.rsample()  # type: ignore[union-attr]
        return -self._log_prob_from_pre_tanh(self._last_pre_tanh_sample)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute transformed log probability with the tanh Jacobian correction."""
        bounded_outputs = outputs.clamp(min=-1.0 + self.epsilon, max=1.0 - self.epsilon)
        pre_tanh = torch.atanh(bounded_outputs)
        return self._log_prob_from_pre_tanh(pre_tanh)

    def _log_prob_from_pre_tanh(self, pre_tanh: torch.Tensor) -> torch.Tensor:
        # Stable equivalent of log(1 - tanh(x)^2), used by SAC-style policies.
        log_abs_det_jacobian = 2.0 * (
            math.log(2.0) - pre_tanh - functional.softplus(-2.0 * pre_tanh)
        )
        latent_log_prob = self._distribution.log_prob(pre_tanh)  # type: ignore[union-attr]
        return (latent_log_prob - log_abs_det_jacobian).sum(dim=-1)

