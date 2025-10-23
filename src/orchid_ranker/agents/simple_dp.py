# src/agents/simple_dp.py
from dataclasses import dataclass
from typing import Iterable, Dict, Any, List, Tuple
import math
import torch

@dataclass
class SimpleDPConfig:
    enabled: bool = True
    noise_multiplier: float = 1.0     # σ
    max_grad_norm: float = 1.0        # C
    sample_rate: float = 0.02         # q (≈ batch_fraction)
    delta: float = 1e-5               # δ

class SimpleDPAccountant:
    """
    Very small ε(δ) accountant using Abadi et al. (2016) bound:
      ε ≈ q * sqrt(2 T log(1/δ)) / σ + T * q^2 / σ^2
    where q is the sample rate, T the number of DP steps, σ the noise multiplier.
    We treat q as the configured sample_rate and compose monotonically.
    """
    def __init__(self, q: float, sigma: float, delta: float):
        self.q = float(q)
        self.sigma = float(sigma)
        self.delta = float(delta)
        self.T = 0
        self.eps = 0.0

    def _eps_for(self, T: int) -> float:
        if self.sigma <= 0.0 or self.q <= 0.0 or T <= 0:
            return 0.0
        term1 = self.q * math.sqrt(2.0 * T * math.log(1.0 / self.delta)) / self.sigma
        term2 = (T * (self.q ** 2)) / (self.sigma ** 2)
        return float(term1 + term2)

    def step(self, steps: int) -> Tuple[float, float]:
        """Advance by `steps` DP steps, return (epsilon_delta, epsilon_cum)."""
        steps = int(max(0, steps))
        if steps == 0:
            return 0.0, float(self.eps)
        eps_before = self._eps_for(self.T)
        self.T += steps
        eps_after = self._eps_for(self.T)
        self.eps = eps_after
        return float(max(0.0, eps_after - eps_before)), float(self.eps)


def dp_sgd_step(
    model: torch.nn.Module,
    params: List[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    loss_fn,
    batch_inputs: Tuple[torch.Tensor, ...],
    *,
    max_grad_norm: float,
    noise_multiplier: float,
    device: torch.device,
) -> float:
    """
    Perform one DP-SGD step by looping per-example:
      - compute grad for each sample,
      - clip to max_grad_norm,
      - sum clipped grads,
      - add Gaussian noise,
      - set p.grad := (noisy_sum / batch_size), optimizer.step().
    Returns loss (float).
    """
    model.train()
    # unpack inputs -> we expect last tensor in tuple to be labels
    *x_tensors, y = batch_inputs
    B = int(y.size(0))
    if B <= 0:
        return 0.0

    # zero regular grads
    optimizer.zero_grad(set_to_none=False)

    # storage for summed clipped grads
    sum_grads = [torch.zeros_like(p, device=device) for p in params]
    total_loss = 0.0

    for i in range(B):
        # zero grads before single-example backward
        for p in params:
            if p.grad is not None:
                p.grad.zero_()

        # slice i-th example
        xi = [xt[i:i+1] for xt in x_tensors]
        yi = y[i:i+1]

        # compute loss on single example
        loss_i = loss_fn(*xi, yi)
        total_loss += float(loss_i.detach().item())

        loss_i.backward()

        # flatten & compute global L2 norm
        per_ex_grads = [p.grad.detach() if p.grad is not None else torch.zeros_like(p, device=device) for p in params]
        sqsum = 0.0
        for g in per_ex_grads:
            sqsum += float(torch.sum(g.pow(2)))
        l2 = math.sqrt(max(1e-12, sqsum))

        # clip factor
        c = min(1.0, max_grad_norm / l2)

        # accumulate clipped grads
        for j, g in enumerate(per_ex_grads):
            sum_grads[j].add_(g * c)

    # add Gaussian noise and set averaged grads
    std = float(noise_multiplier) * float(max_grad_norm)
    for j, p in enumerate(params):
        noise = torch.normal(mean=0.0, std=std, size=p.shape, device=device)
        noisy = sum_grads[j] + noise
        p.grad = (noisy / float(B))

    optimizer.step()
    return float(total_loss / max(1, B))
