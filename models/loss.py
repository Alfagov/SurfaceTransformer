from typing import Tuple, Dict

import torch
import torch.nn as nn

class GreeksInformedLoss(nn.Module):
    def __init__(
            self,
            lambda_arb = 1.0,
            theta_floor_base: float = -0.03,
            theta_floor_slope: float = 0.0393,
            theta_floor_eps: float = 1e-4,
            delta_ceiling: float = 0.9999,
            price_spot_margin: float = 1e-4,
            w_delta: float = 1.0,
            w_delta_upper: float = 1.0,
            w_dual_delta: float = 1.0,
            w_gamma: float = 1.0,
            w_theta: float = 1.0,
            w_theta_upper: float = 1.0,
            w_dual_gamma: float = 1.0,
            w_price_upper: float = 1.0,
            vega_weight_eps: float = 1e-8,
            huber_delta: float = 1.0,
    ):
        super().__init__()

        self.price_loss = nn.HuberLoss(reduction="none", delta=huber_delta)

        self.lambda_arb = lambda_arb

        self.theta_floor_base = theta_floor_base
        self.theta_floor_slope = theta_floor_slope
        self.theta_floor_eps = theta_floor_eps
        self.delta_ceiling = delta_ceiling
        self.price_spot_margin = price_spot_margin

        self.w_delta = w_delta
        self.w_delta_upper = w_delta_upper
        self.w_dual_delta = w_dual_delta
        self.w_gamma = w_gamma
        self.w_theta = w_theta
        self.w_theta_upper = w_theta_upper
        self.w_dual_gamma = w_dual_gamma
        self.w_price_upper = w_price_upper
        self.vega_weight_eps = vega_weight_eps

    @staticmethod
    def _flatten_column(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, 1)

    @staticmethod
    def _sanitize_tensor(x: torch.Tensor, clamp: float = 1e6) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=clamp, neginf=-clamp)
        return torch.clamp(x, min=-clamp, max=clamp)

    @staticmethod
    def _build_loss_mask(reference: torch.Tensor, sample_mask: torch.Tensor | None) -> torch.Tensor:
        if sample_mask is None:
            return torch.ones_like(reference)

        mask = sample_mask.reshape(-1, 1).to(dtype=reference.dtype, device=reference.device)
        if mask.shape != reference.shape:
            raise ValueError(
                f"Mask shape {tuple(mask.shape)} must match flattened tensor shape {tuple(reference.shape)}."
            )
        return mask

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = torch.clamp(mask.sum(), min=1.0)
        return (values * mask).sum() / denom

    def forward(
            self,
            y_pred: Tuple[torch.Tensor, Dict[str, torch.Tensor]],
            y_true: torch.Tensor,
            t: torch.Tensor,
            s: torch.Tensor,
            k: torch.Tensor,
            sample_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:

        pred, greeks = y_pred
        pred = self._sanitize_tensor(self._flatten_column(pred))
        y_true = self._sanitize_tensor(self._flatten_column(y_true))
        t = self._sanitize_tensor(self._flatten_column(t))
        s = self._sanitize_tensor(self._flatten_column(s))
        k = self._sanitize_tensor(self._flatten_column(k))
        loss_mask = self._build_loss_mask(pred, sample_mask)

        delta = self._sanitize_tensor(self._flatten_column(greeks["delta"]))
        dual_delta = self._sanitize_tensor(self._flatten_column(greeks["dual_delta"]))
        gamma = self._sanitize_tensor(self._flatten_column(greeks["gamma"]))
        theta = self._sanitize_tensor(self._flatten_column(greeks["theta"]))
        dual_gamma = self._sanitize_tensor(self._flatten_column(greeks["dual_gamma"]))

        call_prices = pred * k

        price_loss = self._masked_mean(self.price_loss(call_prices, y_true), loss_mask)

        # Normalize theta to K
        theta_normalized = theta / torch.clamp(k, min=1e-8)

        # Calculate the theta floor based on Time-to-Maturity
        theta_floor = self.theta_floor_base - self.theta_floor_slope / torch.sqrt(
            torch.clamp(t, min=self.theta_floor_eps)
        )

        # Allow only negative floors this is for Call options
        theta_floor = torch.minimum(theta_floor, torch.zeros_like(theta_floor))

        delta_loss = self._masked_mean(torch.relu(-delta).pow(2), loss_mask)
        delta_upper_loss = self._masked_mean(torch.relu(delta - self.delta_ceiling).pow(2), loss_mask)
        dual_delta_loss = self._masked_mean(torch.relu(dual_delta).pow(2), loss_mask)
        gamma_loss = self._masked_mean(torch.relu(-gamma).pow(2), loss_mask)
        theta_loss = self._masked_mean(torch.relu(theta_floor - theta_normalized).pow(2), loss_mask)
        theta_upper_loss = self._masked_mean(torch.relu(theta).pow(2), loss_mask)
        dual_gamma_loss = self._masked_mean(torch.relu(-dual_gamma).pow(2), loss_mask)

        # Call cannot cost more than the underlying asset
        price_upper_loss = self._masked_mean(
            torch.relu(call_prices - (s - self.price_spot_margin)).pow(2),
            loss_mask,
        )

        greek_penalty = (
            self.w_delta * delta_loss
            + self.w_delta_upper * delta_upper_loss
            + self.w_dual_delta * dual_delta_loss
            + self.w_gamma * gamma_loss
            + self.w_theta * theta_loss
            + self.w_theta_upper * theta_upper_loss
            + self.w_dual_gamma * dual_gamma_loss
            + self.w_price_upper * price_upper_loss
        )

        total_loss = price_loss + self.lambda_arb * greek_penalty

        return {
            "loss": total_loss,
            "price_loss": price_loss,
            "greek_penalty": greek_penalty,
            "delta_loss": delta_loss,
            "delta_upper_loss": delta_upper_loss,
            "gamma_loss": gamma_loss,
            "theta_loss": theta_loss,
            "theta_upper_loss": theta_upper_loss,
            "dual_delta_loss": dual_delta_loss,
            "dual_gamma_loss": dual_gamma_loss,
            "price_upper_loss": price_upper_loss,
            "theta_floor_mean": self._masked_mean(theta_floor, loss_mask),
        }
