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
            delta_ceiling: float = 1.0,
            price_spot_margin: float = 1e-4,
            w_delta: float = 1.0,
            w_delta_upper: float = 1.0,
            w_dual_delta: float = 1.0,
            w_gamma: float = 1.0,
            w_theta: float = 1.0,
            w_theta_upper: float = 1.0,
            w_dual_gamma: float = 1.0,
            w_price_upper: float = 1.0,
            w_delta_target: float = 1.0,
            w_gamma_target: float = 1.0,
            w_theta_target: float = 1.0,
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
        self.w_delta_target = w_delta_target
        self.w_gamma_target = w_gamma_target
        self.w_theta_target = w_theta_target
        self.vega_weight_eps = vega_weight_eps

    @staticmethod
    def _validate_and_flatten_target_greeks(
            target_greeks: Dict[str, torch.Tensor],
            reference: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        required_keys = ("delta", "gamma", "theta")
        missing_keys = [key for key in required_keys if key not in target_greeks]
        if missing_keys:
            raise ValueError(
                f"target_greeks is missing required keys: {missing_keys}. "
                f"Expected keys: {list(required_keys)}."
            )

        flattened: Dict[str, torch.Tensor] = {}
        for key in required_keys:
            value = target_greeks[key]
            if not torch.is_tensor(value):
                raise TypeError(f"target_greeks['{key}'] must be a torch.Tensor, got {type(value).__name__}.")

            value_flat = value.reshape(-1, 1)
            if value_flat.shape != reference.shape:
                raise ValueError(
                    f"target_greeks['{key}'] has flattened shape {tuple(value_flat.shape)}, "
                    f"but expected {tuple(reference.shape)}."
                )
            flattened[key] = value_flat

        return flattened

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
            target_greeks: Dict[str, torch.Tensor],
            sample_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:

        pred, greeks = y_pred
        pred = self._sanitize_tensor(self._flatten_column(pred))
        y_true = self._sanitize_tensor(self._flatten_column(y_true))
        t = self._sanitize_tensor(self._flatten_column(t))
        s = self._sanitize_tensor(self._flatten_column(s))
        k = self._sanitize_tensor(self._flatten_column(k))
        loss_mask = self._build_loss_mask(pred, sample_mask)

        flattened_target_greeks = self._validate_and_flatten_target_greeks(target_greeks, reference=pred)

        delta = self._sanitize_tensor(self._flatten_column(greeks["delta"]))
        dual_delta = self._sanitize_tensor(self._flatten_column(greeks["dual_delta"]))
        gamma = self._sanitize_tensor(self._flatten_column(greeks["gamma"]))
        theta = self._sanitize_tensor(self._flatten_column(greeks["theta"]))
        dual_gamma = self._sanitize_tensor(self._flatten_column(greeks["dual_gamma"]))
        delta_target = self._sanitize_tensor(flattened_target_greeks["delta"])
        gamma_target = self._sanitize_tensor(flattened_target_greeks["gamma"])
        theta_target = self._sanitize_tensor(flattened_target_greeks["theta"])

        call_prices = pred * k

        weight = 1.0 / (torch.abs(delta.detach()) + self.theta_floor_eps)
        weight = torch.clamp(weight, max=10.0)
        price_loss = self._masked_mean(torch.abs(call_prices - y_true) * weight, loss_mask)

        # Normalize theta to K
        theta_normalized = theta / torch.clamp(k, min=1e-8)
        theta_target_normalized = theta_target / torch.clamp(k, min=1e-8)

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

        delta_target_loss = self._masked_mean((delta - delta_target).pow(2), loss_mask)
        gamma_target_loss = self._masked_mean((gamma - gamma_target).pow(2), loss_mask)
        theta_target_loss = self._masked_mean((theta_normalized - theta_target_normalized).pow(2), loss_mask)

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
            + self.w_delta_target * delta_target_loss
            + self.w_gamma_target * gamma_target_loss
            + self.w_theta_target * theta_target_loss
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
            "delta_target_loss": delta_target_loss,
            "gamma_target_loss": gamma_target_loss,
            "theta_target_loss": theta_target_loss,
            "theta_floor_mean": self._masked_mean(theta_floor, loss_mask),
        }
