import torch
import torch.nn as nn

from models.utils import _surface_log_m_and_sqrt_t, _decode_log_price_norm

class SurfaceTransformerNet(nn.Module):
    def __init__(
            self,
            raw_input_dim: int = 6,
            d_model: int = 128,
            n_heads: int = 4,
            n_layers: int = 4,
            ff_dim: int = 256,
            head_hidden: int = 256,
            dropout: float = 0.1,
            price_log_eps: float = 1e-6,
            log_price_clip_max: float = 20.0,
    ):
        super().__init__()
        self.encoder = SurfaceEncoderTransformer(
            token_dim=raw_input_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            price_log_eps=price_log_eps,
        )
        self.price_head = SurfacePriceHead(
            raw_input_dim=raw_input_dim,
            d_model=d_model,
            head_hidden=head_hidden,
            price_log_eps=price_log_eps,
            log_price_clip_max=log_price_clip_max,
        )

    @staticmethod
    def engineer_surface_features(
            s: torch.Tensor,
            k: torch.Tensor,
            t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _surface_log_m_and_sqrt_t(s=s, k=k, t=t)

    def decode_log_price_norm(self, log_price_norm: torch.Tensor) -> torch.Tensor:
        return _decode_log_price_norm(
            log_price_norm=log_price_norm,
            price_log_eps=self.price_head.price_log_eps,
            log_price_clip_max=self.price_head.log_price_clip_max,
        )

    def forward(
            self,
            query_x: torch.Tensor,
            context_x: torch.Tensor,
            context_y: torch.Tensor,
            context_mask: torch.Tensor,
    ):
        z_date = self.encoder(context_x=context_x, context_y=context_y, context_mask=context_mask)

        with torch.enable_grad():
            query_xg = query_x.detach().requires_grad_(True)
            s = query_xg[:, :, 0:1]
            k = query_xg[:, :, 1:2]
            t = query_xg[:, :, 2:3]
            v = query_xg[:, :, 3:4]
            rest = query_xg[:, :, 4:]
            log_m, sqrt_t = self.engineer_surface_features(s=s, k=k, t=t)

            log_price_norm = self.price_head(
                log_m=log_m,
                sqrt_t=sqrt_t,
                v=v,
                rest=rest,
                z_date=z_date,
            )
            normalized_prices = self.decode_log_price_norm(log_price_norm=log_price_norm)
            prices = normalized_prices * k

            delta, dual_delta, theta, vega = torch.autograd.grad(
                outputs=prices,
                inputs=[s, k, t, v],
                grad_outputs=torch.ones_like(prices),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )

            gamma = torch.autograd.grad(
                outputs=delta,
                inputs=s,
                grad_outputs=torch.ones_like(delta),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            dual_gamma = torch.autograd.grad(
                outputs=dual_delta,
                inputs=k,
                grad_outputs=torch.ones_like(dual_delta),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

        greeks = {
            "delta": delta,
            "dual_delta": dual_delta,
            "gamma": gamma,
            "dual_gamma": dual_gamma,
            "theta": -theta,
            "vega": vega,
        }
        return normalized_prices, greeks

class SurfaceEncoderTransformer(nn.Module):
    def __init__(
            self,
            token_dim: int = 6,
            d_model: int = 128,
            n_heads: int = 4,
            n_layers: int = 4,
            ff_dim: int = 256,
            dropout: float = 0.1,
            price_log_eps: float = 1e-6,
    ):
        super().__init__()
        self.price_log_eps = price_log_eps
        self.token_projection = nn.Linear(token_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(
            self,
            context_x: torch.Tensor,
            context_y: torch.Tensor,
            context_mask: torch.Tensor,
    ) -> torch.Tensor:

        s = context_x[:, :, 0:1]
        k = context_x[:, :, 1:2]
        t = context_x[:, :, 2:3]
        v = context_x[:, :, 3:4]
        rest = context_x[:, :, 4:]

        log_m, sqrt_t = _surface_log_m_and_sqrt_t(s=s, k=k, t=t)
        context_price_norm = context_y / torch.clamp(k, min=1e-8)
        log_context_price_norm = torch.log(torch.clamp(context_price_norm, min=0.0) + self.price_log_eps)
        token_input = torch.cat([log_m, sqrt_t, v, rest, log_context_price_norm], dim=-1)

        tokens = self.token_projection(token_input)
        padding_mask = ~context_mask.bool()
        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)

        mask = context_mask.unsqueeze(-1).to(dtype=encoded.dtype)
        pooled = (encoded * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        return pooled


class SurfacePriceHead(nn.Module):
    def __init__(
            self,
            raw_input_dim: int = 6,
            d_model: int = 128,
            head_hidden: int = 256,
            price_log_eps: float = 1e-6,
            log_price_clip_max: float = 20.0,
    ):
        super().__init__()
        self.price_log_eps = price_log_eps
        self.log_price_clip_max = log_price_clip_max
        engineered_dim = 1 + 1 + 1 + (raw_input_dim - 4)
        self.model = nn.Sequential(
            nn.Linear(engineered_dim + d_model, head_hidden),
            nn.SiLU(),
            nn.Linear(head_hidden, head_hidden),
            nn.SiLU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(
            self,
            log_m: torch.Tensor,
            sqrt_t: torch.Tensor,
            v: torch.Tensor,
            rest: torch.Tensor,
            z_date: torch.Tensor,
    ) -> torch.Tensor:
        z_expanded = z_date.unsqueeze(1).expand(-1, log_m.shape[1], -1)
        model_input = torch.cat([log_m, sqrt_t, v, rest, z_expanded], dim=-1)
        return self.model(model_input)
