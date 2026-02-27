import torch
import torch.nn as nn
from rational.torch import Rational
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

        query_feat_dim = 1 + 1 + 1 + (raw_input_dim - 4)
        self.cross_attn = SurfaceCrossAttentionBlock(
            query_dim=query_feat_dim,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )

        # self.price_head = SurfacePriceHead(
        #     raw_input_dim=raw_input_dim,
        #     d_model=d_model,
        #     head_hidden=head_hidden,
        #     price_log_eps=price_log_eps,
        #     log_price_clip_max=log_price_clip_max,
        # )

        self.price_head = SurfaceMoEMoneynessHead(
            raw_input_dim=raw_input_dim,
            d_model=d_model,
            head_hidden=head_hidden,
            num_experts = 3,
            price_log_eps=price_log_eps,
            log_price_clip_max=log_price_clip_max,
            gate_temperature=1.0,
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
        context_memory = self.encoder(context_x=context_x, context_y=context_y, context_mask=context_mask)

        with torch.enable_grad():
            query_xg = query_x.detach().requires_grad_(True)
            s = query_xg[:, :, 0:1]
            k = query_xg[:, :, 1:2]
            t = query_xg[:, :, 2:3]
            v = query_xg[:, :, 3:4]
            rest = query_xg[:, :, 4:]

            log_m, sqrt_t = self.engineer_surface_features(s=s, k=k, t=t)

            query_feats = torch.cat([log_m, sqrt_t, v, rest], dim=-1)
            context_features = self.cross_attn(
                query_features=query_feats,
                context_memory=context_memory,
                context_mask=context_mask
            )

            log_price_norm = self.price_head(
                log_m=log_m,
                sqrt_t=sqrt_t,
                v=v,
                rest=rest,
                context_features=context_features,
            )
            normalized_prices = self.decode_log_price_norm(log_price_norm=log_price_norm)
            prices = normalized_prices * k

            delta, dual_delta, theta = torch.autograd.grad(
                outputs=prices,
                inputs=[s, k, t],
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
        }
        return normalized_prices, greeks

class SurfaceCrossAttentionBlock(nn.Module):
    def __init__(self,
                 query_dim: int,
                 d_model: int,
                 n_heads: int,
                 dropout: float = 0.1
                 ):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, d_model)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4 , d_model),
            nn.Dropout(dropout)
        )

    def forward(self,
                query_features: torch.Tensor,
                context_memory: torch.Tensor,
                context_mask: torch.Tensor
                ) -> torch.Tensor:
        q = self.query_proj(query_features)

        key_padding_mask = ~context_mask.bool()


        attn_out, _ = self.mha(
            query=q,
            key=context_memory,
            value=context_memory,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )

        x = self.norm1(q + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x

class SurfaceEncoderTransformer(nn.Module):
    def __init__(
            self,
            token_dim: int = 7,
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
        return encoded


class SurfacePriceHead(nn.Module):
    def __init__(
            self,
            raw_input_dim: int = 7,
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
            nn.Softplus(),
            nn.Linear(head_hidden, head_hidden),
            nn.Softplus(),
            nn.Linear(head_hidden, 1),
        )

    def forward(
            self,
            log_m: torch.Tensor,
            sqrt_t: torch.Tensor,
            v: torch.Tensor,
            rest: torch.Tensor,
            context_features: torch.Tensor,
    ) -> torch.Tensor:
        model_input = torch.cat([log_m, sqrt_t, v, rest, context_features], dim=-1)
        return self.model(model_input)

class SurfaceMoEMoneynessHead(nn.Module):
    def __init__(
            self,
            raw_input_dim: int = 7,
            d_model: int = 128,
            head_hidden: int = 256,
            num_experts: int = 3,  # e.g., ITM, ATM, OTM
            price_log_eps: float = 1e-6,
            log_price_clip_max: float = 20.0,
            gate_temperature: float = 1.0,
    ):
        super().__init__()
        self.price_log_eps = price_log_eps
        self.log_price_clip_max = log_price_clip_max
        self.num_experts = num_experts
        self.gate_temperature = gate_temperature

        engineered_dim = 1 + 1 + 1 + (raw_input_dim - 4)
        input_dim = engineered_dim + d_model

        self.gate = nn.Sequential(
            nn.Linear(1, head_hidden // 2),
            nn.Tanh(),
            nn.Linear(head_hidden // 2, num_experts)
        )

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, head_hidden),
                nn.SiLU(),
                nn.Linear(head_hidden, head_hidden),
                nn.SiLU(),
                nn.Linear(head_hidden, 1),
            ) for _ in range(num_experts)
        ])

    def forward(
            self,
            log_m: torch.Tensor,
            sqrt_t: torch.Tensor,
            v: torch.Tensor,
            rest: torch.Tensor,
            context_features: torch.Tensor,
    ) -> torch.Tensor:

        gate_logits = self.gate(log_m)
        gate_weights = nn.functional.softmax(gate_logits / self.gate_temperature, dim=-1)

        model_input = torch.cat([log_m, sqrt_t, v, rest, context_features], dim=-1)
        expert_outputs = torch.cat([
            expert(model_input) for expert in self.experts
        ], dim=-1)

        blended_output = torch.sum(gate_weights * expert_outputs, dim=-1, keepdim=True)

        return blended_output