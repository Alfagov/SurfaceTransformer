import torch

def _surface_log_m_and_sqrt_t(
        s: torch.Tensor,
        k: torch.Tensor,
        t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    safe_k = torch.clamp(k, min=1e-8)
    safe_t = torch.clamp(t, min=1e-8)
    log_m = torch.log(torch.clamp(s / safe_k, min=1e-8))
    sqrt_t = torch.sqrt(safe_t)
    return log_m, sqrt_t


def _decode_log_price_norm(
        log_price_norm: torch.Tensor,
        price_log_eps: float,
        log_price_clip_max: float,
) -> torch.Tensor:
    clipped = torch.clamp(log_price_norm, max=log_price_clip_max)
    return torch.clamp_min(torch.exp(clipped) - price_log_eps, 0.0)

def _build_surface_example_input(
        raw_input_dim: int,
        context_size: int,
        query_size: int,
        batch_size: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    query_x = torch.zeros(batch_size, query_size, raw_input_dim, dtype=torch.float32)
    context_x = torch.zeros(batch_size, context_size, raw_input_dim, dtype=torch.float32)
    context_y = torch.full((batch_size, context_size, 1), 10.0, dtype=torch.float32)
    context_mask = torch.ones(batch_size, context_size, dtype=torch.bool)

    for tensor in (query_x, context_x):
        tensor[:, :, 0] = 100.0
        tensor[:, :, 1] = 100.0
        tensor[:, :, 2] = 30.0 / 365.0
        tensor[:, :, 3] = 0.2
        if raw_input_dim > 4:
            tensor[:, :, 4:] = 0.01

    return query_x, context_x, context_y, context_mask


