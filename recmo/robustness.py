"""Structured feedback interventions; sensitivity is not forecasting accuracy."""
import torch


def perturb(block, kind, magnitude=0.1):
    if kind == 'gaussian':
        # Fixed local noise keeps sensitivity evaluation reproducible across models.
        generator = torch.Generator(device=block.device).manual_seed(101)
        return block+magnitude*torch.randn(block.shape, generator=generator, device=block.device, dtype=block.dtype)
    if kind == 'bias':
        return block+magnitude
    if kind == 'scale':
        return block*(1+magnitude)
    if kind == 'trend':
        return block+magnitude*torch.linspace(-1, 1, block.shape[1], device=block.device)[None, :, None]
    if kind == 'shift':
        # Delay inside the already predicted block; no future data are used.
        return torch.cat([block[:, :1], block[:, :-1]], 1)
    raise ValueError(kind)


@torch.no_grad()
def sensitivity(model, history, magnitude=0.1):
    base = model(history)
    if model.strategy not in {'recursive', 'recmo'} or model.window >= model.horizon:
        return {}
    values = {}
    for kind in ['gaussian', 'bias', 'scale', 'trend', 'shift']:
        changed = model(history, intervention=(0, kind, magnitude))
        delta = changed[:, model.window:]-base[:, model.window:]
        values[kind] = {'later_prediction_change_mse': float(delta.square().mean()),
                        'per_lead_change_mse': delta.square().mean((0, 2)).tolist()}
    return values
