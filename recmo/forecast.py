"""Shared-parameter causal recursive rollouts; no teacher forcing or detachment."""
import math
import torch
from torch import nn
from recmo.models import BlockPredictor


def next_context(context, block, history, mode):
    joined = torch.cat([context, block], 1)
    return joined[:, -history:] if mode == 'rolling' else joined


class Forecaster(nn.Module):
    def __init__(self, backbone, history, horizon, strategy='recmo', window=12,
                 context='rolling', width=32):
        super().__init__()
        if context not in {'rolling', 'expanding', 'separate'}:
            raise ValueError(context)
        if min(history, horizon, window) < 1:
            raise ValueError('Lengths must be positive.')
        self.history, self.horizon, self.context, self.strategy = history, horizon, context, strategy
        self.window = horizon if strategy == 'direct' else 1 if strategy == 'recursive' else window
        if strategy not in {'direct', 'recursive', 'recmo', 'dirmo'}:
            raise ValueError(strategy)
        count = math.ceil(horizon/self.window) if strategy == 'dirmo' else 1
        self.models = nn.ModuleList([BlockPredictor(backbone, history, self.window, width,
                                                   separate=context == 'separate') for _ in range(count)])

    def forward(self, history, return_blocks=False, intervention=None):
        context, blocks, feedback = history, [], []
        for i, start in enumerate(range(0, self.horizon, self.window)):
            model = self.models[i] if self.strategy == 'dirmo' else self.models[0]
            if self.strategy == 'dirmo':
                block = model(history)
            elif self.context == 'separate':
                previous = torch.cat(feedback, 1) if feedback else None
                block = model(history, previous)
            else:
                block = model(context)
            block = block[:, :min(self.window, self.horizon-start)]
            blocks.append(block)
            fed = block
            if intervention is not None and i == intervention[0]:
                from recmo.robustness import perturb
                fed = perturb(block, intervention[1], intervention[2])
            feedback.append(fed)
            if self.strategy != 'dirmo':
                context = next_context(context, fed, self.history, self.context)
        forecast = torch.cat(blocks, 1)
        return (forecast, blocks) if return_blocks else forecast


class WindowLoss(nn.Module):
    def __init__(self, count, mode='uniform', alpha=0.1):
        super().__init__()
        if mode not in {'uniform', 'increasing', 'decreasing', 'softmax', 'uncertainty'}:
            raise ValueError(mode)
        self.mode = mode
        self.values = nn.Parameter(torch.zeros(count), requires_grad=mode in {'softmax', 'uncertainty'})
        self.register_buffer('fixed', torch.arange(count).float()*alpha)

    def weights(self):
        if self.mode == 'uncertainty':
            return 0.5*torch.exp(-self.values.clamp(-10, 10))
        if self.mode == 'softmax':
            # Ten percent uniform mass prevents a window losing all importance.
            return 0.9*self.values.softmax(0)+0.1/len(self.values)
        if self.mode == 'uniform':
            return torch.ones_like(self.values)/len(self.values)
        return (self.fixed if self.mode == 'increasing' else -self.fixed).softmax(0)

    def forward(self, blocks, target):
        losses, offset = [], 0
        for block in blocks:
            losses.append((block-target[:, offset:offset+block.shape[1]]).square().mean())
            offset += block.shape[1]
        losses = torch.stack(losses)
        weights = self.weights()
        if self.mode == 'uncertainty':
            return (weights*losses+0.5*self.values.clamp(-10, 10)).sum()
        regularizer = 0.01*(weights*torch.log(weights*len(weights))).sum() if self.mode == 'softmax' else 0
        return (weights*losses).sum()+regularizer


def load_checkpoint(path, device='cpu'):
    """Restore a predictor on any available device; inputs use saved normalization."""
    from recmo.devices import resolve_device
    saved = torch.load(path, map_location='cpu', weights_only=True)
    job, settings = saved['job'], saved['settings']
    model = Forecaster(job['backbone'], settings['history'], job['horizon'],
                       job['strategy'], job['window'], job['context'], settings.get('width', 32))
    model.load_state_dict(saved['model'])
    model.to(resolve_device(device)).eval()
    return model, saved
