"""Chronological forecasting windows with train-only normalization."""
import gzip
import hashlib
from pathlib import Path

import numpy as np
import torch


def synthetic(kind, length=6000, seed=101, noise=0.1, dt=1.0):
    rng = np.random.default_rng(seed)
    t = np.arange(length)*dt
    if kind == 'ar':
        x = np.zeros(length)
        for i in range(2, length):
            x[i] = 0.8*x[i-1]-0.15*x[i-2]+rng.normal(0, noise)
        return x
    if kind == 'nonlinear':
        x = np.zeros(length)
        for i in range(2, length):
            x[i] = 0.7*np.tanh(x[i-1])+0.3*np.sin(x[i-2])+rng.normal(0, noise)
        return x
    if kind == 'regime':
        return np.where(t < t[-1]/2, np.sin(t/8), 0.5*np.sin(t/4))+rng.normal(0, noise, length)
    if kind == 'trend':
        clean = 0.0002*t+np.sin(t/12)+0.4*np.sin(t/4)
    elif kind == 'damped':
        clean = np.exp(-t/(length*dt))*np.cos(t/8)
    elif kind == 'periodic':
        clean = np.sin(t/8)
    elif kind == 'sine':
        clean = np.sin(t/12)+0.4*np.sin(t/5)+0.2*np.cos(t/23)
    else:
        raise ValueError(kind)
    return clean+rng.normal(0, noise, length)


REAL_FILES = {'electricity': 'electricity/electricity.txt.gz',
              'traffic': 'traffic/traffic.txt.gz', 'solar': 'solar-energy/solar_AL.txt.gz',
              'exchange': 'exchange_rate/exchange_rate.txt.gz'}


def load_real(root, name, count=3):
    path = Path(root)/REAL_FILES[name]
    with gzip.open(path, 'rt') as stream:
        matrix = np.loadtxt(stream, delimiter=',', dtype=np.float32)
    indices = np.unique(np.linspace(0, matrix.shape[1]-1, count, dtype=int))
    provenance = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'shape': list(matrix.shape), 'indices': indices.tolist(),
                  'source': 'https://github.com/laiguokun/multivariate-time-series-data'}
    return {int(i): matrix[:, i].copy() for i in indices}, provenance


def split_windows(series, history, horizon, stride=8, train_limit=2048, eval_limit=256):
    x = np.asarray(series, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite series; missing values need an explicit policy.')
    n = len(x)
    a, b = int(n*0.6), int(n*0.8)
    mean, scale = float(x[:a].mean()), max(float(x[:a].std()), 1e-6)
    normalized = (x-mean)/scale
    splits, origins = {}, {}
    for name, start, stop in [('train', history, a), ('val', a, b), ('test', b, n)]:
        # An origin is the index of the first target; all targets stay in their split.
        ids = np.arange(max(start, history), stop-horizon+1, stride)
        limit = train_limit if name == 'train' else eval_limit
        if len(ids) > limit:
            ids = ids[np.linspace(0, len(ids)-1, limit, dtype=int)]
        if not len(ids):
            raise ValueError('Split too short for requested history and horizon.')
        inputs = np.stack([normalized[i-history:i] for i in ids])
        targets = np.stack([normalized[i:i+horizon] for i in ids])
        splits[name] = (torch.from_numpy(inputs[..., None]), torch.from_numpy(targets[..., None]))
        origins[name] = ids.tolist()
    return splits, {'mean': mean, 'scale': scale, 'split_boundaries': [a, b], 'origins': origins}
