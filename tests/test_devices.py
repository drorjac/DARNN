"""Device selection, portable artifacts, and conditional accelerator integration."""
import json
import time

import pytest
import torch

from recmo.data import synthetic, split_windows
from recmo.devices import cpu_state, resolve_device, synchronize
from recmo.experiment import neural_fit, run_job, make_plan
from recmo.forecast import Forecaster, WindowLoss, load_checkpoint
from recmo.robustness import sensitivity


def test_auto_selection_and_explicit_errors(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    assert resolve_device('auto') == torch.device('cpu')
    with pytest.raises(ValueError, match='CUDA unavailable'):
        resolve_device('cuda')
    with pytest.raises(ValueError, match='MPS unavailable'):
        resolve_device('mps')
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    assert resolve_device('auto').type == 'mps'
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    assert resolve_device('auto').type == 'cuda'
    assert str(resolve_device('cuda:1')) == 'cuda:1'
    with pytest.raises(ValueError, match='does not exist'):
        resolve_device('cuda:2')


@pytest.mark.parametrize('device', ['cpu', 'cuda', 'mps'])
@pytest.mark.parametrize('kind', ['mlp', 'lstm', 'gru', 'darnn'])
def test_training_feedback_and_portable_state(device, kind):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA hardware unavailable')
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS hardware unavailable')
    chosen = resolve_device(device)
    splits, _ = split_windows(synthetic('sine', 200), 6, 11, train_limit=8, eval_limit=4)
    splits = {name: tuple(t.to(chosen) for t in pair) for name, pair in splits.items()}
    model = Forecaster(kind, 6, 11, window=4, context='separate', width=4).to(chosen)
    loss = WindowLoss(3, 'softmax').to(chosen)
    neural_fit(model, loss, splits, {'updates': 1, 'batch_size': 2}, 7, time.monotonic()+60)
    x = splits['test'][0]
    stats = sensitivity(model, x[:2])
    assert 'gaussian' in stats
    state = cpu_state(model)
    assert all(t.device.type == 'cpu' for t in state.values())
    clone = Forecaster(kind, 6, 11, window=4, context='separate', width=4)
    clone.load_state_dict(state)
    model.eval()
    clone.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(x).cpu(), clone(x.cpu()), rtol=1e-3, atol=1e-4)
    synchronize(chosen)


def test_job_artifacts_record_execution_and_restore_on_cpu(tmp_path):
    job = dict(dataset='synthetic_sine', seed=7, horizon=11, backbone='gru',
               strategy='recmo', window=4, context='rolling', loss='uniform')
    settings = dict(history=6, updates=1, batch_size=2, width=4, train_limit=8, eval_limit=4)
    run_job(job, synthetic('sine',200), settings, tmp_path, time.monotonic()+60, torch.device('cpu'))
    folder = next((tmp_path/'jobs').iterdir())
    metrics = json.loads((folder/'metrics.json').read_text())
    assert metrics['device'] == 'cpu'
    assert metrics['inference_ms_per_origin'] >= 0
    clone, loaded = load_checkpoint(folder/'checkpoint.pt', 'cpu')
    assert loaded['job'] == job
    assert next(clone.parameters()).device.type == 'cpu'
    assert clone(torch.zeros(2,6,1)).shape == (2,11,1)


def test_disjoint_shards_cover_exact_plan():
    config = dict(backbones=['lstm'], windows=[4], seeds=[7,17], horizons=[12,24])
    plan = make_plan(config, ['a','b'])
    shards = [plan[i::3] for i in range(3)]
    serialize = lambda jobs: {json.dumps(j,sort_keys=True) for j in jobs}
    assert serialize(plan) == set.union(*(serialize(s) for s in shards))
    assert sum(len(s) for s in shards) == len(plan)
    assert not serialize(shards[0]) & serialize(shards[1])
