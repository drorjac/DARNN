"""Resumable comparisons with validation-only selection and incremental artifacts."""
import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from recmo.data import load_real, synthetic, split_windows
from recmo.forecast import Forecaster, WindowLoss


def neural_fit(model, loss_fn, splits, settings, seed, deadline):
    optimizer = torch.optim.AdamW(list(model.parameters())+list(loss_fn.parameters()),
                                  lr=settings.get('lr', 0.001), weight_decay=1e-4)
    x, y = splits['train']
    generator = torch.Generator().manual_seed(seed)
    best, best_score, logs = copy.deepcopy(model.state_dict()), float('inf'), []
    best_weights = copy.deepcopy(loss_fn.state_dict())
    for step in range(settings['updates']+1):
        if time.monotonic() > deadline:
            raise TimeoutError('Run time budget reached; unfinished job will be retried on resume.')
        if step:
            model.train()
            ids = torch.randint(len(x), (settings.get('batch_size', 32),), generator=generator)
            optimizer.zero_grad(set_to_none=True)
            prediction, blocks = model(x[ids], return_blocks=True)
            # Primary uniform objective gives each time step equal weight, including partial windows.
            loss = (prediction-y[ids]).square().mean() if loss_fn.mode == 'uniform' else loss_fn(blocks, y[ids])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss.')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
            optimizer.step()
        if step % settings.get('validate_every', 20) == 0 or step == settings['updates']:
            model.eval()
            with torch.no_grad():
                vx, vy = splits['val']
                errors = []
                for start in range(0, len(vx), 32):
                    errors.append((model(vx[start:start+32])-vy[start:start+32]).square().mean().item())
                # Batch sizes vary; use exact sample weights for the final batch.
                sizes = [min(32, len(vx)-start) for start in range(0, len(vx), 32)]
                score = float(np.average(errors, weights=sizes))
            logs.append({'step': step, 'validation_mse': score, 'weights': loss_fn.weights().detach().tolist()})
            if not math.isfinite(score):
                raise RuntimeError('Nonfinite validation error.')
            if score < best_score:
                best_score, best = score, copy.deepcopy(model.state_dict())
                best_weights = copy.deepcopy(loss_fn.state_dict())
    model.load_state_dict(best)
    loss_fn.load_state_dict(best_weights)
    model.eval()
    return logs


def baseline_predictions(name, splits):
    x, y = splits['train']
    vx, vy = splits['val']
    tx, ty = splits['test']
    horizon = ty.shape[1]
    if name == 'persistence':
        return tx[:, -1:].expand(-1, horizon, -1), 0
    design = torch.cat([x.flatten(1), torch.ones(len(x), 1)], 1)
    valid = torch.cat([vx.flatten(1), torch.ones(len(vx), 1)], 1)
    test = torch.cat([tx.flatten(1), torch.ones(len(tx), 1)], 1)
    candidates = []
    for alpha in [0.01, 0.1, 1, 10, 100]:
        penalty = torch.eye(design.shape[1])*alpha
        penalty[-1, -1] = 0
        weights = torch.linalg.solve(design.T@design+penalty, design.T@y.squeeze(-1))
        candidates.append((float((valid@weights-vy.squeeze(-1)).square().mean()), weights))
    weights = min(candidates, key=lambda item: item[0])[1]
    return (test@weights).unsqueeze(-1), weights.numel()


def run_job(job, series, settings, root, deadline):
    torch.manual_seed(job['seed'])
    splits, metadata = split_windows(series, settings['history'], job['horizon'],
                                     settings.get('stride', 12), settings.get('train_limit', 2048),
                                     settings.get('eval_limit', 128))
    identity = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()[:16]
    folder = root/'jobs'/identity
    if (folder/'metrics.json').exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    logs = []
    if job['backbone'] in {'persistence', 'ridge'}:
        prediction, params = baseline_predictions(job['backbone'], splits)
        training_seconds = time.perf_counter()-started
        inference_ms = None  # Closed form fit+prediction not conflated with neural timing.
    else:
        model = Forecaster(job['backbone'], settings['history'], job['horizon'], job['strategy'],
                           job['window'], job['context'], settings.get('width', 32))
        loss_fn = WindowLoss(math.ceil(job['horizon']/model.window), job['loss'])
        logs = neural_fit(model, loss_fn, splits, settings, job['seed'], deadline)
        training_seconds = time.perf_counter()-started
        params = sum(p.numel() for p in model.parameters())
        tx = splits['test'][0]
        with torch.no_grad():
            model(tx[:1])
            started = time.perf_counter()
            prediction = torch.cat([model(tx[i:i+32]) for i in range(0, len(tx), 32)])
            inference_ms = (time.perf_counter()-started)*1000/len(tx)
        from recmo.robustness import sensitivity
        (folder/'sensitivity.json').write_text(json.dumps(sensitivity(model, tx[:16]), indent=2))
        torch.save({'model': model.state_dict(), 'loss': loss_fn.state_dict(),
                    'job': job, 'settings': settings, 'normalization': metadata}, folder/'checkpoint.pt')
    target = splits['test'][1]
    if not torch.isfinite(prediction).all():
        raise RuntimeError('Nonfinite test prediction.')
    error = (prediction-target).numpy()*metadata['scale']
    lead = np.mean(error**2, axis=(0, 2))
    window = job['window']
    metrics = {**job, 'mse': float(np.mean(error**2)), 'rmse': float(np.sqrt(np.mean(error**2))),
               'mae': float(np.mean(np.abs(error))), 'normalized_mse': float((prediction-target).square().mean()),
               'lead_mse': lead.tolist(), 'window_mse': [float(np.mean(lead[i:i+window])) for i in range(0, len(lead), window)],
               'parameters': params, 'training_seconds': training_seconds, 'inference_ms_per_origin': inference_ms,
               'iterations': 1 if job['strategy']=='direct' else math.ceil(job['horizon']/window),
               'test_origins': len(target), 'id': identity}
    (folder/'data.json').write_text(json.dumps(metadata, indent=2))
    (folder/'training.json').write_text(json.dumps(logs, indent=2))
    np.savez_compressed(folder/'predictions.npz', prediction=prediction.numpy()*metadata['scale']+metadata['mean'],
                        target=target.numpy()*metadata['scale']+metadata['mean'],
                        history=splits['test'][0].numpy()*metadata['scale']+metadata['mean'])
    (folder/'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(f'{identity} {job}: RMSE={metrics["rmse"]:.4g} ({training_seconds:.1f}s)', flush=True)


def summarize(root):
    from recmo.plots import summarize_results
    summarize_results(root)


def make_plan(config, names):
    methods = [('persistence', 'direct', None, 'rolling', 'uniform'),
               ('ridge', 'direct', None, 'rolling', 'uniform')]
    for backbone in config['backbones']:
        methods.append((backbone, 'direct', None, 'rolling', 'uniform'))
        for window in config['windows']:
            methods.append((backbone, 'recmo', window, 'rolling', 'uniform'))
    # Put the expensive recursive and extension ablations after initial breadth coverage.
    for backbone in config['backbones']:
        methods.extend([(backbone, 'recursive', 1, 'rolling', 'uniform'),
                        (backbone, 'dirmo', config['windows'][0], 'rolling', 'uniform')])
        if config.get('contexts', False):
            methods.extend([(backbone, 'recmo', config['windows'][0], c, 'uniform') for c in ['expanding', 'separate']])
        if config.get('weighted_losses', False):
            methods.extend([(backbone, 'recmo', config['windows'][0], 'rolling', loss)
                            for loss in ['increasing', 'decreasing', 'softmax', 'uncertainty']])
    plan = []
    for backbone, strategy, window, context, loss in methods:
        for seed in config['seeds']:
            for horizon in config['horizons']:
                for name in names:
                    plan.append(dict(dataset=name, seed=seed, horizon=horizon, backbone=backbone,
                                     strategy=strategy, window=horizon if window is None else window,
                                     context=context, loss=loss))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if args.summarize_only:
        summarize(args.output)
        return
    torch.set_num_threads(1)
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    saved_config = args.output/'config.json'
    if saved_config.exists() and json.loads(saved_config.read_text()) != config:
        raise ValueError('Use a new output directory for a changed configuration.')
    saved_config.write_text(json.dumps(config, indent=2))
    deadline = time.monotonic()+config.get('max_hours', 6)*3600
    series, provenance = {}, {}
    for kind in config['synthetic']:
        series['synthetic_'+kind] = synthetic(kind, config.get('synthetic_length', 6000))
    for name in config['real']:
        selected, info = load_real(config['data_root'], name, config.get('real_variables', 3))
        provenance[name] = info
        for index, values in selected.items():
            series[f'{name}_{index}'] = values
    provenance_path = args.output/'provenance.json'
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
        raise ValueError('Source data changed; use a fresh output directory.')
    provenance_path.write_text(json.dumps(provenance, indent=2))
    plan = make_plan(config, list(series))
    (args.output/'plan.json').write_text(json.dumps(plan, indent=2))
    failures = []
    (args.output/'status.json').write_text(json.dumps({'running': True, 'pid': os.getpid(), 'planned': len(plan)}, indent=2))
    for job in plan:
        if time.monotonic() >= deadline:
            break
        try:
            run_job(job, series[job['dataset']], config, args.output, deadline)
            completed = len(list((args.output/'jobs').glob('*/metrics.json')))
            (args.output/'status.json').write_text(json.dumps({'running': True, 'pid': os.getpid(), 'completed': completed, 'planned': len(plan), 'failures': len(failures)}, indent=2))
        except TimeoutError:
            break
        except Exception as error:
            failures.append({'job': job, 'error': repr(error)})
            (args.output/'failures.json').write_text(json.dumps(failures, indent=2))
            print(f'FAILED {job}: {error!r}', flush=True)
    summarize(args.output)
    completed = len(list((args.output/'jobs').glob('*/metrics.json')))
    (args.output/'status.json').write_text(json.dumps({'completed': completed, 'planned': len(plan),
                                                     'failures': failures, 'running': False, 'complete': completed == len(plan)}, indent=2))
    print(f'Finished this session: {completed}/{len(plan)} jobs. Resume with the same command.', flush=True)


if __name__ == '__main__':
    main()
