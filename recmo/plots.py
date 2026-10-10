"""Comparable numerical summaries and scientific figures from completed jobs."""
import csv
import json
import os
from pathlib import Path

import numpy as np


def summarize_results(root):
    rows = [json.loads(p.read_text()) for p in sorted((root/'jobs').glob('*/metrics.json'))]
    if not rows:
        return
    columns = [key for key, value in rows[0].items() if not isinstance(value, list)]
    with (root/'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    keys = ['dataset', 'horizon', 'backbone', 'strategy', 'window', 'context', 'loss']
    groups = {}
    for row in rows:
        group = tuple(row[k] for k in keys)
        groups.setdefault(group, []).append(row)
    summary = []
    for group, values in groups.items():
        summary.append({**dict(zip(keys, group)), 'seeds_completed': len(values),
                        'rmse_mean': float(np.mean([v['rmse'] for v in values])),
                        'rmse_std': float(np.std([v['rmse'] for v in values], ddof=1)) if len(values)>1 else None})
    (root/'summary.json').write_text(json.dumps(summary, indent=2))
    os.environ.setdefault('MPLCONFIGDIR', str(root/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = root/'figures'
    figures.mkdir(exist_ok=True)
    for dataset, horizon in sorted({(r['dataset'], r['horizon']) for r in rows}):
        selected = [r for r in rows if r['dataset']==dataset and r['horizon']==horizon and r['seed']==min(v['seed'] for v in rows)]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
        first = np.load(root/'jobs'/selected[0]['id']/'predictions.npz')
        axes[1].plot(np.arange(1, horizon+1), first['target'][0, :, 0], color='black', linewidth=2, label='Truth')
        for row in selected:
            label = f'{row["backbone"]}/{row["strategy"]}/W{row["window"]}/{row["context"]}/{row["loss"]}'
            axes[0].plot(np.arange(1, horizon+1), np.sqrt(row['lead_mse']), label=label)
            pred = np.load(root/'jobs'/row['id']/'predictions.npz')['prediction'][0, :, 0]
            axes[1].plot(np.arange(1, horizon+1), pred, linewidth=1)
        axes[0].set(xlabel='Forecast lead', ylabel='RMSE (original units)', title=f'{dataset}, H={horizon}')
        axes[1].set(xlabel='Forecast lead', ylabel='Value', title='Predetermined first held-out origin')
        axes[0].legend(fontsize=5)
        fig.savefig(figures/f'{dataset}_H{horizon}.png', dpi=150)
        plt.close(fig)
    (root/'REPORT.md').write_text('# RecMO experiment results\n\n'+
        f'{len(rows)} completed jobs; see metrics.csv, summary.json, and figures/.\n\n'+
        'All results are retained. Compare methods within the same dataset and horizon. '+
        'Metrics use original units; normalized MSE is provided separately. '+
        'Seed standard deviations measure initialization variability on a fixed chronological split. '+
        'Partial queues and unequal completed seed counts do not support a superiority claim.\n')
