import csv,json
from collections import defaultdict
from pathlib import Path
import argparse
parser=argparse.ArgumentParser(description='Compare complete three-seed RecMO and Direct MIMO groups.')
parser.add_argument('--results', type=Path, default=Path('results/overnight_half'))
parser.add_argument('--report', type=Path, default=Path('docs/results/overnight-half'))
args=parser.parse_args()
output=args.results
status=json.loads((output/'status.json').read_text())
rows=[json.loads(p.read_text()) for p in (output/'jobs').glob('*/metrics.json')]
groups=defaultdict(dict)
for r in rows:
    key=tuple(r[k] for k in ['dataset','horizon','backbone','strategy','window','context','loss'])
    groups[key][r['seed']]=r
complete={k:v for k,v in groups.items() if set(v)=={7,17,27}}
pairs=[]
for key,values in complete.items():
    dataset,horizon,backbone,strategy,window,context,loss=key
    if strategy!='recmo':continue
    direct=(dataset,horizon,backbone,'direct',horizon,'rolling','uniform')
    if direct not in complete:continue
    a=sum(v['mse'] for v in values.values())/3
    b=sum(v['mse'] for v in complete[direct].values())/3
    pairs.append(dict(dataset=dataset,horizon=horizon,backbone=backbone,window=window,
                      recmo_mse_mean=a,direct_mse_mean=b,mse_improvement_percent=100*(b-a)/b))
destination=args.report;destination.mkdir(parents=True,exist_ok=True)
with (destination/'matched_comparisons.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(sorted(pairs,key=lambda r:(r['backbone'],r['horizon'],r['window'],r['dataset'])))
import numpy as np
lines=['# Reduced RecMO experiment: four-hour partial results','',
       f'Saved {len(rows)} of {status["planned"]} planned jobs, including 876 reused completed jobs from the earlier queue. Recorded failures: {len(status["failures"])}. Unfinished jobs: {status["planned"]-len(rows)}. The time limit was reached and no new batch was started.','',
       f'{len(complete)} model/configuration groups have all three seeds (7, 17, 27). {len(groups)-len(complete)} groups are partial. The comparisons below use only complete three-seed groups for both RecMO and Direct MIMO.','',
       'Compare the same dataset, horizon, and backbone. Each job trains for 200 updates with width 32, history 48, and batch size 32. Model selection uses validation MSE. Values below compare mean test MSE across seeds, not statistical significance. Synthetic data use known generators; real datasets use separate variables and chronological splits.','',
       '| Backbone | Horizon | Window | Matched series | RecMO wins / losses / ties | Median MSE improvement (%) |',
       '|---|---:|---:|---:|---:|---:|']
for group in sorted({(r['backbone'],r['horizon'],r['window']) for r in pairs}):
    selected=[r for r in pairs if (r['backbone'],r['horizon'],r['window'])==group]
    delta=np.array([r['mse_improvement_percent'] for r in selected])
    wins=int((delta>1e-6).sum());losses=int((delta < -1e-6).sum());ties=len(delta)-wins-losses
    lines.append(f'| {group[0]} | {group[1]} | {group[2]} | {len(delta)} | {wins} / {losses} / {ties} | {np.median(delta):.2f} |')
lines+=['','Positive improvements favor RecMO; negative improvements favor Direct MIMO. All matched results are retained in [the CSV](matched_comparisons.csv). W=H is mathematically identical to Direct MIMO and is retained as a control. Series are not statistically independent replicates; median relative improvement is descriptive and is not a pooled cross-dataset error metric.','',
        'The queue reached DA-RNN block forecasting before the budget expired. Later recursive-one-step and DirMO jobs are unfinished, so this report does not claim a full forecasting-strategy comparison. Context and learned-loss ablations were intentionally deferred in the reduced plan. GPU support was added to the repository but these results used CPU.','',
        'These small, fixed-budget experiments do not establish novelty or general superiority. Larger architecture-specific tuning and multiple data splits remain necessary.','']
(destination/'REPORT.md').write_text('\n'.join(lines))
(destination/'status.json').write_text((output/'status.json').read_text())
(destination/'config.json').write_text((output/'config.json').read_text())
print('\n'.join(lines))
