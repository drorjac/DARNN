# Reduced RecMO experiment: four-hour partial results

Saved 1773 of 2964 planned jobs, including 876 reused completed jobs from the earlier queue. Recorded failures: 0. Unfinished jobs: 1191. The time limit was reached and no new batch was started.

570 model/configuration groups have all three seeds (7, 17, 27). 38 groups are partial. The comparisons below use only complete three-seed groups for both RecMO and Direct MIMO.

Compare the same dataset, horizon, and backbone. Each job trains for 200 updates with width 32, history 48, and batch size 32. Model selection uses validation MSE. Values below compare mean test MSE across seeds, not statistical significance. Synthetic data use known generators; real datasets use separate variables and chronological splits.

| Backbone | Horizon | Window | Matched series | RecMO wins / losses / ties | Median MSE improvement (%) |
|---|---:|---:|---:|---:|---:|
| gru | 24 | 4 | 19 | 14 / 5 / 0 | 4.69 |
| gru | 24 | 12 | 19 | 14 / 5 / 0 | 5.48 |
| gru | 24 | 24 | 19 | 0 / 0 / 19 | 0.00 |
| gru | 48 | 4 | 19 | 11 / 8 / 0 | 7.64 |
| gru | 48 | 12 | 19 | 9 / 10 / 0 | -0.19 |
| gru | 48 | 24 | 19 | 11 / 8 / 0 | 0.55 |
| lstm | 24 | 4 | 19 | 13 / 6 / 0 | 2.54 |
| lstm | 24 | 12 | 19 | 14 / 5 / 0 | 7.54 |
| lstm | 24 | 24 | 19 | 0 / 0 / 19 | 0.00 |
| lstm | 48 | 4 | 19 | 11 / 8 / 0 | 5.59 |
| lstm | 48 | 12 | 19 | 13 / 6 / 0 | 7.05 |
| lstm | 48 | 24 | 19 | 10 / 9 / 0 | 0.22 |
| mlp | 24 | 4 | 19 | 15 / 4 / 0 | 9.26 |
| mlp | 24 | 12 | 19 | 14 / 5 / 0 | 4.26 |
| mlp | 24 | 24 | 19 | 0 / 0 / 19 | 0.00 |
| mlp | 48 | 4 | 19 | 16 / 3 / 0 | 38.20 |
| mlp | 48 | 12 | 19 | 16 / 3 / 0 | 4.58 |
| mlp | 48 | 24 | 19 | 14 / 5 / 0 | 1.33 |

Positive improvements favor RecMO; negative improvements favor Direct MIMO. All matched results are retained in [the CSV](matched_comparisons.csv). W=H is mathematically identical to Direct MIMO and is retained as a control. Series are not statistically independent replicates; median relative improvement is descriptive and is not a pooled cross-dataset error metric.

The queue reached DA-RNN block forecasting before the budget expired. Later recursive-one-step and DirMO jobs are unfinished, so this report does not claim a full forecasting-strategy comparison. Context and learned-loss ablations were intentionally deferred in the reduced plan. GPU support was added to the repository but these results used CPU.

These small, fixed-budget experiments do not establish novelty or general superiority. Larger architecture-specific tuning and multiple data splits remain necessary.
