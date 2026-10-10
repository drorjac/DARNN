# DARNN
An implementation of the paper 

[A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction](https://arxiv.org/abs/1704.02971).                   
  Yao Qin, Dongjin Song, Haifeng Cheng, Wei Cheng, Guofei Jiang, Garrison. W. Cottrell                 
 International Joint Conference on Artificial Intelligence (IJCAI), 2017. 

run in tf 1.3  

This repository is used to obtain a baseline result for the following paper:
[A Memory-Network Based Solution for Multivariate Time-Series Forecasting
](https://arxiv.org/abs/1809.02105).

## Time Series Datasets

### Univariate
* [Beijing PM2.5](https://github.com/petwill/DARNN/univariate/data/beijing)
* [GefCom Electricity(2014)](https://github.com/petwill/DARNN/univariate/data/GEFC2014)

### Multivariate

[Source](https://github.com/laiguokun/multivariate-time-series-data)

- Solar-Energy
- Traffic
- Electricity
- Exchange-Rate

## RecMO research (PyTorch)

The historical implementation above is preserved. The new framework compares
univariate multi-step Direct MIMO, recursive one-step, RecMO, and DirMO using
MLP, LSTM, GRU, and DA-RNN backbones, plus persistence and ridge regression.
It includes rolling, expanding, and separately encoded prediction contexts;
fixed, softmax, and uncertainty window objectives; seven synthetic generators;
and explicit single-variable extraction from the four real datasets.

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python -m recmo.experiment --config configs/smoke.json --output results/smoke
python -m recmo.experiment --config configs/overnight.json --output results/overnight
```

Set `data_root` in each configuration to a local checkout of the linked dataset
collection. The overnight run has an eight-hour budget and resumes completed
jobs when rerun with the same configuration and output. Outputs include per-job
checkpoints, forecasts, original-unit and normalized metrics, per-lead errors,
learned weight histories, provenance, CSV summaries, and comparison figures.
Regenerate a partial report while training continues:

```bash
python -m recmo.experiment --config configs/overnight.json \
  --output results/overnight --summarize-only
```

See [research protocol](docs/RESEARCH.md). End-to-end validation establishes
correctness, not evidence that RecMO improves forecasts. Later propagation,
feedback, refinement, and adaptive-window mechanisms remain separate phases.
