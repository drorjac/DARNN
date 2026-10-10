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

### CPU, CUDA, and Apple GPU execution

Select a device using `--device cpu`, `--device cuda`, `--device cuda:1`,
`--device mps`, or `--device auto`. Alternatively set `"device": "auto"` in
configuration. `auto` prefers CUDA, then Apple MPS, then CPU. Omitting the option
preserves the CPU default for existing runs. An explicit unavailable GPU request
fails with a diagnostic rather than silently running on CPU. Install a PyTorch
build compatible with the host OS and NVIDIA drivers when applicable; see the
[PyTorch installation selector](https://pytorch.org/get-started/locally/),
[CUDA notes](https://docs.pytorch.org/docs/stable/notes/cuda.html), and
[MPS notes](https://docs.pytorch.org/docs/stable/notes/mps.html).

```bash
python -m recmo.experiment --config configs/overnight_half.json \
  --output results/gpu --device cuda --data-root /path/to/multivariate-time-series-data
```

Models, loss parameters, and tensors use the chosen device. Persistence and ridge
remain CPU baselines. Metrics record actual device, host, PyTorch version, and
platform. GPU inference timing synchronizes before and after measurement and
excludes copying results to the host. The measured interval includes batch
iteration and forecast concatenation. Checkpoints store CPU tensors for portable
restoration:

```python
from recmo.forecast import load_checkpoint
model, saved = load_checkpoint('results/gpu/jobs/JOB_ID/checkpoint.pt', device='auto')
# Standardize new histories with saved['normalization']['mean'] and ['scale'].
# Move histories to next(model.parameters()).device before calling model.
```

Use a fresh output folder when changing devices or configuration. GPU and CPU
results may differ numerically; hardware timings must be compared within the same
host/device. Tiny recurrent models can be slower on accelerators. Unsupported
backend operations are recorded as job failures; no silent CPU fallback is enabled.

### Sharing a queue between computers

Use the same Git commit, scientific configuration, and dataset files. Override
only the device and dataset location as needed. Shard indices are zero-based:

```bash
# Machine A: first half of the queue
python -m recmo.experiment --config configs/overnight_half.json \
  --output results/machine-a --device cuda --num-shards 2 --shard-index 0 \
  --data-root /path/to/multivariate-time-series-data

# Machine B: second half, with its own output folder
python -m recmo.experiment --config configs/overnight_half.json \
  --output results/machine-b --device cpu --num-shards 2 --shard-index 1 \
  --data-root /path/to/multivariate-time-series-data
```

Shards partition the deterministic job plan without duplicates and save their
own plan and execution metadata. Resume using exactly the same shard arguments
and output folder. Do not share a live output folder between workers. Collect
completed job directories from disjoint shards into a separate analysis folder,
then use `--summarize-only` there to generate a combined report. Keep the original
configuration, provenance, plans, and execution metadata alongside the analysis;
verify dataset hashes agree and retain device metadata for interpreting timing.
