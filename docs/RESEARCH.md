# RecMO research protocol

The historical `univariate/` and `multivariate/` TensorFlow code remains intact.
`multivariate/main.py` uses input-attention encoding and a temporally attentive
recurrent decoder from `multivariate/utils.py`; `Generate_stock_data.py` supplies
legacy windows. The original requirements omit TensorFlow; the README explicitly
names TensorFlow 1.3. Legacy examples and results remain identifiable.

The new `recmo/` package uses PyTorch. Its DA-RNN adaptation follows the input
and temporal attention ideas in Qin et al. (2017), with a multi-output head.
Single-feature input attention is trivial and offers no feature-selection claim.
The MLP and feature-trajectory projection pool expanding contexts to the configured
history length. Recurrent and temporal-attention computations otherwise process
the complete context. Separate context uses two encoders, so parameter counts
must be considered when interpreting context comparisons.

Chronological splits are 60/20/20. Every target belongs entirely to its split;
validation/test histories may contain earlier observed values. Normalization
uses training observations only. Forecast origins are deterministic and shared
across methods at each horizon. Fixed stride, training/evaluation caps, variable
indices, compressed-source hashes, and normalization are recorded. No unrelated
series are averaged. Primary experiments are univariate; multivariate inputs
are not yet implemented.

Direct predicts the full horizon; recursive predicts one step at a time; RecMO
reuses one block predictor; DirMO uses independent heads/backbones per block.
No future target is passed to a predictor. Recursive outputs stay in the graph.
Final blocks are truncated to exactly the requested horizon.

The primary uniform objective averages over time steps, including partial
blocks. Window weighting instead takes each block MSE then combines it using
fixed exponential, regularized softmax with a uniform floor, or uncertainty
weights. Learned weights are optimization parameters, not calibrated intervals.
Validation and test always use unweighted full-horizon MSE. GradNorm, dropout,
propagation-aware training, learned feedback, refinement, and adaptive windows
are future phases; these are not claimed as implemented.

The overnight queue first establishes standard comparisons, then context and
weighting ablations. Each job saves its predictions, selected checkpoint, losses,
metrics, and split origins before the next job starts. Repeat runs skip completed
jobs; configuration changes require a fresh output directory. The time budget
interrupts training cleanly, leaving an incomplete job to retry. Reports retain
all outcomes, including losses. An incomplete queue is explicitly labeled.

Equal optimizer updates and batch size do not imply equal parameter counts or
wall-clock budgets. Architectures use a common small width, not an extensive
hyperparameter search. Cross-dataset raw errors are not directly comparable.
Seed variation is initialization variation on one split, not dataset uncertainty.
No novelty, superiority, or statistical significance is presumed.

References: [Qin et al., DA-RNN](https://arxiv.org/abs/1704.02971),
[dataset collection](https://github.com/laiguokun/multivariate-time-series-data),
and the papers already attributed in the historical README.

Device selection is explicit and recorded in each new job. Existing configuration
files continue to default to CPU. All neural inputs and parameters move to the
selected accelerator; sampling and sensitivity noise use CPU-local generators
then move the generated tensors, preserving a common random stream across
backends. Backend differences still prevent exact cross-device reproducibility.
Accelerator tests run conditionally on hosts with usable CUDA/MPS hardware.
