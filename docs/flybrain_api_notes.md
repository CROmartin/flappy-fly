# Installed flybrain API verification

Inspected and executed PyPI `flybrain==0.1.0` on Python 3.11 / Apple Silicon.
The unmodified package uses NumPy/SciPy/Numba on CPU, or CuPy on NVIDIA CUDA.
`auto` selected CPU here; no MPS backend exists in this release.

- `FlyBrain(device="auto", seed=64, data=..., dt=0.020)` loads the real graph.
- `cells([type_or_superclass], side="L" | "R" | None)` returns neuron indices.
- `step(inject=[(indices, amount), ...])` returns fired indices for batch=1.
- `reset(seed=...)` is public: clears voltages/spikes and restarts noise.
- `Trace(brain, types=["descending_neuron"], tau=0.1)`; call `observe(fired)`
  after every neural step. `features()` returns a copy. `reset()` clears history.
- `Readout.fit(X, y, kind="logistic", groups=episode_ids, ...)` supports grouped
  cross-validation, PCA and L2 logistic regression. `predict` gives probabilities;
  `save` / `load` use non-pickled NPZ. The library's fold metric is AUC; our held-out
  reports also give FLAP precision/recall/F1 and confusion matrices.
- `run` exists but the interactive loop uses `step` + `Trace.observe` directly.
- `positions` is available; a coordinate renderer is outside this MVP.

Actual populations and timing are recorded in `brain_inspection.json`.
There are 1,314 descending cells, including cells without L/R labels; readout uses
all of them. DNp01 contains two cells. Matched seed=64, 100-step experiments gave
mean DNp01 trace 0 without input versus 4.698 with LC4/LPLC2 injection 0.8.
This is a stimulus check, not evidence of task learning.

Reset both brain and traces between episodes, with the episode seed. Determinism
is expected on the same backend, package and Numba thread count; floating-point
summation order and CPU/CUDA random streams can differ. No weights are modified.
Keep `sensory_input=True`: the alternative removes synapses, which this project
explicitly avoids. The library's weights are signed, normalized synapse counts,
not measured physiological synaptic strengths. Downloads are SHA256-checked by
flybrain on first acquisition, stored in `data/brain` (or `$FLY_DATA`).

Sources: [flybrain source](https://github.com/alextitonis/fly.ai),
[MaleCNS](https://male-cns.janelia.org/). Package source was inspected before
implementing BrainAdapter; no speculative APIs or private-array reset are used.
