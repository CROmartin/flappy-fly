# Experiment ledger

## Baseline v1

Python 3.11, flybrain 0.1.0, CPU, four Numba threads, 20 ms neural/game dt.
Connectome settings are the package defaults; `sensory_input=True` preserves all
provided connections. No synaptic weights or brain gain/noise parameters change.

Before collecting data, the first oracle settings (horizon .22 s, offset -12 px)
failed by flapping too early and clipping the upper pipe. Set horizon .20 s and
offset +20 px, with deadband 12 px and cooldown .12 s. This is a teacher-specific
prediction target, not a change to game physics. Twenty development seeds verify
that the oracle averages at least five pipes over 30 seconds and beats random.
All later comparisons use the same physics and teacher settings.

The baseline_v1, trained_v1 and trained_v2 configuration files deliberately match.
Version 2 changes the training data through DAgger, not the encoder or physics.
Training selects 20 PCA components and L2=.1 in advance, with threshold=.5.
Five grouped folds contain whole training episodes. Class balancing applies only
to fitting/CV; validation and test remain untouched. No threshold tuning on test.

Initial full dataset: 50 oracle episodes, seeds 0–49, capped at 750 decisions
(15 seconds) per episode. This cap keeps the first experiment bounded; it does not
change simulation speed or physics. Train seeds 0–34, validation 35–41, test 42–49.
Three normalized raw inputs are retained for the direct logistic baseline.

Development smoke data (5 x 80 steps) proved save/train/load/act, but its metrics
are not presented as the experiment's result. The first loaded-model run exposed
an import-order error from setting NUMBA_NUM_THREADS after importing flybrain;
fixed by using public numba.set_num_threads without changing its environment.

Frozen-brain tests compare weights before and after simulation, and replay traces
after public seeded resets. CPU timing includes cold JIT on the first episode;
subsequent episode timings are measured separately.
