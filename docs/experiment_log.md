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

## Recorded outcomes

Initial neural readout: test precision .1035, recall .7990, F1 .1833. Closed loop
(seeds 10000–10029, 750 steps): mean pipes .0667, max 2, mean survival 3.308 s.
Random: .2333 mean pipes. Oracle: 6.7. Instinct and direct logistic v1: zero.
No tuning was performed after seeing this benchmark.

DAgger: neural v1 controls 30 fresh seeds 1000–1029, adding 4,857 states and 181
teacher FLAP labels. Combined data: 41,249 samples / 80 episodes. Existing held-out
splits are unchanged. Same 20-component/L2=.1 readout is refit; direct logistic v2
is also refit on exactly the combined data. Physics, encoder and threshold remain
unchanged. Neural v2 test F1 .1766; 30-seed benchmark mean pipes zero. Direct v2
also scores zero. DAgger did not improve this configuration.

All per-controller episodes, truncation flags, survival measures and class metrics
are preserved in `docs/results/`. Benchmark seeds are reused for paired version
comparison, not presented as an independent second confirmation. This small result
does not establish that connectomes are useless; it shows this interface/readout
has not produced a useful policy. Further encoder or threshold work would be a
new explicitly recorded experiment, not a silent replacement of these results.

The first collector's brain timing metadata refers to its last episode; sidecars
now explicitly label that scope. Logs contain each episode's timing. The collector
was corrected to accumulate all neural steps for future runs. Evaluation timing
already accumulated all evaluated episodes. No neural samples or outcomes changed.

## Gap-threat encoder v3

Physics, oracle teacher, readout class (20 PCA components, L2=.1, threshold=.5) and
brain settings match baseline v1. Only the sensory map changes: LC4/LPLC2 now carry
upper/lower wall-collision threat (`proximity ×` shortfall from `clearance_margin`),
not identical pipe-distance looming. LC10a and LPLC1 still carry signed gap error and
velocity. Direct baseline raw features are `[top_threat, bottom_threat, signed_error]`.
Config: `configs/gap_threat_v3.json` (`fly_radius=12`, `clearance_margin=95`).
Dataset/model/eval artifacts: `oracle_v3`, `flap_readout_v3`, `direct_readout_v3`,
`evaluation_v3`. v1/v2 artifacts are retained unchanged.

Closed-loop (seeds 10000–10029, 750 steps): random 0.23 mean pipes; oracle 6.70;
instinct 0.00 (mean survival 0.84 s); fly readout v3 0.13 mean pipes (max 3, survival
3.54 s); **direct logistic v3 7.00 mean pipes / 15.00 s** (every episode truncated at
cap). Neural test F1 .215 (precision .123, recall .869). The wall/gap feature map is
highly informative for a linear policy; the frozen-connectome readout still does not
match that closed-loop performance.

## Lookahead encoder v4

Same physics/teacher/readout class as v3. Sensory map adds short-horizon imagination
(`look_ahead=0.24 s`) using game `integrate` physics: LC4 = max(current upper threat,
flap-trajectory risk); LPLC2 = max(current lower threat, wait-trajectory risk). Direct
raw features are `[flap_risk, wait_risk, signed_error]`. Config `gap_threat_v4.json`.

Closed-loop: random 0.23; oracle 6.70; instinct 0.00 (3.14 s); fly readout v4 **0.00**
pipes (3.33 s); **direct v4 7.00 / 15.00 s**. Neural test F1 .217. Lookahead makes the
“last safe flap” problem explicit for a linear controller; the connectome readout still
fails to use it in closed loop.

## Gated lookahead v5

v4 falsely lit **wait fatal** while above the gap (falling could clip the upper lip).
v5 scores flap risk only on upper hits and wait risk only on lower hits, then gates:
flap-fatal × too_high, wait-fatal × too_low. `encoder.map_version=5`.

Closed-loop: random 0.23; oracle 6.70; instinct **2.67** pipes (8.16 s); fly readout v5
0.00 pipes (3.48 s); direct v5 4.13 pipes (10.60 s). Gating helped the DNp01 instinct
path a lot; direct stayed strong but below the perfect v3/v4 cap; brain readout still
fails closed-loop.

## Height bias v6

Readout was flap-happy (high recall). v6 strengthens too-high/too-low (`target_gain=1`,
`position_gate_range=70`) and adds actuator bias: dampen P(FLAP) when too high, boost
when too low, slightly higher decision threshold. Play can apply the actuator bias on
v5 weights via `configs/play_v5_fall_bias.json`.

Closed-loop (seeds 10000–10029, 750 steps): random 0.23; oracle 6.70; **instinct,
brain, and direct all 7.00 mean pipes / 15.00 s** (every episode hit the cap). Height
bias + stronger position channels finally let the policy wait/fall when too high.
