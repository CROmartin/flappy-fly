# Architecture and contracts

- `game/`: fixed-dt simulation with seeded pipe RNG; immutable observation snapshots.
  Circle/rectangle collision; an opening remains the target until the whole fly clears
  its trailing edge. Terminal steps are idempotent; reset recreates the pipe sequence.
- `controllers/`: reset(seed), act(state). Human input is a consumed one-shot event.
  The oracle's query is side-effect free; actual actions advance its cooldown clock.
- `brain/`: normalized task inputs, cached biological population indices, one public
  FlyBrain step per decision by default, decaying library traces. No fit method on
  BrainAdapter and no assignment to weights. `sensory_input=True` preserves edges.
- `training/`: observation → neural advance → teacher label/action → game step.
  Features and labels refer to the pre-action state; terminal outcomes cannot leak
  into labels. DAgger reuses the learner's feature vector and advances only once.
- `ui/`: a Pygame view, measured telemetry and bounded decision history. Rendering
  never edits state or neural features. The environment has no Pygame imports.

## Timing

One decision integrates 20 ms of game physics. Multiple neural steps per decision
are an explicit experimental setting and accelerate neural time relative to game
physics; metadata records this. The standard experiment uses exactly one. Public
seeded resets restart brain noise and zero all traces, including displayed traces.
There is no hidden warmup or unrecorded washout.

## Frozen parameters and fitted parameters

The library constructs signed normalized weights from the supplied connectome.
The project passes the default connectivity-preserving constructor settings.
Only library readout PCA/centering/scaling/logistic coefficients are fitted; runtime
cooldown is a hand-designed actuator rule. Raw game variables never bypass the
connectome in `brain` mode. Direct-input mode is explicitly named and separate.

## Artifact format

Schema 1 stores NumPy arrays without pickle and JSON sidecars with SHA256 digests.
Feature order is the ascending neuron indices returned by the library. All 1,314
DNs are included, even those without a lateral annotation. Readout probabilities
come from balanced training and are not probability-calibrated to natural labels.
Dataset splits are explicit string arrays; both seed and episode must stay within
one split. Combined datasets preserve original held-out episodes and reject reused
seeds. Saved model inference rejects incompatible dynamics, encoders and controls.

## Limitations

A missing/corrupt brain download raises an actionable upstream exception, never a
fallback to a fake network. Collectors retain arrays in memory until writing compressed
NPZ; the current 50-episode experiment fits on this machine, but very long runs should
be collected in chunks. Partial datasets are not checkpointed. Pygame can slow in wall
time on a slow CPU; physics and brain steps are never dropped to fake realtime.
There is no automatic hyperparameter sweep, reinforcement learning or graph mutation.
