# Verification performed

- Installed the public package in a Python 3.11 virtual environment; inspected
  actual source/signatures; ran checksum-verified real connectome download.
- Ran quiet/stimulated LC4/LPLC2 inspection and recorded real DNp01 response.
- Environment, oracle and encoder tests passed before full dataset collection.
- Human/oracle UI and trained-brain/death rendering exercised with SDL dummy video
  and audio drivers. Space, restart and escape are covered by event-loop tests.
  Physical desktop interaction has not been manually tested by a human.
- Saved a five-episode smoke dataset; trained, saved and reloaded the library
  readout; used it to control an unseen episode. Fixed the discovered import-order
  Numba thread configuration issue without changing the third-party package.
- Collected 50 oracle episodes / 36,392 samples; fitted neural and direct models.
- Evaluated all controllers on the same 30 unseen seeds and cap. Captured actual
  classifier and episode metrics, including failures and truncation.
- Collected 30 DAgger learner episodes / 4,857 samples, aggregated training-only
  data, fitted v2 neural/direct models and evaluated the same benchmark seeds.
- `RUN_BRAIN_TESTS=1 .venv/bin/python -m pytest -q`: **17 passed**. Ordinary tests:
  15 passed, two expensive brain tests skipped. Pygame emits one upstream
  pkg_resources deprecation warning; it does not affect test results.
- `python -m compileall -q flappy_fly scripts`, `python -m pip check`, and
  `git diff --check` passed. No runtime dependencies were modified or forked.

Screenshots in this directory are generated from actual neural-run observations,
not mocked telemetry. Model/dataset binaries and downloaded brain files are local
ignored artifacts; reproducible commands and small measured reports are tracked.
