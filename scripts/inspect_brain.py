"""Inspect the installed public API and run a real stimulation experiment."""
import os
from pathlib import Path
import json
import inspect
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("FLY_DATA", str(ROOT / "data/brain"))
os.environ.setdefault("NUMBA_NUM_THREADS", "4")


def main():
    import flybrain
    from flybrain import FlyBrain
    from flybrain.reservoir import Trace

    brain = FlyBrain(device="auto")
    populations = {}
    for name in ("LC4", "LPLC2", "LC10a", "LPLC1", "DNp01", "descending_neuron"):
        populations[name] = {side or "all": len(brain.cells([name], side=side))
                             for side in (None, "L", "R")}
        print(name, populations[name], flush=True)
    trace = Trace(brain, types=["descending_neuron"])
    dn = Trace(brain, types=["DNp01"])
    loom = brain.cells(["LC4", "LPLC2"])
    results = {}
    for name, strength in (("quiet", 0.0), ("loom", 0.8)):
        brain.reset(seed=64)
        trace.reset()
        dn.reset()
        activity = []
        start = time.perf_counter()
        for _ in range(100):
            fired = brain.step(inject=[(loom, strength)])
            trace.observe(fired)
            activity.append(float(dn.observe(fired).mean()))
        results[name] = {"dnp01_mean_trace": float(np.mean(activity)),
                         "step_ms_including_jit": 10 * (time.perf_counter() - start)}
    report = {"flybrain_version": flybrain.__version__, "device": brain.device,
              "neurons": brain.n, "connections": len(brain.weights), "dt": brain.dt,
              "populations": populations, "trace_shape": list(trace.features().shape),
              "reset_signature": str(inspect.signature(brain.reset)), "experiments": results}
    print(json.dumps(report, indent=2), flush=True)
    (ROOT / "docs/brain_inspection.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
