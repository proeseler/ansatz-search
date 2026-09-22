"""Measure expressibility on a (simulated) IBM quantum computer and compare it with the ideal value.

A quantum computer cannot return statevectors, so the fidelities between random
parameter pairs are measured by compute–uncompute: run U(θa), then U(θb)†, and
count how often all qubits return 0. The device here is a fake IBM backend, a
local simulation with that device's calibrated noise, so no IBM account is needed.

    pip install "ansatz-search[ibm,plot]"
    python examples/hardware_expressibility.py [--pairs 200] [--shots 2000] [--seeds 2] [--run examples/runs/quickstart]

`--run` adds the three best circuits of a finished 4-qubit search, e.g. the quickstart.

To use a real device instead (needs an IBM Quantum account), replace `device`:

    from qiskit_ibm_runtime import QiskitRuntimeService
    device = QiskitRuntimeService().least_busy(operational=True, simulator=False, min_num_qubits=4)

Reading hardware results: noise mixes the output states, which pushes the
fidelities towards those of random states and can make a circuit look *more*
expressive than it is. Shot noise slightly raises the KL divergence.
"""

import argparse
from pathlib import Path

from qiskit_ibm_runtime.fake_provider import FakeManilaV2

from ansatz_search import Evaluation, HierarchicalCostFunction, evaluate
from ansatz_search.analysis import top_circuits
from ansatz_search.backends.numpy import NumpyCompiler
from ansatz_search.backends.qiskit import QiskitCompiler, QiskitFidelityProvider
from ansatz_search.benchmarks import SIM2019
from ansatz_search.metrics import Expressibility

parser = argparse.ArgumentParser(description="Expressibility on a simulated IBM device vs. ideal simulation.")
parser.add_argument("--pairs", type=int, default=200, help="random parameter pairs per circuit and seed")
parser.add_argument("--shots", type=int, default=2000, help="shots per pair")
parser.add_argument("--seeds", type=int, default=2)
parser.add_argument("--run", type=Path, default=None, help="folder of a finished 4-qubit search")
parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "runs" / "hardware_expressibility")
args = parser.parse_args()

device = FakeManilaV2()  # 5-qubit IBM device model with its calibrated noise
circuits = {f"Sim {i}": SIM2019[i]() for i in (1, 2, 9, 11, 15)}  # from barely to very expressive
if args.run is not None:
    for rank, found in enumerate(top_circuits(args.run, NumpyCompiler(), k=3), start=1):
        circuits[f"search #{rank}"] = found.spec


def expressibility(**provider):
    return HierarchicalCostFunction({0: [
        Expressibility(num_qubits=4, num_of_param_samples=2 * args.pairs, log_scale=True, seed=42, **provider),
    ]})


# Ideal: fidelities from exact statevectors (NumPy). Device: fidelities measured on the transpiled circuits.
ideal = evaluate(circuits, cost_fn=expressibility(), seeds=args.seeds)
on_device = QiskitFidelityProvider(backend=device, shots=args.shots, seed=42)
noisy = evaluate(circuits, cost_fn=expressibility(fidelity_provider=on_device), compiler=QiskitCompiler(),
                 seeds=args.seeds)

print(f"\n{'circuit':>10} {'KL ideal':>10} {'KL ' + device.name:>16}   (lower is more expressive)")
for label in circuits:
    kl = [e.circuits[label]["metrics"]["expressibility"]["value"]["mean"] for e in (ideal, noisy)]
    print(f"{label:>10} {kl[0]:10.4f} {kl[1]:16.4f}")

# One plot with each circuit's ideal and device rows next to each other.
rows = {}
for label in circuits:
    rows[f"{label} · ideal"] = {**ideal.circuits[label], "source": "baseline"}
    rows[f"{label} · device"] = {**noisy.circuits[label], "source": "given"}
both = Evaluation(rows, {**noisy.settings, "device": device.name, "shots": args.shots})
both.save(args.out / "eval.json")
both.plot(metrics=["expressibility"], sort=False, out=args.out / "eval.png",
          title=f"Expressibility on {device.name} vs. ideal simulation",
          legend={"found": f"{device.name}: measured fidelities ({args.shots} shots)",
                  "baseline": "ideal: exact statevectors"})
print(f"saved {args.out / 'eval.json'} and {args.out / 'eval.png'}")
