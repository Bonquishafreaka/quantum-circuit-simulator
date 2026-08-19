# Quantum Circuit Simulator

A lightweight state-vector quantum computing simulator in pure NumPy.

## Features
- Single-qubit gates: X, Y, Z, H, S, T, and parametric RX/RY/RZ
- Two-qubit controlled gates: CX (CNOT), CZ
- State-vector inspection, measurement sampling, probability distributions

## Usage
```python
from simulator import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)      # Bell state
print(qc.measure(shots=1000))
```

## Run
```bash
pip install numpy
python examples.py
python test_simulator.py
```

## How it works
The n-qubit state is a 2ⁿ complex vector. Single-qubit gates are lifted to the full space via Kronecker products; controlled gates are built by iterating the computational basis. Measurement samples from |amplitude|².
