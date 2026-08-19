import numpy as np
from simulator import QuantumCircuit


def test_bell_entanglement():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    p = qc.probabilities()
    assert np.isclose(p[0], 0.5) and np.isclose(p[3], 0.5)
    assert np.isclose(p[1], 0) and np.isclose(p[2], 0)


def test_x_gate():
    qc = QuantumCircuit(1)
    qc.x(0)
    assert np.isclose(qc.probabilities()[1], 1.0)


def test_normalization():
    qc = QuantumCircuit(3)
    qc.h(0); qc.ry(1, 0.7); qc.cx(0, 2)
    assert np.isclose(np.sum(qc.probabilities()), 1.0)


def test_hadamard_involution():
    qc = QuantumCircuit(1)
    qc.h(0); qc.h(0)
    assert np.isclose(qc.probabilities()[0], 1.0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"{name} passed")
