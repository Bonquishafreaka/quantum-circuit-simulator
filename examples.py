from simulator import QuantumCircuit
import numpy as np


def bell_state():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    print("Bell state (should be ~50/50 on 00 and 11):")
    print(qc)
    print(qc.measure(shots=1000))


def ghz_state(n=3):
    qc = QuantumCircuit(n)
    qc.h(0)
    for q in range(n - 1):
        qc.cx(q, q + 1)
    print(f"\nGHZ state ({n} qubits):")
    print(qc.measure(shots=1000))


def superposition():
    qc = QuantumCircuit(3)
    for q in range(3):
        qc.h(q)
    print("\nUniform superposition (all 8 outcomes ~equal):")
    print(qc.measure(shots=800))


if __name__ == "__main__":
    bell_state()
    ghz_state()
    superposition()
