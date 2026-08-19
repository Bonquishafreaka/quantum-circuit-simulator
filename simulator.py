import numpy as np
from functools import reduce

# --- Single-qubit gates ---
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def RX(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def RY(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def RZ(theta):
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]], dtype=complex)


class QuantumCircuit:
    """State-vector simulator for an n-qubit system."""

    def __init__(self, n_qubits):
        self.n = n_qubits
        self.state = np.zeros(2 ** n_qubits, dtype=complex)
        self.state[0] = 1.0  # |00...0>
        self.history = []

    def _apply_single(self, gate, target):
        ops = [gate if q == target else I for q in range(self.n)]
        full = reduce(np.kron, ops)
        self.state = full @ self.state

    def _apply_controlled(self, gate, control, target):
        dim = 2 ** self.n
        full = np.zeros((dim, dim), dtype=complex)
        for basis in range(dim):
            bits = [(basis >> (self.n - 1 - q)) & 1 for q in range(self.n)]
            if bits[control] == 0:
                full[basis, basis] = 1.0
            else:
                new_bits = bits.copy()
                for out in range(2):
                    amp = gate[out, bits[target]]
                    if amp == 0:
                        continue
                    new_bits[target] = out
                    idx = sum(b << (self.n - 1 - q) for q, b in enumerate(new_bits))
                    full[idx, basis] += amp
        self.state = full @ self.state

    # --- Gate API ---
    def x(self, q): self._apply_single(X, q); self.history.append(("X", q))
    def y(self, q): self._apply_single(Y, q); self.history.append(("Y", q))
    def z(self, q): self._apply_single(Z, q); self.history.append(("Z", q))
    def h(self, q): self._apply_single(H, q); self.history.append(("H", q))
    def s(self, q): self._apply_single(S, q); self.history.append(("S", q))
    def t(self, q): self._apply_single(T, q); self.history.append(("T", q))
    def rx(self, q, th): self._apply_single(RX(th), q); self.history.append(("RX", q))
    def ry(self, q, th): self._apply_single(RY(th), q); self.history.append(("RY", q))
    def rz(self, q, th): self._apply_single(RZ(th), q); self.history.append(("RZ", q))
    def cx(self, c, t): self._apply_controlled(X, c, t); self.history.append(("CX", c, t))
    def cz(self, c, t): self._apply_controlled(Z, c, t); self.history.append(("CZ", c, t))

    def probabilities(self):
        return np.abs(self.state) ** 2

    def measure(self, shots=1024):
        probs = self.probabilities()
        outcomes = np.random.choice(len(probs), size=shots, p=probs)
        counts = {}
        for o in outcomes:
            key = format(o, f"0{self.n}b")
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def statevector(self):
        return self.state.copy()

    def __str__(self):
        lines = []
        for i, amp in enumerate(self.state):
            if abs(amp) > 1e-10:
                lines.append(f"  {amp:+.3f} |{format(i, f'0{self.n}b')}>")
        return "\n".join(lines) if lines else "  (zero state)"
