import numpy as np

class Circuit:

    def __init__(self, qubits):
        # For now, this will be all |0>
        self.qubits = np.array([np.ones(qubits), np.zeros(n)])

    