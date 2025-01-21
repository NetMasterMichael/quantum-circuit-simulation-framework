import numpy as np

class Circuit:

    def __init__(self, qubits):
        # For now, this will be all |0>
        self._qubits = np.array([[1.0, 0.0]] * qubits)

    def get_qubit(self, qubit_index):
        return self._qubits[qubit_index]

    def set_qubit(self, qubit_index, new_state):
        # Validate that it is a valid quantum state
        # Condition: it's modulus/length must be equal (or close, to account for FP errors) to 1
        if np.isclose(np.sqrt(np.sum(new_state * new_state)), 1) and new_state.shape == (2,):
            self._qubits[qubit_index] = new_state
            return True
        else:
            return False