import numpy as np

class Circuit:

    HADAMARD = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
    # Floating point precision for rounding, since this is running on a classical computer after all, which are famously bad at division
    FP_PRECISION = 15 

    def __init__(self, qubits):
        # For now, this will be all |0>
        self._qubits = np.array([[1.0, 0.0]] * qubits)
        self._total_qubits = qubits

    def get_qubit(self, qubit_index):
        return self._qubits[qubit_index]
    
    def get_all_qubits(self):
        return self._qubits

    def set_qubit(self, qubit_index, new_state):
        # Validate that it is a valid quantum state
        # Condition: it's modulus/length must be equal (or close, to account for FP errors) to 1
        if round(np.sqrt(np.sum(new_state * new_state))) == 1 and new_state.shape == (2,):
            self._qubits[qubit_index] = new_state
            return True
        else:
            return False
        
    def get_total_qubits(self):
        return self._total_qubits
        
    def round(self, qubit_state):
        return np.round(qubit_state, decimals=Circuit.FP_PRECISION)
        
    def hadamard(self, qubit_index):
        new_state = self.round(np.dot(Circuit.HADAMARD, self._qubits[qubit_index]))
        self._qubits[qubit_index] = new_state

    def hadamard_range(self, start_qubit_index, end_qubit_index):
        for i in range(start_qubit_index, end_qubit_index + 1):
            self.hadamard(i)

    def hadamard_all(self):
        for i in range(0, self._total_qubits):
            self.hadamard(i)