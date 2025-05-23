import numpy as np
import random

class Circuit:

    IDENTITY = np.array([[1,0],[0,1]])
    PAULI_X = np.array([[0,1],[1,0]])
    PAULI_Y = np.array([0,-1j],[1j,0])
    PAULI_Z = np.array([[1,0],[0,-1]])
    HADAMARD = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
    # Floating point precision for rounding, since this is running on a classical computer after all, which are famously bad at division
    FP_PRECISION = 15 

    def __init__(self, qubits):
        # Setup basis vector of qubit states
        self._circuit_state = qubits
        self._circuit_state = np.zeros(2 ** qubits)
        # Set circuit state to |0> (tensored with # of qubits)
        self._circuit_state[0] = 1

    def get_state(self, index):
        # Note: It is currently impossible to observe a state in quantum mechanics without it collapsing! To be used for debugging only
        return self._circuit_state[index]
    
    def get_circuit_state(self):
        return self._circuit_state

    def set_qubit(self, index, new_state):
        # Validate that it is a valid quantum state
        # Condition: it's modulus/length must be equal (or close, to account for FP errors) to 1
        if round(np.sqrt(np.sum(new_state * new_state))) == 1 and new_state.shape == (2,):
            self._circuit_state[qubit_index] = new_state
            return True
        else:
            return False
        
    def round(self, qubit_state):
        return np.round(qubit_state, decimals=Circuit.FP_PRECISION)
        
    def hadamard(self, qubit_index):
        new_state = self.round(np.dot(Circuit.HADAMARD, self._circuit_state[qubit_index]))
        self._circuit_state[qubit_index] = new_state

    def hadamard_range(self, start_qubit_index, end_qubit_index):
        for i in range(start_qubit_index, end_qubit_index + 1):
            self.hadamard(i)

    def hadamard_all(self):
        for i in range(0, self._circuit_state):
            self.hadamard(i)

    def measure_qubit(self, qubit_index):
        qubit_state = self._circuit_state[qubit_index]
        # Apply Born's Rule
        qubit_state = [qubit_state[0] ** 2, qubit_state[1] ** 2]
        # Collapse qubit to either |0> or |1>
        probabilities_sum = np.sum(qubit_state)
        collapse = random.uniform(0, probabilities_sum)
        if collapse > qubit_state[0]:
            # Collapse to |1>
            self._circuit_state[qubit_index] = np.array([0, 1])
            return np.array([0, 1])
        else:
            # Collapse to |0>
            self._circuit_state[qubit_index] = np.array([1, 0])
            return np.array([1, 0])