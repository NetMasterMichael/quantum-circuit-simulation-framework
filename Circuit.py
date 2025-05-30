import numpy as np
import random
import re

class Circuit:

    IDENTITY = np.array([[1,0],[0,1]])
    PAULI_X = np.array([[0,1],[1,0]])
    HADAMARD = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
    # Map strings to operators
    GATES = {
        'I': IDENTITY,
        'X': PAULI_X,
        'H': HADAMARD
    }
    # Floating point precision for rounding, since this is running on a classical computer after all, which are famously bad at division
    FP_PRECISION = 15 

    def __init__(self, qubits):
        # Setup basis vector of qubit states
        self._qubits = qubits
        self._circuit_state = np.zeros(2 ** self._qubits)
        # Set circuit state to |0> (tensored with # of qubits)
        self._circuit_state[0] = 1
        self._operator_cache = {}

    """
    def get_state(self, index):
        # Note: It is currently impossible to observe a state in quantum mechanics without it collapsing! To be used for debugging only
        return self._circuit_state[index]
    """
    
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

    def construct_operator(self, operatorKey):
        # If operator U is already cached, skip    
        if operatorKey in self._operator_cache:
            return
        # Assumption: Gates are provided in order
        # Start with 1 dimensional identity matrix
        operator_matrix = 1
        # Apply Regex to tokenize string into array of gates
        operators = re.findall(r'\w+', operatorKey)
        qubit_count = 0
        for operator in operators:
            # Tokenize gate further into [0] = gate type and [1] = target qubit
            tokenized_operator = re.findall(r'[A-Za-z]+|\d+', operator)
            # If there is a gap between gates, pad empty space with tensored identity matrices
            while int(tokenized_operator[1]) > qubit_count:
                operator_matrix = np.kron(operator_matrix, self.IDENTITY)
                qubit_count += 1
            # Tensor gate onto the operator matrix
            operator_matrix = np.kron(operator_matrix, self.GATES[tokenized_operator[0]])
            qubit_count += 1
        # All gates have been tensored, pad with tensored identity matrices until 2^n x 2^n dimensions, where n = qubits
        while qubit_count < self._qubits:
            operator_matrix = np.kron(operator_matrix, self.IDENTITY)
            qubit_count += 1
        # Add new operator into the cache
        self._operator_cache[operatorKey] = operator_matrix

    def apply_operator(self, key):
        if key not in self._operator_cache:
            self.construct_operator(key)
        self._circuit_state = np.dot(self._operator_cache[key], self._circuit_state)

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