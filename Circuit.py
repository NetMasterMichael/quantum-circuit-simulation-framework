import numpy as np
import random
import re

class Circuit:

    IDENTITY = np.array([[1,0],[0,1]])
    PAULI_X = np.array([[0,1],[1,0]])
    PAULI_Y = np.array([[0,-1j],[1j,0]])
    PAULI_Z = np.array([[1,0],[0,-1]])
    HADAMARD = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
    
    # Map strings to operators
    GATES = {
        'I': IDENTITY,
        'X': PAULI_X,
        'Y': PAULI_Y,
        'Z': PAULI_Z,
        'H': HADAMARD
    }

    def __init__(self, qubits):
        # Setup basis vector of qubit states
        self._qubits = qubits
        self._circuit_state = np.zeros(2 ** self._qubits)
        # Set circuit state to |0> (tensored with # of qubits)
        self._circuit_state[0] = 1
        self._operator_cache = {}
    
    def get_circuit_state(self):
        return self._circuit_state

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