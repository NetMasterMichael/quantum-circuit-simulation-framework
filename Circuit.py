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
        # Start with 1 dimensional identity matrix
        operator_matrix = 1
        # Apply Regex to tokenize string into tuples of gates
        operators = re.findall(r'^([A-Za-z]+)(\d+(?:,\d+)*)$', operatorKey)
        print(operators)
        operator_construction = [None] * self._qubits
        for operator in operators:
            target_index = int(operator[1])
            operator_construction[target_index] = operator
        print(operator_construction)
        for operator in operator_construction:
            if operator == None:
                operator_matrix = np.kron(operator_matrix, self.IDENTITY)
                print("No operator, tensored identity")
            elif operator[0] in self.GATES:
                operator_matrix = np.kron(operator_matrix, self.GATES[operator[0]])
                print("Tensored " + operator[0])
            else:
                print("WARNING: Gate " + operator[0] + " not recognized, substituting for I")
                operator_matrix = np.kron(operator_matrix, self.IDENTITY)
        self._operator_cache[operatorKey] = operator_matrix

    def apply_operator(self, key):
        if key not in self._operator_cache:
            self.construct_operator(key)
        self._circuit_state = np.dot(self._operator_cache[key], self._circuit_state)