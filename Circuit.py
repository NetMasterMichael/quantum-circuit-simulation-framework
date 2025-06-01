import random
import re

class Circuit:

    def __init__(self, qubits: int, operator_cache: bool = False, hardware_mode: str = 'CPU'):
        # Choose to load cupy (numpy but for GPUs) or numpy
        if hardware_mode == 'GPU':
            import cupy as np
        else:
            import numpy as np
        self.np = np
        # Setup basis vector of qubit states
        self._qubits = qubits
        self._circuit_state = np.zeros(2 ** self._qubits, dtype = complex)
        # Set circuit state to |0> (tensored with # of qubits)
        self._circuit_state[0] = 1
        self._operator_cache_state = operator_cache
        self._operator_cache = {}

        # Gates
        self.IDENTITY = np.array([[1,0],[0,1]], dtype = complex)
        self.PAULI_X = np.array([[0,1],[1,0]], dtype = complex)
        self.PAULI_Y = np.array([[0,-1j],[1j,0]], dtype = complex)
        self.PAULI_Z = np.array([[1,0],[0,-1]], dtype = complex)
        self.HADAMARD = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype = complex)
        
        # Map strings to gates
        self.SINGLE_QUBIT_GATES = {
            'I': self.IDENTITY,
            'X': self.PAULI_X,
            'Y': self.PAULI_Y,
            'Z': self.PAULI_Z,
            'H': self.HADAMARD
        }
    
    def get_circuit_state(self):
        return self._circuit_state

    def construct_operator(self, operator_key):
        np = self.np
        # If operator U is already cached, skip    
        if operator_key in self._operator_cache:
            return self._operator_cache[operator_key]
        # Start with 1 dimensional identity matrix
        operator_U = np.array([1], dtype = complex)
        # Apply Regex to tokenize string into tuples of gates
        gates = re.split(" ", operator_key)
        print(gates)
        # Initialize empty sequence of gates for constructing the operator
        operator_construction = [None] * self._qubits
        # Fill the operator construction with gates present in the gates variable (expected to be partial)
        for gate in gates:
            # Read the 2nd index, which contains the target qubit of the gate
            target_index = int(gate[1])
            operator_construction[target_index] = gate
        # Construct U by tensoring gates together into one matrix
        print(operator_construction)
        for gate in operator_construction:
            if gate == None:
                # No gate provided, so tensor I matrix
                operator_U = np.kron(operator_U, self.IDENTITY)
            elif gate[0] in self.SINGLE_QUBIT_GATES:
                operator_U = np.kron(operator_U, self.SINGLE_QUBIT_GATES[gate[0]])
            else:
                # Gate not recognized, so print warning and tensor I matrix
                print("WARNING: Gate " + gate[0] + " not recognized, substituting for I")
                operator_U = np.kron(operator_U, self.IDENTITY)
        # If cache is enabled, then add it to the cache
        if self._operator_cache_state:
            self._operator_cache[operator_key] = operator_U
        return operator_U

    def apply_operator(self, key):
        np = self.np
        U = self.construct_operator(key)
        self._circuit_state = np.dot(U, self._circuit_state)