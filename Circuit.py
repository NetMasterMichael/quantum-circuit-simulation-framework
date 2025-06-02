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

    def parse_operator_key(self, operator_key):
        gate_cursor = 0
        gates = []
        tokenized_key = re.split(" ", operator_key)

        for token in tokenized_key:
            gate = token[0]
            gate_index = int(token[1:])

            # Pad before gate with identity gates
            while (gate_cursor < gate_index):
                gates.append(("I", gate_cursor))
                gate_cursor += 1
                
            # Add gate to list
            gates.append((gate, gate_index))
            gate_cursor += 1

        # Pad with identity gates after all gates have been added
        while gate_cursor < self._qubits:
            gates.append(("I", gate_cursor))
            gate_cursor += 1

        return gates

    def construct_operator(self, operator_key):
        np = self.np
        # If operator U is already cached, skip    
        if operator_key in self._operator_cache:
            return self._operator_cache[operator_key]
        # Parse string to array of operators
        operator_construction = self.parse_operator_key(operator_key)
        # Start with 1 dimensional identity matrix
        operator_U = np.array([1], dtype = complex)
        # Construct U by tensoring gates together into one matrix
        for gate in operator_construction:
            if gate[0] in self.SINGLE_QUBIT_GATES:
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