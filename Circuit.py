import cupy as np
import random
import re

class Circuit:

    IDENTITY = np.array([[1,0],[0,1]])
    PAULI_X = np.array([[0,1],[1,0]])
    PAULI_Y = np.array([[0,-1j],[1j,0]])
    PAULI_Z = np.array([[1,0],[0,-1]])
    HADAMARD = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
    
    # Map strings to operators
    SINGLE_QUBIT_GATES = {
        'I': IDENTITY,
        'X': PAULI_X,
        'Y': PAULI_Y,
        'Z': PAULI_Z,
        'H': HADAMARD
    }

    def __init__(self, qubits: int, operator_cache: bool = True):
        # Setup basis vector of qubit states
        self._qubits = qubits
        self._circuit_state = np.zeros(2 ** self._qubits)
        # Set circuit state to |0> (tensored with # of qubits)
        self._circuit_state[0] = 1
        self._operator_cache_state = operator_cache
        self._operator_cache = {}
    
    def get_circuit_state(self):
        return self._circuit_state

    def construct_operator(self, operator_key):
        # If operator U is already cached, skip    
        if operator_key in self._operator_cache:
            return self._operator_cache[operator_key]
        # Start with 1 dimensional identity matrix
        operator_U = 1
        # Apply Regex to tokenize string into tuples of gates
        gates = re.findall(r'^([A-Za-z]+)(\d+(?:,\d+)*)$', operator_key)
        # Initialize empty sequence of gates for constructing the operator
        operator_construction = [None] * self._qubits
        # Fill the operator construction with gates present in the gates variable (expected to be partial)
        for gate in gates:
            target_index = int(gate[1])
            operator_construction[target_index] = gate
        # Construct U by tensoring gates together into one matrix
        for gate in operator_construction:
            if gate == None:
                # No gate provided, tensor I matrix
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
        U = self.construct_operator(key)
        self._circuit_state = np.dot(U, self._circuit_state)