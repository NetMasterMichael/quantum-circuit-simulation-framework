import random
import re

class Circuit:


    def __init__(self, qubits: int, operator_cache: bool = False, hardware_mode: str = 'CPU'):
        if qubits < 1:
            raise ValueError("Qubits parameter must be at least 1, got " + str(qubits))

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


    def reset_circuit_state(self):
        np = self.np
        # Soft reset the circuit state to |0..0>
        self._circuit_state = np.zeros(2 ** self._qubits, dtype = complex)
        self._circuit_state[0] = 1


    def apply_operator(self, key):
        np = self.np
        preprocessed_key = self.apply_key_preprocessing(key)
        if preprocessed_key in self._operator_cache:
            U = self._operator_cache[preprocessed_key]
        else:
            U = self.construct_operator(key)
        self._circuit_state = np.dot(U, self._circuit_state)


    def construct_operator(self, operator_key):
        np = self.np
        # If operator U is already cached, skip
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


    def apply_key_preprocessing(self, operator_key):
        ## Tokenization
        # Remove spaces at start and end
        stripped_key = operator_key.strip(" ")
        # Split operator into tokens
        tokenized_key = re.split(" ", stripped_key)

        ## Validation
        for token in tokenized_key:
            if len(token) < 2:
                raise ValueError(f"Invalid gate provided: Received {token}, expected a gate of length at least 2")

            gate = token[0]
            if gate not in self.SINGLE_QUBIT_GATES:
                raise ValueError(f"Invalid gate provided: Received {token}, {gate} cannot be resolved to a valid gate")

        ## Ordering
        # Tokenize into tuples
        aux_gates = []
        for token in tokenized_key:
            aux_gates.append((token[0], int(token[1:])))
        # Use insertion sort to sort the tuples
        for i in range(1, len(aux_gates)):
            selected_gate = aux_gates[i]
            j = i - 1
            while j >= 0 and aux_gates[j][1] > selected_gate[1]:
                aux_gates[j + 1] = aux_gates[j]
                j -= 1
            aux_gates[j + 1] = selected_gate

        ## Padding
        # Pad empty space with I gates
        gate_cursor = 0
        gates = []
        for gate in aux_gates:
            # Pad before gate with identity gates
            while (gate_cursor < gate[1]):
                gates.append(("I", gate_cursor))
                gate_cursor += 1
                
            # Add gate to list
            gates.append((gate[0], gate[1]))
            gate_cursor += 1

        # gate_cursor should not be larger than the number of qubits. If so, this means more gates have been added than there are qubits in the circuit
        if gate_cursor > self._qubits:
            raise ValueError("Operator has more gates than there are qubits in the circuit, invalid shape. Qubits in circuit is " + str(self._qubits))

        # Pad with identity gates after all gates have been added
        while gate_cursor < self._qubits:
            gates.append(("I", gate_cursor))
            gate_cursor += 1

        ## Parse tuples back into a string
        output_key = gates[0][0] + str(gates[0][1])
        for i in range(1, len(gates)):
            output_key += " " + gates[i][0] + str(gates[i][1])
    
        return output_key


    def parse_operator_key(self, operator_key):
        gate_cursor = 0
        gates = []
        tokenized_key = re.split(" ", operator_key)

        for token in tokenized_key:
            if len(token) < 2:
                raise ValueError(f"Invalid gate provided: Received {token}, expected a gate of length at least 2")

            gate = token[0]
            if gate not in self.SINGLE_QUBIT_GATES:
                raise ValueError(f"Invalid gate provided: Received {token}, {gate} cannot be resolved to a valid gate")
            gate_index = int(token[1:])

            # Pad before gate with identity gates
            while (gate_cursor < gate_index):
                gates.append(("I", gate_cursor))
                gate_cursor += 1
                
            # Add gate to list
            gates.append((gate, gate_index))
            gate_cursor += 1

        # gate_cursor should not be larger than the number of qubits. If so, this means more gates have been added than there are qubits in the circuit
        if gate_cursor > self._qubits:
            raise ValueError("Operator has more gates than there are qubits in the circuit, invalid shape. Qubits in circuit is " + str(self._qubits))

        # Pad with identity gates after all gates have been added
        while gate_cursor < self._qubits:
            gates.append(("I", gate_cursor))
            gate_cursor += 1

        return gates    


    def measure(self):
        np = self.np
        # Make a copy of self._circuit_state with Born's rule applied
        probabilities = np.abs(self._circuit_state) ** 2
        # Select a random state based on the probability of that outcome
        outcome = np.random.choice(2 ** self._qubits, p = probabilities)
        # Collapse circuit state to the selected state
        collapsed_circuit_state = np.zeros(2 ** self._qubits, dtype = complex)
        collapsed_circuit_state[outcome] = 1 + 0j
        self._circuit_state = collapsed_circuit_state