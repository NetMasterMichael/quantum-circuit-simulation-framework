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
        np.set_printoptions(
            threshold = np.inf,
            linewidth = np.inf,
            precision = 15,
            suppress = True
        )

        # Setup basis vector of qubit states
        self._qubits = qubits
        self._circuit_state = np.zeros(2 ** self._qubits, dtype = complex)
        # Set circuit state to |00...0>
        self._circuit_state[0] = 1
        self._operator_cache_state = operator_cache
        self._operator_cache = {}

        # Gates
        self.IDENTITY = np.array([[1,0],[0,1]], dtype = complex)
        self.PAULI_X = np.array([[0,1],[1,0]], dtype = complex)
        self.PAULI_Y = np.array([[0,-1j],[1j,0]], dtype = complex)
        self.PAULI_Z = np.array([[1,0],[0,-1]], dtype = complex)
        self.HADAMARD = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype = complex)

        self.KET_0 = np.array([1,0], dtype = complex)             # |0>
        self.KET_1 = np.array([0,1], dtype = complex)             # |1>
        self.KETBRA_00 = np.outer(self.KET_0, self.KET_0.conj())  # |0><0|
        self.KETBRA_11 = np.outer(self.KET_1, self.KET_1.conj())  # |1><1|
                
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
        # Soft reset the circuit state to |00...0>
        self._circuit_state = np.zeros(2 ** self._qubits, dtype = complex)
        self._circuit_state[0] = 1


    def apply_operator(self, key):
        np = self.np
        normalized_key = self.apply_key_preprocessing(key)
        if normalized_key in self._operator_cache:
            U = self._operator_cache[normalized_key]
        else:
            U = self.compile_operator(normalized_key)
        self._circuit_state = np.dot(U, self._circuit_state)


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

            gate_index = int(token[1:])
            if gate_index > self._qubits:
                raise ValueError(f"Invalid gate provided: Received {token}, applies to wire {str(gate_index)} but there are only {str(self._qubits)} qubits")

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


    def compile_operator(self, operator_key):
        np = self.np
        # If operator U is already cached, skip
        # Parse string to array of operators
        operator_construction = self.parse_key_to_matrices(operator_key)

        # Start with 1x1 dimensional identity vector
        operator_U = np.array([1], dtype = complex)

        # Construct U by tensoring gates together into one matrix
        for gate in operator_construction:
            operator_U = np.kron(operator_U, gate)

        # If cache is enabled, then add it to the cache
        if self._operator_cache_state:
            self._operator_cache[operator_key] = operator_U
        return operator_U


    def parse_key_to_matrices(self, operator_key):
        gates = []
        tokenized_key = re.split(" ", operator_key)

        for token in tokenized_key:
            if token[0] == 'C':
                gates.append(self.compile_controlled_gate(token))
            else:
                gates.append(self.SINGLE_QUBIT_GATES[token[0]])

        return gates
    

    def compile_controlled_gate(self, token):
        np = self.np

        # Count number of control wires
        control_wires = 0
        for char in token:
            if char == 'C':
                control_wires += 1

        ## Decompose token into array of [letter, index] pairs
        # Side note: Far from the most beautiful solution, but functional for a prototype. To refactor into something less cursed later
        gate_structure = []
        tail_cursor = len(token) - 1
        for i in range(control_wires + 1):
            gate_structure.append([token[i], 0])
            
            # Compute the index by passing each digit backwards until we reach a comma or a gate
            digit = 0
            while token[tail_cursor] != ',':
                if i == tail_cursor:
                    break
                gate_structure[i][1] += int(token[tail_cursor]) * (10 ** digit)
                digit += 1
                tail_cursor -= 1
            tail_cursor -= 1

        ## Normalize indices so that the lowest index is 0
        # Find the lowest index
        lowest_index = 0
        for i in range(1, len(gate_structure)):
            if gate_structure[i][1] < gate_structure[lowest_index][1]:
                lowest_index = i
        # Subtract the lowest index from all indexes
        sub_val = gate_structure[lowest_index][1]
        for i in range(0, len(gate_structure)):
            gate_structure[i][1] = gate_structure[i][1] - sub_val

        ## Order operations with insertion sort
        for i in range(1, len(gate_structure)):
            key_item = gate_structure[i]
            key_value = key_item[1]
            j = i - 1
            while j >= 0 and gate_structure[j][1] > key_value:
                gate_structure[j + 1] = gate_structure[j]
                j -= 1
            gate_structure[j + 1] = key_item

        # Currently there's assumed to only be one gate in this operator, to add validation later

        ## Find target gate in gate_structure
        i = 0
        while gate_structure[i][0] == 'C':
            i += 1
        target_gate_index = i
        U = self.SINGLE_QUBIT_GATES[gate_structure[target_gate_index][0]]

        ## Tensor matrices together to compile an operator
        # Compile control wires on the left side of the target wire
        operation_cursor = target_gate_index
        for i in range(target_gate_index - 1, -1, -1):
            index_difference = gate_structure[operation_cursor][1] - gate_structure[i][1]
            U_0 = np.kron(np.kron(self.KETBRA_00, np.eye(2 ** (index_difference - 1))), np.eye(U.shape[0]))
            U_1 = np.kron(np.kron(self.KETBRA_11, np.eye(2 ** (index_difference - 1))), U)
            U = U_0 + U_1
            operation_cursor -= 1

        # Compile control wires on the right side of the target wire
        operation_cursor = target_gate_index
        for i in range(target_gate_index + 1, len(gate_structure)):
            index_difference = gate_structure[i][1] - gate_structure[operation_cursor][1]
            U_0 = np.kron(np.kron(np.eye(U.shape[0]), np.eye(2 ** (index_difference - 1))), self.KETBRA_00)
            U_1 = np.kron(np.kron(U, np.eye(2 ** (index_difference - 1))), self.KETBRA_11)
            U = U_0 + U_1
            operation_cursor += 1

        return U


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