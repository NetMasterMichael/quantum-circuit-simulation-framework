import unittest
import numpy as np
import Circuit

# For repeating tests, set the max qubits that the tests will repeat up until
MAX_QUBITS = 12

IDENTITY = np.array([[1,0],[0,1]], dtype = complex)
PAULI_X = np.array([[0,1],[1,0]], dtype = complex)
PAULI_Y = np.array([[0,-1j],[1j,0]], dtype = complex)
PAULI_Z = np.array([[1,0],[0,-1]], dtype = complex)
HADAMARD = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype = complex)

SINGLE_QUBIT_GATES = {
    'I': IDENTITY,
    'X': PAULI_X,
    'Y': PAULI_Y,
    'Z': PAULI_Z,
    'H': HADAMARD
}

def generate_ket_0_state(qubits):
    state = np.zeros(2 ** qubits, dtype = complex)
    state[0] = 1 + 0j
    return state

def generate_uniform_single_gate_circuit_DSL(gate: str, qubits: int):
    dsl = gate + "0"
    for i in range(1, qubits):
        dsl += " " + gate + str(i)
    return dsl

class Test_Circuit(unittest.TestCase):

    def test_initialise_circuit(self):
        for i in range(1, MAX_QUBITS + 1):
            qubits = i
            testCircuit = Circuit.Circuit(qubits)
            expectedState = generate_ket_0_state(qubits)
            self.assertTrue(np.array_equal(testCircuit.get_circuit_state(), expectedState), 
                msg = f"test_initialise_circuit: Test failed with {i} qubits;\nExpected: {expectedState}\nActual: {testCircuit.get_circuit_state()}")
            print(f"test_initialise_circuit: Test passed with {i} qubits")

    def test_invalid_qubits(self):
        # Test zero
        self.assertRaises(ValueError, Circuit.Circuit, 0)
        # Test negative
        self.assertRaises(ValueError, Circuit.Circuit, -1)
        print(f"test_invalid_qubits: Test passed")

    def test_single_qubit_gate_start(self):
        for gate in SINGLE_QUBIT_GATES:
            for i in range(1, MAX_QUBITS + 1):
                qubits = i
                testCircuit = Circuit.Circuit(qubits)
                zero_state = generate_ket_0_state(qubits)
                expectedState = zero_state
                U = SINGLE_QUBIT_GATES[gate]
                for j in range(1, i):
                    U = np.kron(U, IDENTITY)
                expectedState = np.dot(U, expectedState)
                testCircuit.apply_operator(gate + "0")
                self.assertTrue(np.array_equal(testCircuit.get_circuit_state(), expectedState), 
                    msg = f"test_single_qubit_gate_start: Test failed with {gate} gate and {i} qubits;\nExpected: {expectedState}\nActual: {testCircuit.get_circuit_state()}")
                print(f"test_single_qubit_gate_start: Test passed with {gate} gate and {i} qubits")
    
    def test_single_qubit_gate_end(self):
        for gate in SINGLE_QUBIT_GATES:
            for i in range(1, MAX_QUBITS + 1):
                qubits = i
                testCircuit = Circuit.Circuit(qubits)
                expectedState = generate_ket_0_state(qubits)
                U = np.array([1], dtype = complex)
                for j in range(0, i - 1):
                    U = np.kron(U, IDENTITY)
                U = np.kron(U, SINGLE_QUBIT_GATES[gate])
                expectedState = np.dot(U, expectedState)
                testCircuit.apply_operator(gate + str(i - 1))
                self.assertTrue(np.array_equal(testCircuit.get_circuit_state(), expectedState), 
                    msg = f"test_single_qubit_gate_end: Test failed with {gate} gate and {i} qubits;\nExpected: {expectedState}\nActual: {testCircuit.get_circuit_state()}")
                print(f"test_single_qubit_gate_end: Test passed with {gate} gate and {i} qubits")

    def test_uniform_single_qubit_gate(self):
        for gate in SINGLE_QUBIT_GATES:
            for i in range(1, MAX_QUBITS + 1):
                qubits = i
                testCircuit = Circuit.Circuit(qubits)
                expectedState = generate_ket_0_state(qubits)
                U = np.array([1], dtype = complex)
                for j in range(0, i):
                    U = np.kron(SINGLE_QUBIT_GATES[gate], U)
                expectedState = np.dot(U, expectedState)
                U_key = generate_uniform_single_gate_circuit_DSL(gate, qubits)
                testCircuit.apply_operator(U_key)
                self.assertTrue(np.array_equal(testCircuit.get_circuit_state(), expectedState), 
                    msg = f"test_uniform_single_qubit_gate: Test failed with {gate} gate and {i} qubits;\nExpected: {expectedState}\nActual: {testCircuit.get_circuit_state()}")
                print(f"test_uniform_single_qubit_gate: Test passed with {gate} gate and {i} qubits")