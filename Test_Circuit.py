import unittest
import numpy as np
import Circuit

# For repeating tests, set the max qubits that the tests will repeat up until
MAX_QUBITS = 8

IDENTITY = np.array([[1,0],[0,1]], dtype = complex)
PAULI_X = np.array([[0,1],[1,0]], dtype = complex)
PAULI_Y = np.array([[0,-1j],[1j,0]], dtype = complex)
PAULI_Z = np.array([[1,0],[0,-1]], dtype = complex)
HADAMARD = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype = complex)

def generate_ket_0_state(qubits):
    state = np.zeros(2 ** qubits, dtype = complex)
    state[0] = 1 + 0j
    return state

def generate_single_gate_circuit_DSL(gate: str, qubits: int):
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

    def test_pauli_X_start(self):
        for i in range(1, MAX_QUBITS + 1):
            qubits = i
            testCircuit = Circuit.Circuit(qubits)
            zero_state = generate_ket_0_state(qubits)
            expectedState = zero_state
            U = PAULI_X
            for j in range(1, i):
                U = np.kron(U, IDENTITY)
            expectedState = np.dot(U, expectedState)
            testCircuit.apply_operator("X0")
            self.assertTrue(np.array_equal(testCircuit.get_circuit_state(), expectedState), 
                msg = f"test_pauli_X_start: Test failed with {i} qubits;\nExpected: {expectedState}\nActual: {testCircuit.get_circuit_state()}")
            print(f"test_pauli_X_start: Test passed with {i} qubits")
    
    def test_pauli_X_end(self):
        for i in range(2, MAX_QUBITS + 1):
            qubits = i
            testCircuit = Circuit.Circuit(qubits)
            expectedState = generate_ket_0_state(qubits)
            U = np.array([1], dtype = complex)
            for j in range(0, i - 1):
                U = np.kron(U, IDENTITY)
            U = np.kron(U, PAULI_X)
            expectedState = np.dot(U, expectedState)
            testCircuit.apply_operator("X" + str(i - 1))
            self.assertTrue(np.array_equal(testCircuit.get_circuit_state(), expectedState), 
                msg = f"test_pauli_X_end: Test failed with {i} qubits;\nExpected: {expectedState}\nActual: {testCircuit.get_circuit_state()}")
            print(f"test_pauli_X_end: Test passed with {i} qubits")

    def test_uniform_pauli_X(self):
        for i in range(1, MAX_QUBITS + 1):
            qubits = i
            testCircuit = Circuit.Circuit(qubits)
            expectedState = generate_ket_0_state(qubits)
            U = np.array([1], dtype = complex)
            for j in range(0, i):
                U = np.kron(PAULI_X, U)
            expectedState = np.dot(U, expectedState)
            U_key = generate_single_gate_circuit_DSL("X", qubits)
            testCircuit.apply_operator(U_key)
            self.assertTrue(np.array_equal(testCircuit.get_circuit_state(), expectedState), 
                msg = f"test_uniform_pauli_X: Test failed with {i} qubits;\nExpected: {expectedState}\nActual: {testCircuit.get_circuit_state()}")
            print(f"test_uniform_pauli_X: Test passed with {i} qubits")

    def test_pauli_Y_start(self):
        for i in range(1, MAX_QUBITS + 1):
            qubits = i
            testCircuit = Circuit.Circuit(qubits)
            zero_state = generate_ket_0_state(qubits)
            expectedState = zero_state
            U = PAULI_Y
            for j in range(1, i):
                U = np.kron(U, IDENTITY)
            expectedState = np.dot(U, expectedState)
            testCircuit.apply_operator("Y0")
            self.assertTrue(np.array_equal(testCircuit.get_circuit_state(), expectedState), 
                msg = f"test_pauli_Y_start: Test failed with {i} qubits;\nExpected: {expectedState}\nActual: {testCircuit.get_circuit_state()}")
            print(f"test_pauli_Y_start: Test passed with {i} qubits")
    
    def test_pauli_Y_end(self):
        for i in range(2, MAX_QUBITS + 1):
            qubits = i
            testCircuit = Circuit.Circuit(qubits)
            expectedState = generate_ket_0_state(qubits)
            U = np.array([1], dtype = complex)
            for j in range(0, i - 1):
                U = np.kron(U, IDENTITY)
            U = np.kron(U, PAULI_Y)
            expectedState = np.dot(U, expectedState)
            testCircuit.apply_operator("Y" + str(i - 1))
            self.assertTrue(np.array_equal(testCircuit.get_circuit_state(), expectedState), 
                msg = f"test_pauli_Y_end: Test failed with {i} qubits;\nExpected: {expectedState}\nActual: {testCircuit.get_circuit_state()}")
            print(f"test_pauli_Y_end: Test passed with {i} qubits")

    def test_uniform_pauli_Y(self):
        for i in range(1, MAX_QUBITS + 1):
            qubits = i
            testCircuit = Circuit.Circuit(qubits)
            expectedState = generate_ket_0_state(qubits)
            U = np.array([1], dtype = complex)
            for j in range(0, i):
                U = np.kron(PAULI_Y, U)
            expectedState = np.dot(U, expectedState)
            U_key = generate_single_gate_circuit_DSL("Y", qubits)
            testCircuit.apply_operator(U_key)
            self.assertTrue(np.array_equal(testCircuit.get_circuit_state(), expectedState), 
                msg = f"test_uniform_pauli_Y: Test failed with {i} qubits;\nExpected: {expectedState}\nActual: {testCircuit.get_circuit_state()}")
            print(f"test_uniform_pauli_Y: Test passed with {i} qubits")
