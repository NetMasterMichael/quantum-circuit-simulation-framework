import unittest
import numpy as np
from Circuit import Circuit

# For repeating tests, set the max qubits that the tests will repeat up until
# A higher MAX_QUBITS value means more rigorous testing, but an increase in 1 results in double the memory usage and 8x testing time
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

def generate_ket_1_state(qubits):
    state = np.zeros(2 ** qubits, dtype = complex)
    state[(2 ** qubits) - 1] = 1 + 0j
    return state

def generate_uniform_single_gate_circuit_DSL(gate: str, qubits: int):
    dsl = gate + "0"
    for i in range(1, qubits):
        dsl += " " + gate + str(i)
    return dsl

def check_if_operator_is_cached(circuit: Circuit, operator_key: str):
    if operator_key in circuit._operator_cache:
        return True
    else:
        return False

class Test_Circuit(unittest.TestCase):

    def test_initialise_circuit(self):
        for i in range(1, MAX_QUBITS + 1):
            qubits = i
            testCircuit = Circuit(qubits)
            expectedState = generate_ket_0_state(qubits)
            self.assertTrue(np.array_equal(testCircuit.get_circuit_state(), expectedState), 
                msg = f"test_initialise_circuit: Test failed with {i} qubits;\nExpected: {expectedState}\nActual: {testCircuit.get_circuit_state()}")
            print(f"test_initialise_circuit: Test passed with {i} qubits")

    def test_invalid_qubits(self):
        # Test zero
        self.assertRaises(ValueError, Circuit, 0)
        # Test negative
        self.assertRaises(ValueError, Circuit, -1)
        print(f"test_invalid_qubits: Test passed")

    def test_invalid_gates(self):
        testCircuit = Circuit(2)
        # Test a syntactically incorrect operator
        self.assertRaises(ValueError, testCircuit.apply_operator, "")
        self.assertRaises(ValueError, testCircuit.apply_operator, "H")
        self.assertRaises(ValueError, testCircuit.apply_operator, "00")
        self.assertRaises(ValueError, testCircuit.apply_operator, "  ")
        self.assertRaises(ValueError, testCircuit.apply_operator, "!!")
        # Test too many gates
        self.assertRaises(ValueError, testCircuit.apply_operator, "H0 H1 H2")
        print(f"test_invalid_gates: Test passed")

    def test_equivalent_cached_operators(self):
        testCircuit = Circuit(4, operator_cache = True)
        # Test ordering
        self.assertFalse(check_if_operator_is_cached(testCircuit, "H0 H1 H2 H3"),
            msg = f"test_equivalent_cached_operators: Test failed at the ordering subtest, assertFalse received True")
        testCircuit.apply_operator("H3 H1 H2 H0")
        self.assertTrue(check_if_operator_is_cached(testCircuit, "H0 H1 H2 H3"),
            msg = f"test_equivalent_cached_operators: Test failed at the ordering subtest, assertTrue received False")
        # Test front padding
        self.assertFalse(check_if_operator_is_cached(testCircuit, "I0 H1 H2 H3"),
            msg = f"test_equivalent_cached_operators: Test failed at the front padding subtest, assertFalse received True")
        testCircuit.apply_operator("H1 H2 H3")
        self.assertTrue(check_if_operator_is_cached(testCircuit, "I0 H1 H2 H3"),
            msg = f"test_equivalent_cached_operators: Test failed at the front padding subtest, assertTrue received False")
        # Test end padding
        self.assertFalse(check_if_operator_is_cached(testCircuit, "H0 I1 I2 I3"),
            msg = f"test_equivalent_cached_operators: Test failed at the end padding subtest, assertFalse received True")
        testCircuit.apply_operator("H0")
        self.assertTrue(check_if_operator_is_cached(testCircuit, "H0 I1 I2 I3"),
            msg = f"test_equivalent_cached_operators: Test failed at the end padding subtest, assertTrue received False")
        # Test front padding and end padding
        self.assertFalse(check_if_operator_is_cached(testCircuit, "I0 H1 I2 I3"),
            msg = f"test_equivalent_cached_operators: Test failed at the front padding and end padding subtest, assertFalse received True")
        testCircuit.apply_operator("H1")
        self.assertTrue(check_if_operator_is_cached(testCircuit, "I0 H1 I2 I3"),
            msg = f"test_equivalent_cached_operators: Test failed at the front padding and end padding subtest, assertTrue received False")
        # Test front padding, end padding, and ordering
        self.assertFalse(check_if_operator_is_cached(testCircuit, "I0 H1 H2 I3"),
            msg = f"test_equivalent_cached_operators: Test failed at the ordering, front padding and end padding subtest, assertFalse received True")
        testCircuit.apply_operator("H2 H1")
        self.assertTrue(check_if_operator_is_cached(testCircuit, "I0 H1 H2 I3"),
            msg = f"test_equivalent_cached_operators: Test failed at the ordering, front padding and end padding subtest, assertTrue received False")
        print(f"test_equivalent_cached_operators: Test passed")

    def test_single_qubit_gate_start(self):
        for gate in SINGLE_QUBIT_GATES:
            for i in range(1, MAX_QUBITS + 1):
                qubits = i
                testCircuit = Circuit(qubits)
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
                testCircuit = Circuit(qubits)
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
                testCircuit = Circuit(qubits)
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

    def test_measurement(self):
        for i in range(1, MAX_QUBITS + 1):
            qubits = i
            testCircuit = Circuit(qubits)
            # testCircuit is initialised as |0..0>, check that we measure zero
            expectedState = generate_ket_0_state(qubits)
            testCircuit.measure()
            self.assertTrue(np.array_equal(testCircuit.get_circuit_state(), expectedState))
            print(f"test_measurement: Test measurement of |{'0' * qubits}> state passed")

            # Apply X⊗n to testCircuit to get the state |1..1>, then check measurement
            expectedState = generate_ket_1_state(qubits)
            testCircuit.apply_operator(generate_uniform_single_gate_circuit_DSL("X", qubits))
            testCircuit.measure()
            self.assertTrue(np.array_equal(testCircuit.get_circuit_state(), expectedState))
            print(f"test_measurement: Test measurement of |{'1' * qubits}> state passed")

            # Revert circuit to |0..0> and apply Y⊗n to get |i..i>, then check that it collapses to |1..1>
            expectedState = generate_ket_1_state(qubits)
            testCircuit.apply_operator(generate_uniform_single_gate_circuit_DSL("X", qubits))
            testCircuit.apply_operator(generate_uniform_single_gate_circuit_DSL("Y", qubits))
            testCircuit.measure()
            self.assertTrue(np.array_equal(testCircuit.get_circuit_state(), expectedState))
            print(f"test_measurement: Test measurement of |{'1' * qubits}> state collapsed from complex plane passed")

    def test_random_measurement_sampling(self):
        for i in range(1, MAX_QUBITS + 1):
            qubits = i
            samples = (2 ** qubits) / 2
            coverage_threshold = 0.375
            test_passed = False
            reruns = 5
            while reruns != 0:
                testCircuit = Circuit(qubits, operator_cache = True)
                seen_states = set()
                for j in range(0, int(samples)):
                    testCircuit.reset_circuit_state()
                    testCircuit.apply_operator(generate_uniform_single_gate_circuit_DSL("H", qubits))
                    testCircuit.measure()
                    collapsed_state = testCircuit.get_circuit_state()
                    for k in range(0, 2 ** qubits):
                        if collapsed_state[k] == 1:
                            seen_states.add(k)
                            break
                coverage = len(seen_states) / len(collapsed_state)
                if coverage >= coverage_threshold:
                    reruns = 0
                    test_passed = True
                else:
                    reruns -= 1
                    print(f"test_random_measurement_sampling: Repeating test for {i} qubits")
            self.assertTrue(test_passed, f"test_random_measurement_sampling: Test failed, coverage threshold of {coverage_threshold} has been passed too many times")
            print(f"test_random_measurement_sampling: Test passed with {i} qubits and {coverage}% coverage")

    def test_partial_superposition_measurement(self):
        for i in range(2, MAX_QUBITS + 1):
            qubits = i
            testCircuit = Circuit(qubits, operator_cache = True)
            repeats = 10
            accepted_0_state = np.zeros(2 ** qubits, dtype = complex)
            accepted_0_state[0] = 1 + 0j
            accepted_1_state = np.zeros(2 ** qubits, dtype = complex)
            accepted_1_state[2 ** (qubits - 1)] = 1 + 0j
            for j in range(0, repeats):
                testCircuit.reset_circuit_state()
                testCircuit.apply_operator("H0")
                testCircuit.measure()
                measuredState = testCircuit.get_circuit_state()
                self.assertTrue(np.array_equal(accepted_0_state, measuredState) | np.array_equal(accepted_1_state, measuredState))
            print(f"test_partial_superposition_measurement: Test passed with {i} qubits")
        