import unittest
import numpy as np
import Circuit

# For repeating tests, set the max qubits that the tests will repeat up until
MAX_QUBITS = 8

class Test_Circuit(unittest.TestCase):

    def test_initialise_circuit(self):
        for i in range(1, MAX_QUBITS + 1):
            qubits = i
            testCircuit = Circuit.Circuit(qubits)
            expectedState = np.zeros(2 ** qubits, dtype = complex)
            expectedState[0] = 1 + 0j
            self.assertTrue(np.array_equal(testCircuit.get_circuit_state(), expectedState), 
                msg = f"test_initialise_circuit: Test failed with {i} qubits;\nExpected: {expectedState}\nActual: {testCircuit.get_circuit_state()}")
            print(f"test_initialise_circuit: Test passed with {i} qubits")