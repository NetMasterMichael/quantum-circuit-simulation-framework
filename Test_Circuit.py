import unittest
import numpy as np
import Circuit

class Test_Circuit(unittest.TestCase):
    def test_initialise_circuit(self):
        qubits = 1
        testCircuit = Circuit.Circuit(qubits)
        expectedState = np.zeros(2 ** qubits, dtype = complex)
        expectedState[0] = 1 + 0j
        self.assertTrue(np.array_equal(testCircuit.get_circuit_state(), expectedState))