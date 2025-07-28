"""
This demo contains the equivalent of a full adder circuit from classical computing, based on the quantum circuit below:

https://commons.wikimedia.org/wiki/File:Quantum_Full_Adder.png

The input wires are as follows:
Wire 0 : A
Wire 1 : B
Wire 2 : Cin (Carry bit in, from another adder)
Wire 3 : |0>

The output wires are as follows:
Wire 0 : A
Wire 1 : B
Wire 2 : S (Sum)
Wire 3 : Cout (Carry bit out, to another adder)
"""

from quantum_circuit import Circuit

# Play around with these variables to change the results
input_A = False
input_B = True
Cin = True

circuit = Circuit(4)

if input_A:
    circuit.apply_operator("X0")

if input_B:
    circuit.apply_operator("X1")

if Cin:
    circuit.apply_operator("X2")

circuit.apply_operator("TOFF3,0,1")
circuit.apply_operator("CNOT1,0")
circuit.apply_operator("TOFF3,1,2")
circuit.apply_operator("CNOT2,1")
circuit.apply_operator("CNOT1,0")

# To print in a more readable way with to_string() function once implemented in Circuit class
print(circuit.DEBUG_get_circuit_state())