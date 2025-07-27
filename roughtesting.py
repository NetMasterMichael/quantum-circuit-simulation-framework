from quantum_circuit import Circuit
import numpy as np

test_circuit = Circuit(8)

def save_compiled_controlled_gate(n):
    """
    Compiles the controlled gate for input n and writes the result to a file named `n`.

    Parameters:
    -----------
    n : str or int
        The identifier passed to compile_controlled_gate and used as the output filename.
    """
    # Run the compilation

    print(n)

    result = test_circuit.compile_controlled_gate(n)

    print(result)
    
    # Determine filename (ensure it's a string)
    filename = str(n)
    
    # Write the result to the file
    with open(filename, 'w') as f:
        f.write(str(result))
    
"""
save_compiled_controlled_gate("CX1,0")
save_compiled_controlled_gate("CX0,1")
#save_compiled_controlled_gate("CX1,10")
#save_compiled_controlled_gate("CX10,0")
#save_compiled_controlled_gate("CX10,5")
save_compiled_controlled_gate("CCX2,1,0")
save_compiled_controlled_gate("CCX0,1,2")
save_compiled_controlled_gate("CCX1,2,0")

save_compiled_controlled_gate("CCCH5,1,2,6")
"""

"""
CCX = np.array([[1.+0.j,0.+0.j,0.+0.j,0.+0.j,0.+0.j,0.+0.j,0.+0.j,0.+0.j],
 [0.+0.j,1.+0.j,0.+0.j,0.+0.j,0.+0.j,0.+0.j,0.+0.j,0.+0.j],
 [0.+0.j,0.+0.j,1.+0.j,0.+0.j,0.+0.j,0.+0.j,0.+0.j,0.+0.j],
 [0.+0.j,0.+0.j,0.+0.j,1.+0.j,0.+0.j,0.+0.j,0.+0.j,0.+0.j],
 [0.+0.j,0.+0.j,0.+0.j,0.+0.j,1.+0.j,0.+0.j,0.+0.j,0.+0.j],
 [0.+0.j,0.+0.j,0.+0.j,0.+0.j,0.+0.j,0.+0.j,0.+0.j,1.+0.j],
 [0.+0.j,0.+0.j,0.+0.j,0.+0.j,0.+0.j,0.+0.j,1.+0.j,0.+0.j],
 [0.+0.j,0.+0.j,0.+0.j,0.+0.j,0.+0.j,1.+0.j,0.+0.j,0.+0.j]], dtype = complex)

basis = np.array([1+0j,0+0j,0+0j,0+0j,0+0j,0+0j,0+0j,0+0j], dtype = complex)
X = np.array([[0+0j,1+0j],[1+0j,0+0j]], dtype = complex)
I = np.array([[1+0j,0+0j],[0+0j,1+0j]], dtype = complex)
IIX = np.kron(I, np.kron(I, X))
XII = np.kron(X, np.kron(I, I))
print(basis)
basis = np.dot(IIX, basis)
print(basis)
basis = np.dot(XII, basis)
print(basis)
basis = np.dot(CCX, basis)
print(basis)
"""

test_circuit2 = Circuit(4, DEBUG_syntax_validation=False)
test_circuit2.apply_operator("X3")
test_circuit2.apply_operator("H0 H1 H2 H3")
test_circuit2.apply_operator("CX3,0")
test_circuit2.apply_operator("CX3,1")
test_circuit2.apply_operator("CX3,2")
test_circuit2.apply_operator("H0 H1 H2")
test_circuit2.measure()
print(test_circuit2.DEBUG_get_circuit_state())

test_circuit3 = Circuit(8, DEBUG_syntax_validation=True)
test_circuit3.apply_operator("H0 H1 H2 H3")
test_circuit3.apply_operator("CX4,0")
test_circuit3.apply_operator("CX7,3")
test_circuit3.apply_operator("H0 H1 H2 H3")
test_circuit3.measure()
print(test_circuit3.DEBUG_get_circuit_state())