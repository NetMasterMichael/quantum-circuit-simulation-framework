# DSL Syntax

## Single-Qubit Operators
Convention: `{gate}{qubit index}`
- Gate: The gate to use (list of all gates is further down)
- Qubit index: Which wire to apply the gate on
	- For `n` qubits, the minimum index is 0 and the maximum index is `n - 1`
- Examples: 
	- `H0` applies the Hadamard operator on the 0th index, i.e. the first qubit
	- `X3` applies the Pauli X operator on the 3rd index, i.e. the fourth qubit

Multiple gates can be used in one single operator, as shown in the examples below:
- `H0 H1` applies the Hadamard operator to the first two qubits
- `H0 H1 X2 X3` applies the Hadamard operator to the first two qubits and the Pauli X operator to the last two qubits
- `H0 X1 H2 X3` applies the Hadamard operator to the first and third qubits, and the Pauli X operator to the second and fourth qubits

If there are gaps, then these will automatically be filled in with identity `I` gates by the operator compilation process:
- For 4 qubits, the operator `H0` would be parsed to `H0 I1 I2 I3`
- For 4 qubits, the operator `H1 H2` would be parsed to `I0 H1 H2 I3`

If the gates are provided out of order, then these will automatically be ordered by the operator compilation process to prevent duplicates in the cache:
- `H0 H2 H1 H3` → `H0 H1 H2 H3`
- `H3 H2 H1 H0` → `H0 H1 H2 H3`

### List of all single-qubit gates
Implemented
- `I` : Identity
- `H` : Hadamard
- `X` : Pauli X
- `Y` : Pauli Y
- `Z` : Pauli Z

Unimplemented
- `S` or `P` : Phase
- `T` : Pi divided by 8
- `Rx` : Rotation around X axis by theta
	- Alias: `ROTx`
- `Ry` : Rotation around Y axis by theta
	- Alias: `ROTy`
- `Rz` : Rotation around Z axis by theta
	- Alias: `ROTz`

## Multi-Qubit Operators
### Controlled gates
To make a gate controlled by another qubit's output, you can prefix the gate with `C` and append the gate with `,n`, where `n` is the control wire.

Convention: `C{gate}{target wire},{control wire}`
- `C` indicates that the gate is controlled
- Gate: The gate to use (as detailed in the Single-Qubit Operators section)
- Target wire: Which wire to apply the gate on
	- Equivalent to qubit index in the Single-Qubit Operators section
- Control wire: Which wire to check in order to determine if the gate should be applied to the target wire or not
	- If the control wire in a state is a 0, then the state remains unchanged
	- If the control wire in a state is a 1, then the gate is applied to the target wire
- Examples:
	- `CX1,0` applies the Pauli X operator to the second qubit only if the first qubit of a state is equal to 1
	- `CH0,1` applies the Hadamard operator to the first qubit only if the second qubit of a state is equal to 1

If a gate should have multiple control wires, these can be chained together recursively:
- `CCX2,1,0` applies the Pauli X operator on the third qubit only if the first & second qubits of a state are both equal to 1

The following aliases can be used below to improve readability, and will be automatically translated as follows:
- `CNOT` → `CX`
- `CCNOT` → `CCX`
- `TOFF` → `CCX`
- `TOFFOLI` → `CCX`

### SWAP Gate
Placeholder
## Oracles
Placeholder
## Phase Oracles
Placeholder
