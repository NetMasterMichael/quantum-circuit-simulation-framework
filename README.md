# Quantum Circuit Simulation Framework
Welcome to the QCSF repository! This project aims to create a working simulation of a quantum computer with only pure Python and NumPy, based on the circuit model. This is to build upon and put into practice what I learnt while taking the CS3600 Quantum Computation module at Royal Holloway, University of London, and enable myself and others interested in quantum computing to experiment with algorithms hands-on at a small number of qubits, without having access to a real quantum computer.

> Important note: This project does not aim to solve classically difficult problems at quantum running times on a classical computer. Instead, it aims to simulate qubits and prove that quantum algorithms do indeed find correct solutions through quantum mechanics (i.e. lots of linear algebra!).
## Current Features
- Functional quantum circuit model with probabilistic measurement and single qubit gates, including:
  - Identity (**I**) - No effect on the qubit
  - Hadamard (**H**) - Puts a qubit in superposition
  - Pauli X (**X**) - Flips a qubit state
  - Pauli Y (**Y**) - Flips a qubit state and applies a phase flip & rotation into the complex plane
  - Pauli Z (**Z**) - Flips the phase of a qubit in the |1> state
- Domain specific language for constructing operators from string inputs
- Support for running on the CPU or GPU
- Thorough unit testing, to rigorously prove the validity and correctness of the framework
## Planned Features
- Multi Qubit Gates
	- Two-qubit gates, such as CNOT and SWAP gates
	- Three-qubit gates, such as Toffoli and Fredkin gates
	- General n-qubit controlled gates, which can be constructed from scratch and apply any gate for any n control bits
- Oracles and Phase Oracles, with predefined functions and support for user-defined functions
- Quantum Fourier Transforms
- Simulations of several quantum algorithms for solving different problems
	- Deutsch's Algorithm
	- Deutsch-Jozsa Algorithm
	- Bernstein-Vazirani Algorithm
	- Simon's Algorithm
	- Grover's Algorithm
	- Shor's Algorithm
- Optimisations to speed up simulation of circuits
- Extensive documentation on the quantum circuit model and adjacent algorithms
