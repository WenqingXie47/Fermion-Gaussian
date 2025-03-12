from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Operator

# Define the Pauli X1 operator
X1 = Pauli('IX')

# Create the SWAP gate
swap_circuit = QuantumCircuit(2)
swap_circuit.swap(0, 1)

# Get the unitary representation of the SWAP gate
swap_operator = Operator(swap_circuit)

# Evolve the Pauli X1 operator using the SWAP gate
evolved_X1 = X1.evolve(swap_operator)

# Print the result
print("Evolved X1 operator after SWAP:")
print(evolved_X1)
